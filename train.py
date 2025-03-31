import logging
import os
import time
import torch
import gc
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from Evaluate.evaluateMV_Foul import evaluate
from tqdm import tqdm

def trainer(train_loader,
            val_loader2,
            test_loader2,
            model,
            optimizer,
            scheduler,
            criterion,
            best_model_path,
            epoch_start,
            model_name,
            path_dataset,
            max_epochs=1000,
            phase="standard",
            use_amp=False  # Enable mixed precision if specified
            ):
    

    logging.info("start training")
    counter = 0
    
    # Initialize scaler for mixed precision training if needed
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    
    # Keep track of best validation performance
    best_val_acc = 0.0

    for epoch in range(epoch_start, max_epochs):
        
        print(f"Epoch {epoch+1}/{max_epochs}")
    
        # Create a progress bar
        pbar = tqdm(total=len(train_loader), desc=f"Training ({phase})", position=0, leave=True)

        ###################### TRAINING ###################
        prediction_file, loss_action, loss_offence_severity = train(
            train_loader,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=True,
            set_name="train",
            pbar=pbar,
            scaler=scaler,
            use_amp=use_amp
        )

        results = evaluate(os.path.join(path_dataset, "Train", "annotations.json"), prediction_file)
        print("TRAINING")
        print(results)

        ###################### VALIDATION ###################
        prediction_file, loss_action, loss_offence_severity = train(
            val_loader2,
            model,
            criterion,
            optimizer,
            epoch + 1,
            model_name,
            train=False,
            set_name="valid",
            use_amp=use_amp
        )

        results = evaluate(os.path.join(path_dataset, "Valid", "annotations.json"), prediction_file)
        print("VALIDATION")
        print(results)
        
        # Extract validation accuracy for model saving (adjust metric as needed)
        val_acc = results.get('mAP', 0.0)  # Replace with your actual accuracy metric


        ###################### TEST ###################
        prediction_file, loss_action, loss_offence_severity = train(
                test_loader2,
                model,
                criterion,
                optimizer,
                epoch + 1,
                model_name,
                train=False,
                set_name="test",
                use_amp=use_amp
            )

        results = evaluate(os.path.join(path_dataset, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)
        

        scheduler.step()

        counter += 1

        if counter > 3:
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None
            }
            path_aux = os.path.join(best_model_path, f"{phase}_{epoch+1}_model.pth.tar")
            torch.save(state, path_aux)
            
        # Save best model based on validation performance
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            state = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict() if use_amp else None,
                'val_acc': val_acc
            }
            best_model_path_file = os.path.join(best_model_path, f"{phase}_best_model.pth.tar")
            torch.save(state, best_model_path_file)
            print(f"Saved new best model with validation accuracy: {val_acc:.4f}")
        
    pbar.close()    
    return

def train(dataloader,
          model,
          criterion,
          optimizer,
          epoch,
          model_name,
          train=False,
          set_name="train",
          pbar=None,
          scaler=None,
          use_amp=False
        ):
    

    # switch to train mode
    if train:
        model.train()
    else:
        model.eval()

    loss_total_action = 0
    loss_total_offence_severity = 0
    total_loss = 0

    if not os.path.isdir(model_name):
        os.mkdir(model_name) 

    # path where we will save the results
    prediction_file = "predicitions_" + set_name + "_epoch_" + str(epoch) + ".json"
    data = {}
    data["Set"] = set_name

    actions = {}

    # Process all batches
    for batch_idx, (targets_offence_severity, targets_action, mvclips, action) in enumerate(dataloader):
        targets_offence_severity = targets_offence_severity.cuda()
        targets_action = targets_action.cuda()
        mvclips = mvclips.cuda().float()
        
        if pbar is not None:
            pbar.update()

        # Choose appropriate context based on mode and amp setting
        forward_context = torch.cuda.amp.autocast(enabled=use_amp) if use_amp else torch.no_grad() if not train else torch.enable_grad()
            
        # Forward pass with appropriate precision context
        with forward_context:
            # compute output
            outputs_offence_severity, outputs_action, _ = model(mvclips)
            
            # Handle batch size 1 case
            if len(outputs_offence_severity.size()) == 1:
                outputs_offence_severity = outputs_offence_severity.unsqueeze(0)   
            if len(outputs_action.size()) == 1:
                outputs_action = outputs_action.unsqueeze(0)  
   
            # compute the loss
            loss_offence_severity = criterion[0](outputs_offence_severity, targets_offence_severity)
            loss_action = criterion[1](outputs_action, targets_action)
            loss = loss_offence_severity + loss_action

        # Training step with possible mixed precision
        if train:
            # compute gradient and do SGD step
            optimizer.zero_grad()
            
            if use_amp and scaler is not None:
                # Mixed precision backward
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Regular backward
                loss.backward()
                optimizer.step()

        # Generate predictions and add to results
        if len(action) == 1:
            preds_sev = torch.argmax(outputs_offence_severity, 0)
            preds_act = torch.argmax(outputs_action, 0)

            values = {}
            values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
            if preds_sev.item() == 0:
                values["Offence"] = "No offence"
                values["Severity"] = ""
            elif preds_sev.item() == 1:
                values["Offence"] = "Offence"
                values["Severity"] = "1.0"
            elif preds_sev.item() == 2:
                values["Offence"] = "Offence"
                values["Severity"] = "3.0"
            elif preds_sev.item() == 3:
                values["Offence"] = "Offence"
                values["Severity"] = "5.0"
            actions[action[0]] = values       
        else:
            preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
            preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

            for i in range(len(action)):
                values = {}
                values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                if preds_sev[i].item() == 0:
                    values["Offence"] = "No offence"
                    values["Severity"] = ""
                elif preds_sev[i].item() == 1:
                    values["Offence"] = "Offence"
                    values["Severity"] = "1.0"
                elif preds_sev[i].item() == 2:
                    values["Offence"] = "Offence"
                    values["Severity"] = "3.0"
                elif preds_sev[i].item() == 3:
                    values["Offence"] = "Offence"
                    values["Severity"] = "5.0"
                actions[action[i]] = values       

        # Update loss tracking
        loss_total_action += float(loss_action)
        loss_total_offence_severity += float(loss_offence_severity)
        total_loss += 1
          
        # Clear memory periodically to avoid fragmentation
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save results
    data["Actions"] = actions
    with open(os.path.join(model_name, prediction_file), "w") as outfile: 
        json.dump(data, outfile)  
    return os.path.join(model_name, prediction_file), loss_total_action / total_loss, loss_total_offence_severity / total_loss


def evaluation(dataloader,
          model,
          set_name="test",
          use_amp=False
        ):
    
    model.eval()

    prediction_file = "predicitions_" + set_name + ".json"
    data = {}
    data["Set"] = set_name
    actions = {}
    
    # Create a progress bar for evaluation
    pbar = tqdm(total=len(dataloader), desc=f"Evaluating ({set_name})", position=0, leave=True)
           
    # Use torch.no_grad to save memory during evaluation
    with torch.no_grad():
        for batch_idx, (_, _, mvclips, action) in enumerate(dataloader):
            # Update progress bar
            pbar.update(1)
            
            # Process with mixed precision if enabled
            with torch.cuda.amp.autocast(enabled=use_amp):
                mvclips = mvclips.cuda().float()
                outputs_offence_severity, outputs_action, _ = model(mvclips)

                # Generate predictions
                if len(action) == 1:
                    preds_sev = torch.argmax(outputs_offence_severity, 0)
                    preds_act = torch.argmax(outputs_action, 0)

                    values = {}
                    values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act.item()]
                    if preds_sev.item() == 0:
                        values["Offence"] = "No offence"
                        values["Severity"] = ""
                    elif preds_sev.item() == 1:
                        values["Offence"] = "Offence"
                        values["Severity"] = "1.0"
                    elif preds_sev.item() == 2:
                        values["Offence"] = "Offence"
                        values["Severity"] = "3.0"
                    elif preds_sev.item() == 3:
                        values["Offence"] = "Offence"
                        values["Severity"] = "5.0"
                    actions[action[0]] = values       
                else:
                    preds_sev = torch.argmax(outputs_offence_severity.detach().cpu(), 1)
                    preds_act = torch.argmax(outputs_action.detach().cpu(), 1)

                    for i in range(len(action)):
                        values = {}
                        values["Action class"] = INVERSE_EVENT_DICTIONARY["action_class"][preds_act[i].item()]
                        if preds_sev[i].item() == 0:
                            values["Offence"] = "No offence"
                            values["Severity"] = ""
                        elif preds_sev[i].item() == 1:
                            values["Offence"] = "Offence"
                            values["Severity"] = "1.0"
                        elif preds_sev[i].item() == 2:
                            values["Offence"] = "Offence"
                            values["Severity"] = "3.0"
                        elif preds_sev[i].item() == 3:
                            values["Offence"] = "Offence"
                            values["Severity"] = "5.0"
                        actions[action[i]] = values                    

            # Periodically clear cache
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()

    # Close progress bar
    pbar.close()
    
    # Final cleanup
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save results
    data["Actions"] = actions
    with open(prediction_file, "w") as outfile: 
        json.dump(data, outfile)  
    return prediction_file