import os
import logging
import time
import numpy as np
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from Evaluate.evaluateMV_Foul import evaluate
import torch
from torch.cuda.amp import GradScaler
from dataset import MultiViewDataset
from train import trainer, evaluation
import torch.nn as nn
import torchvision.transforms as transforms
from model import MVNetwork
from config.classes import EVENT_DICTIONARY, INVERSE_EVENT_DICTIONARY
from torchvision.models.video import R3D_18_Weights, MC3_18_Weights
from torchvision.models.video import R2Plus1D_18_Weights, S3D_Weights
from torchvision.models.video import MViT_V2_S_Weights, MViT_V1_B_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights, mvit_v1_b, MViT_V1_B_Weights


def print_gpu_memory_stats():
    """Print GPU memory usage statistics"""
    if torch.cuda.is_available():
        print("\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(i) / 1024**3  # GB
            allocated = torch.cuda.memory_allocated(i) / 1024**3  # GB
            free = total_memory - reserved
            print(f"  GPU {i}: Total {total_memory:.2f} GB | Reserved {reserved:.2f} GB | Allocated {allocated:.2f} GB | Free {free:.2f} GB")
        print("")


def checkArguments():

    # args.num_views
    if args.num_views > 5 or  args.num_views < 1:
        print("Could not find your desired argument for --args.num_views:")
        print("Possible number of views are: 1, 2, 3, 4, 5")
        exit()

    # args.data_aug
    if args.data_aug != 'Yes' and args.data_aug != 'No':
        print("Could not find your desired argument for --args.data_aug:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.pooling_type
    if args.pooling_type != 'max' and args.pooling_type != 'mean' and args.pooling_type != 'attention' and args.pooling_type != 'perceiver':
        print("Could not find your desired argument for --args.pooling_type:")
        print("Possible arguments are: max, mean, attention, or perceiver")
        exit()
    # args.weighted_loss
    if args.weighted_loss != 'Yes' and args.weighted_loss != 'No':
        print("Could not find your desired argument for --args.weighted_loss:")
        print("Possible arguments are: Yes or No")
        exit()

    # args.start_frame
    if args.start_frame > 124 or  args.start_frame < 0 or args.end_frame - args.start_frame < 2:
        print("Could not find your desired argument for --args.start_frame:")
        print("Choose a number between 0 and 124 and smaller as --args.end_frame")
        exit()

    # args.end_frame
    if args.end_frame < 1 or  args.end_frame > 125:
        print("Could not find your desired argument for --args.end_frame:")
        print("Choose a number between 1 and 125 and greater as --args.start_frame")
        exit()

    # args.fps
    if args.fps > 25 or  args.fps < 1:
        print("Could not find your desired argument for --args.fps:")
        print("Possible number for the fps are between 1 and 25")
        exit()

    # Check for memory optimization arguments
    if hasattr(args, 'use_amp') and args.use_amp and not torch.cuda.is_available():
        print("Warning: AMP requested but CUDA is not available. Disabling AMP.")
        args.use_amp = False


def main(*args):

    if args:
        args = args[0]
        LR = args.LR
        gamma = args.gamma
        step_size = args.step_size
        start_frame = args.start_frame
        end_frame = args.end_frame
        weight_decay = args.weight_decay
        
        model_name = args.model_name
        pre_model = args.pre_model
        num_views = args.num_views
        fps = args.fps
        number_of_frames = int((args.end_frame - args.start_frame) / ((args.end_frame - args.start_frame) / (((args.end_frame - args.start_frame) / 25) * args.fps)))
        batch_size = args.batch_size
        data_aug = args.data_aug
        path = args.path
        pooling_type = args.pooling_type
        weighted_loss = args.weighted_loss
        max_num_worker = args.max_num_worker
        max_epochs = args.max_epochs
        continue_training = args.continue_training
        only_evaluation = args.only_evaluation
        path_to_model_weights = args.path_to_model_weights
        initial_epochs = args.initial_epochs
        
        # Handle memory optimization options
        use_amp = args.use_amp if hasattr(args, 'use_amp') else False
        use_checkpoint = args.use_checkpoint if hasattr(args, 'use_checkpoint') else False
        
        # Apply memory_efficient settings if enabled
        if hasattr(args, 'memory_efficient') and args.memory_efficient:
            use_amp = True
            use_checkpoint = True
            batch_size = min(batch_size, 2)  # Reduce batch size
            num_views = min(num_views, 2)    # Reduce number of views
            print("Memory efficient mode enabled:")
            print(f"  - Using AMP: {use_amp}")
            print(f"  - Using gradient checkpointing: {use_checkpoint}")
            print(f"  - Batch size limited to: {batch_size}")
            print(f"  - Number of views limited to: {num_views}")
            
            # Enable deterministic mode for consistent memory usage
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            print("  - Deterministic mode enabled for consistent memory usage")
    else:
        print("EXIT")
        exit()

    # Print memory stats before model creation
    print_gpu_memory_stats()

    # Logging information
    numeric_level = getattr(logging, 'INFO'.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % 'INFO')

    os.makedirs(os.path.join("models", os.path.join(model_name, os.path.join(str(num_views), os.path.join(pre_model, os.path.join(str(LR),
                            "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + "_G" + str(gamma) + "_Step" + str(step_size)))))), exist_ok=True)

    best_model_path = os.path.join("models", os.path.join(model_name, os.path.join(str(num_views), os.path.join(pre_model, os.path.join(str(LR),
                            "_B" + str(batch_size) + "_F" + str(number_of_frames) + "_S" + "_G" + str(gamma) + "_Step" + str(step_size))))))

    log_path = os.path.join(best_model_path, "logging.log")

    logging.basicConfig(
        level=numeric_level,
        format=
        "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ])

    # Initialize the data augmentation
    if data_aug == 'Yes':
        transformAug = transforms.Compose([
                                          transforms.RandomAffine(degrees=(0, 0), translate=(0.1, 0.1), scale=(0.9, 1)),
                                          transforms.RandomPerspective(distortion_scale=0.3, p=0.5),
                                          transforms.RandomRotation(degrees=5),
                                          transforms.ColorJitter(brightness=0.5, saturation=0.5, contrast=0.5),
                                          transforms.RandomHorizontalFlip()
                                          ])
    else:
        transformAug = None

    if pre_model == "r3d_18":
        transforms_model = R3D_18_Weights.KINETICS400_V1.transforms()        
    elif pre_model == "s3d":
        transforms_model = S3D_Weights.KINETICS400_V1.transforms()       
    elif pre_model == "mc3_18":
        transforms_model = MC3_18_Weights.KINETICS400_V1.transforms()       
    elif pre_model == "r2plus1d_18":
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
    elif pre_model == "mvit_v2_s":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    else:
        transforms_model = R2Plus1D_18_Weights.KINETICS400_V1.transforms()
        print("Warning: Could not find the desired pretrained model")
        print("Possible options are: r3d_18, s3d, mc3_18, mvit_v2_s and r2plus1d_18")
        print("We continue with r2plus1d_18")
    
    if only_evaluation == 0:
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
        transform_model=transforms_model)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 1:
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views=5, 
        transform_model=transforms_model)

        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    elif only_evaluation == 2:
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
        transform_model=transforms_model)
        dataset_Chall = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Chall', num_views=5, 
        transform_model=transforms_model)

        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        chall_loader2 = torch.utils.data.DataLoader(dataset_Chall,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
    else:
        # Create Train Validation and Test datasets
        dataset_Train = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Train',
            num_views=num_views, transform=transformAug, transform_model=transforms_model)
        dataset_Valid2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Valid', num_views=5, 
            transform_model=transforms_model)
        dataset_Test2 = MultiViewDataset(path=path, start=start_frame, end=end_frame, fps=fps, split='Test', num_views=5, 
            transform_model=transforms_model)

        # Create the dataloaders for train validation and test datasets
        train_loader = torch.utils.data.DataLoader(dataset_Train,
            batch_size=batch_size, shuffle=True,
            num_workers=max_num_worker, pin_memory=True)

        val_loader2 = torch.utils.data.DataLoader(dataset_Valid2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)
        
        test_loader2 = torch.utils.data.DataLoader(dataset_Test2,
            batch_size=1, shuffle=False,
            num_workers=max_num_worker, pin_memory=True)

    ###################################
    #       LOADING THE MODEL         #
    ###################################
    model = MVNetwork(
        net_name=pre_model, 
        agr_type=pooling_type,
        use_amp=use_amp,
        use_checkpoint=use_checkpoint
    ).cuda()
    
    # Print memory stats after model creation
    print_gpu_memory_stats()

    # Load pretrained weights if path provided
    if path_to_model_weights != "":
        path_model = os.path.join(path_to_model_weights)
        
        # First freeze the base_network (MVIT) to train only the new components
        if hasattr(model, 'base_network'):
            for param in model.base_network.parameters():
                param.requires_grad = False
            logging.info("MVIT backbone frozen initially - training only new components")
        
        # Load the weights
        if hasattr(model, 'load_pretrained'):
            # Use the specialized method if available
            model.load_pretrained(path_model)
        else:
            # Fall back to standard loading
            load = torch.load(path_model)
            if 'state_dict' in load:
                model.load_state_dict(load['state_dict'], strict=False)
            else:
                model.load_state_dict(load, strict=False)
            logging.info(f"Loaded pretrained weights from {path_model}")

    if only_evaluation == 3:
        # For training mode
        epoch_start = 0
        phase = "standard"  # Default phase name
        
        if weighted_loss == 'Yes':
            criterion_offence_severity = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[0].cuda())
            criterion_action = nn.CrossEntropyLoss(weight=dataset_Train.getWeights()[1].cuda())
            criterion = [criterion_offence_severity, criterion_action]
        else:
            criterion_offence_severity = nn.CrossEntropyLoss()
            criterion_action = nn.CrossEntropyLoss()
            criterion = [criterion_offence_severity, criterion_action]

        # Phase 1: Train with frozen MVIT backbone for initial epochs
        if path_to_model_weights != "" and hasattr(model, 'base_network'):
            phase = "phase1"
            logging.info(f"Phase 1: Training with frozen MVIT backbone for {initial_epochs} epochs")
            
            # Create optimizer for only unfrozen parameters
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LR, 
                betas=(0.9, 0.999), eps=1e-07,
                weight_decay=weight_decay, amsgrad=False
            )
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            
            if continue_training:
                phase1_path = os.path.join(best_model_path, "phase1_model.pth.tar")
                if os.path.exists(phase1_path):
                    load = torch.load(phase1_path)
                    model.load_state_dict(load['state_dict'])
                    optimizer.load_state_dict(load['optimizer'])
                    scheduler.load_state_dict(load['scheduler'])
                    epoch_start = load['epoch']
            
            # Run Phase 1 training if not completed yet
            if epoch_start < initial_epochs:
                trainer(
                    train_loader, val_loader2, test_loader2, 
                    model, optimizer, scheduler, criterion,
                    best_model_path, epoch_start, 
                    model_name=model_name, 
                    path_dataset=path, 
                    max_epochs=initial_epochs,
                    phase=phase,
                    use_amp=use_amp
                )
            
            # Save Phase 1 model
            phase1_model_path = os.path.join(best_model_path, "phase1_model.pth.tar")
            torch.save({
                'epoch': initial_epochs,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            }, phase1_model_path)
            
            # Phase 2: Unfreeze MVIT and continue training
            phase = "phase2"
            logging.info("Phase 2: Unfreezing MVIT backbone for fine-tuning")
            
            # Unfreeze MVIT backbone
            for param in model.base_network.parameters():
                param.requires_grad = True
            
            # Create new optimizer with lower learning rate for fine-tuning
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=LR * 0.1,  # Lower learning rate for fine-tuning 
                betas=(0.9, 0.999), eps=1e-07,
                weight_decay=weight_decay, amsgrad=False
            )
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            epoch_start = initial_epochs  # Start from where Phase 1 ended
            
            # Load Phase 2 checkpoint if continuing training
            if continue_training:
                phase2_path = os.path.join(best_model_path, "phase2_model.pth.tar")
                if os.path.exists(phase2_path):
                    load = torch.load(phase2_path)
                    model.load_state_dict(load['state_dict'])
                    optimizer.load_state_dict(load['optimizer'])
                    scheduler.load_state_dict(load['scheduler'])
                    epoch_start = load['epoch']
            
            # Run Phase 2 training
            trainer(
                train_loader, val_loader2, test_loader2, 
                model, optimizer, scheduler, criterion,
                best_model_path, epoch_start, 
                model_name=model_name, 
                path_dataset=path, 
                max_epochs=max_epochs,
                phase=phase,
                use_amp=use_amp
            )
        else:
            # Standard training without phases (no pretrained weights or no base_network)
            optimizer = torch.optim.AdamW(
                model.parameters(), 
                lr=LR, 
                betas=(0.9, 0.999), eps=1e-07,
                weight_decay=weight_decay, amsgrad=False
            )
            
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
            
            if continue_training:
                path_model = os.path.join(log_path, 'model.pth.tar')
                if os.path.exists(path_model):
                    load = torch.load(path_model)
                    model.load_state_dict(load['state_dict'])
                    optimizer.load_state_dict(load['optimizer'])
                    scheduler.load_state_dict(load['scheduler'])
                    epoch_start = load['epoch']
            
            # Standard training
            trainer(
                train_loader, val_loader2, test_loader2, 
                model, optimizer, scheduler, criterion,
                best_model_path, epoch_start, 
                model_name=model_name, 
                path_dataset=path, 
                max_epochs=max_epochs,
                phase=phase,
                use_amp=use_amp
            )

    # Start training or evaluation
    if only_evaluation == 0:
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
            use_amp=use_amp
        ) 
        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

    elif only_evaluation == 1:
        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
            use_amp=use_amp
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)

    elif only_evaluation == 2:
        prediction_file = evaluation(
            test_loader2,
            model,
            set_name="test",
            use_amp=use_amp
        )

        results = evaluate(os.path.join(path, "Test", "annotations.json"), prediction_file)
        print("TEST")
        print(results)

        prediction_file = evaluation(
            chall_loader2,
            model,
            set_name="chall",
            use_amp=use_amp
        )

        results = evaluate(os.path.join(path, "Chall", "annotations.json"), prediction_file)
        print("CHALL")
        print(results)
        
    return 0



if __name__ == '__main__':

    parser = ArgumentParser(description='my method', formatter_class=ArgumentDefaultsHelpFormatter)    
    parser.add_argument('--path',   required=True, type=str, help='Path to the dataset folder' )
    parser.add_argument('--max_epochs',   required=False, type=int,   default=60,     help='Maximum number of epochs' )
    parser.add_argument('--initial_epochs',   required=False, type=int,   default=10,     help='Number of epochs to train with frozen MVIT' )
    parser.add_argument('--model_name',   required=False, type=str,   default="VARS",     help='named of the model to save' )
    parser.add_argument('--batch_size', required=False, type=int,   default=2,     help='Batch size' )
    parser.add_argument('--LR',       required=False, type=float,   default=1e-04, help='Learning Rate' )
    parser.add_argument('--GPU',        required=False, type=int,   default=-1,     help='ID of the GPU to use' )
    parser.add_argument('--max_num_worker',   required=False, type=int,   default=1, help='number of worker to load data')
    parser.add_argument('--loglevel',   required=False, type=str,   default='INFO', help='logging level')
    parser.add_argument("--continue_training", required=False, action='store_true', help="Continue training")
    parser.add_argument("--num_views", required=False, type=int, default=5, help="Number of views")
    parser.add_argument("--data_aug", required=False, type=str, default="Yes", help="Data augmentation")
    parser.add_argument("--pre_model", required=False, type=str, default="r2plus1d_18", help="Name of the pretrained model")
    parser.add_argument("--pooling_type", required=False, type=str, default="max", help="Which type of pooling should be done")
    parser.add_argument("--weighted_loss", required=False, type=str, default="Yes", help="If the loss should be weighted")
    parser.add_argument("--start_frame", required=False, type=int, default=0, help="The starting frame")
    parser.add_argument("--end_frame", required=False, type=int, default=125, help="The ending frame")
    parser.add_argument("--fps", required=False, type=int, default=25, help="Number of frames per second")
    parser.add_argument("--step_size", required=False, type=int, default=3, help="StepLR parameter")
    parser.add_argument("--gamma", required=False, type=float, default=0.1, help="StepLR parameter")
    parser.add_argument("--weight_decay", required=False, type=float, default=0.001, help="Weight decacy")
    parser.add_argument("--use_hrnet", required=False, action='store_true', help="Use HRNet as secondary feature extractor")
    
    # Memory optimization arguments
    parser.add_argument("--use_amp", action='store_true', help="Use Automatic Mixed Precision")
    parser.add_argument("--use_checkpoint", action='store_true', help="Use gradient checkpointing")
    parser.add_argument("--memory_efficient", action='store_true', help="Enable all memory optimizations")

    parser.add_argument("--only_evaluation", required=False, type=int, default=3, help="Only evaluation, 0 = on test set, 1 = on chall set, 2 = on both sets and 3 = train/valid/test")
    parser.add_argument("--path_to_model_weights", required=False, type=str, default="", help="Path to the model weights")

    args = parser.parse_args()

    if args.pre_model == "mvit_v2_s" or args.pre_model == "mvit":
        transforms_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    
    ## Checking if arguments are valid
    checkArguments()

    # Setup the GPU
    if args.GPU >= 0:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.GPU)


    # Start the main training function
    start=time.time()
    logging.info('Starting main function')
    main(args, False)
    logging.info(f'Total Execution Time is {time.time()-start} seconds')