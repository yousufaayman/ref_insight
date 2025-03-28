import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import MultiViewDataset
from model import MVNetwork
from torchvision.models.video import MViT_V2_S_Weights
from config.classes import INVERSE_EVENT_DICTIONARY
import json
from Evaluate.evaluateMV_Foul import evaluate

def test_model_inference(
    path_to_dataset,
    path_to_model=None,
    use_cuda=True,
    output_dir="./inference_test_results",
    split="Valid"
):
    """
    Test if the model can correctly perform inference on validation data.
    
    Args:
        path_to_dataset: Path to the SoccerNet MVFouls dataset
        path_to_model: Path to a saved model checkpoint (optional)
        use_cuda: Whether to use GPU if available
        output_dir: Directory to save the results
        split: Dataset split to use for testing ('Valid' or 'Test')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup transforms
    transform_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    
    # Load dataset
    print(f"Loading {split} dataset...")
    try:
        dataset = MultiViewDataset(
            path=path_to_dataset,
            start=0,
            end=125,
            fps=25,
            split=split,
            num_views=5,  # Use all views for testing
            transform=None,
            transform_model=transform_model
        )
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=2
        )
        
        print(f"Dataset loaded. {len(dataset)} samples.")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Create model
    print("Creating model...")
    try:
        model = MVNetwork(
            net_name="mvit_v2_s",  # Use MVITv2
            agr_type="perceiver"   # Use Perceiver for aggregation
        ).to(device)
        
        # Load model weights if provided
        if path_to_model and os.path.exists(path_to_model):
            print(f"Loading model weights from {path_to_model}")
            checkpoint = torch.load(path_to_model, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        
        print("Model created successfully.")
    except Exception as e:
        print(f"Error creating model: {e}")
        return False
    
    # Inference
    print("Starting inference test...")
    model.eval()
    
    action_predictions = []
    offence_predictions = []
    action_ground_truth = []
    offence_ground_truth = []
    
    prediction_data = {"Set": split, "Actions": {}}
    
    try:
        with torch.no_grad():
            for targets_offence, targets_action, mvclips, action_ids in tqdm(dataloader):
                # Move data to device
                targets_offence = targets_offence.to(device)
                targets_action = targets_action.to(device)
                mvclips = mvclips.to(device).float()
                
                # Forward pass
                outputs_offence, outputs_action, _ = model(mvclips)
                
                # Get predictions
                pred_offence = torch.argmax(outputs_offence, dim=0 if outputs_offence.dim() == 1 else 1).item()
                pred_action = torch.argmax(outputs_action, dim=0 if outputs_action.dim() == 1 else 1).item()
                
                # Get ground truth
                gt_offence = torch.argmax(targets_offence, dim=0 if targets_offence.dim() == 1 else 1).item()
                gt_action = torch.argmax(targets_action, dim=0 if targets_action.dim() == 1 else 1).item()
                
                # Store predictions
                action_predictions.append(pred_action)
                offence_predictions.append(pred_offence)
                action_ground_truth.append(gt_action)
                offence_ground_truth.append(gt_offence)
                
                # Generate prediction data for SoccerNet evaluation
                action_id = action_ids[0] if isinstance(action_ids, list) else action_ids.item()
                
                # Map predictions to labels
                action_name = INVERSE_EVENT_DICTIONARY["action_class"][pred_action]
                
                offence_info = {}
                if pred_offence == 0:
                    offence_info = {"Offence": "No offence", "Severity": ""}
                elif pred_offence == 1:
                    offence_info = {"Offence": "Offence", "Severity": "1.0"}
                elif pred_offence == 2:
                    offence_info = {"Offence": "Offence", "Severity": "3.0"}
                elif pred_offence == 3:
                    offence_info = {"Offence": "Offence", "Severity": "5.0"}
                
                # Store prediction
                prediction_data["Actions"][str(action_id)] = {
                    "Action class": action_name,
                    **offence_info
                }
        
        print("Inference test completed successfully!")
        
        # Calculate accuracy
        action_accuracy = np.mean(np.array(action_predictions) == np.array(action_ground_truth))
        offence_accuracy = np.mean(np.array(offence_predictions) == np.array(offence_ground_truth))
        overall_accuracy = np.mean((np.array(action_predictions) == np.array(action_ground_truth)) & 
                                 (np.array(offence_predictions) == np.array(offence_ground_truth)))
        
        print(f"Action Classification Accuracy: {action_accuracy:.4f}")
        print(f"Offence Classification Accuracy: {offence_accuracy:.4f}")
        print(f"Overall Accuracy: {overall_accuracy:.4f}")
        
        # Create confusion matrices
        def plot_confusion_matrix(ground_truth, predictions, class_names, title, filename):
            from sklearn.metrics import confusion_matrix
            import seaborn as sns
            
            cm = confusion_matrix(ground_truth, predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
            plt.title(title)
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, filename))
            plt.close()
        
        # Action confusion matrix
        action_classes = [INVERSE_EVENT_DICTIONARY["action_class"][i] for i in range(8)]
        plot_confusion_matrix(
            action_ground_truth,
            action_predictions,
            action_classes,
            "Action Classification Confusion Matrix",
            "action_confusion_matrix.png"
        )
        
        # Offence confusion matrix
        offence_classes = ["No offence", "Offence (1.0)", "Offence (3.0)", "Offence (5.0)"]
        plot_confusion_matrix(
            offence_ground_truth,
            offence_predictions,
            offence_classes,
            "Offence Classification Confusion Matrix",
            "offence_confusion_matrix.png"
        )
        
        # Save predictions for SoccerNet evaluation
        prediction_file = os.path.join(output_dir, f"predictions_{split.lower()}.json")
        with open(prediction_file, "w") as f:
            json.dump(prediction_data, f, indent=2)
        print(f"Predictions saved to {prediction_file}")
        
        # Run SoccerNet evaluation
        try:
            results = evaluate(
                os.path.join(path_to_dataset, split, "annotations.json"),
                prediction_file
            )
            print(f"SoccerNet Evaluation Results: {results}")
            
            # Save results
            with open(os.path.join(output_dir, f"soccernet_results_{split.lower()}.json"), "w") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"Error during SoccerNet evaluation: {e}")
        
        return True
    
    except Exception as e:
        print(f"Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model inference")
    parser.add_argument("--path", required=True, type=str, help="Path to the dataset folder")
    parser.add_argument("--model", default=None, type=str, help="Path to model checkpoint")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--output_dir", default="./inference_test_results", type=str, help="Output directory")
    parser.add_argument("--split", default="Valid", type=str, choices=["Valid", "Test"], help="Dataset split")
    
    args = parser.parse_args()
    
    test_model_inference(
        path_to_dataset=args.path,
        path_to_model=args.model,
        use_cuda=not args.no_cuda,
        output_dir=args.output_dir,
        split=args.split
    )