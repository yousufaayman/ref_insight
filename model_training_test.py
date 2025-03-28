import os
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import json
import traceback
from dataset import MultiViewDataset
from model import MVNetwork
from torchvision.models.video import MViT_V2_S_Weights
import torchvision.transforms as transforms

def inspect_dataset_structure(path_to_dataset):
    """
    Inspect the dataset structure to help diagnose loading issues
    """
    print("\n=== DATASET INSPECTION ===")
    
    # Check directory structure
    print("\nChecking directory structure...")
    for split in ['Train', 'Valid', 'Test', 'Chall']:
        split_path = os.path.join(path_to_dataset, split)
        if os.path.exists(split_path):
            print(f"✓ Found {split} directory at {split_path}")
            
            # Check for annotations file
            ann_path = os.path.join(split_path, "annotations.json")
            if os.path.exists(ann_path):
                print(f"  ✓ Found annotations.json in {split} directory")
                
                # Try to read the annotations file
                try:
                    with open(ann_path, 'r') as f:
                        data = json.load(f)
                    print(f"  ✓ Successfully loaded {split} annotations.json")
                    
                    # Check expected keys
                    if "Set" in data:
                        print(f"  ✓ Found 'Set' key with value: {data['Set']}")
                    else:
                        print(f"  ✗ Missing 'Set' key in {split} annotations.json")
                        
                    if "Actions" in data:
                        print(f"  ✓ Found 'Actions' key with {len(data['Actions'])} actions")
                        # Check a sample action
                        if len(data["Actions"]) > 0:
                            sample_key = list(data["Actions"].keys())[0]
                            sample_action = data["Actions"][sample_key]
                            print(f"  ✓ Sample action keys: {list(sample_action.keys())}")
                    else:
                        print(f"  ✗ Missing 'Actions' key in {split} annotations.json")
                        
                except Exception as e:
                    print(f"  ✗ Error reading {split} annotations.json: {e}")
            else:
                print(f"  ✗ Missing annotations.json in {split} directory")
                
            # Check for action directories
            action_dirs = [d for d in os.listdir(split_path) if d.startswith('action_') and os.path.isdir(os.path.join(split_path, d))]
            if action_dirs:
                print(f"  ✓ Found {len(action_dirs)} action directories")
                
                # Check sample action directory
                sample_dir = os.path.join(split_path, action_dirs[0])
                clip_files = [f for f in os.listdir(sample_dir) if f.startswith('clip_') and f.endswith('.mp4')]
                if clip_files:
                    print(f"  ✓ Found {len(clip_files)} clip files in sample action directory")
                else:
                    print(f"  ✗ No clip files found in sample action directory {sample_dir}")
            else:
                print(f"  ✗ No action directories found in {split} directory")
        else:
            print(f"✗ Missing {split} directory")
    
    print("\n=== END DATASET INSPECTION ===\n")

def test_model_training(
    path_to_dataset,
    batch_size=2,
    num_epochs=2,
    learning_rate=1e-4,
    use_cuda=True,
    output_dir="./training_test_results"
):
    """
    Test if the model trains properly by running a few epochs on a small subset of data
    and plotting the loss curves.
    
    Args:
        path_to_dataset: Path to the SoccerNet MVFouls dataset
        batch_size: Batch size for training
        num_epochs: Number of epochs to train
        learning_rate: Learning rate for optimizer
        use_cuda: Whether to use GPU if available
        output_dir: Directory to save the results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # First, inspect the dataset structure
    inspect_dataset_structure(path_to_dataset)
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup transforms
    transform_aug = transforms.Compose([
        transforms.RandomHorizontalFlip()
    ])
    
    transform_model = MViT_V2_S_Weights.KINETICS400_V1.transforms()
    
    # Load a small subset of the training data (first 20 samples)
    print("Loading dataset...")
    try:
        # Check if classes.py exists and EVENT_DICTIONARY is defined
        try:
            from config.classes import EVENT_DICTIONARY
            print(f"✓ Successfully imported EVENT_DICTIONARY from config.classes")
            print(f"  EVENT_DICTIONARY keys: {list(EVENT_DICTIONARY.keys())}")
            if "action_class" in EVENT_DICTIONARY:
                print(f"  action_class mappings: {EVENT_DICTIONARY['action_class']}")
        except Exception as e:
            print(f"✗ Error importing EVENT_DICTIONARY: {e}")
            print("  This will likely cause issues in data_loader.py")
        
        # Try to load dataset with detailed error tracing
        print("\nTrying to create MultiViewDataset...")
        try:
            dataset = MultiViewDataset(
                path=path_to_dataset,
                start=0,
                end=125,
                fps=25,
                split='Train',
                num_views=2,  # Use 2 views for faster training
                transform=transform_aug,
                transform_model=transform_model
            )
            print("✓ Dataset created successfully!")
        except Exception as e:
            print(f"✗ Error creating dataset: {e}")
            print("\nDetailed traceback:")
            traceback.print_exc()
            
            # Try to debug the label2vectormerge function in data_loader
            print("\nDebug label2vectormerge function:")
            try:
                from data_loader import label2vectormerge
                
                print("Trying to run label2vectormerge...")
                try:
                    labels_offence_severity, labels_action, distribution_offence_severity, distribution_action, not_taking, number_of_actions = label2vectormerge(
                        path_to_dataset, 'Train', 2
                    )
                    print("✓ label2vectormerge executed successfully")
                except Exception as e:
                    print(f"✗ Error in label2vectormerge: {e}")
                    print("\nDetailed traceback:")
                    traceback.print_exc()
            except Exception as e:
                print(f"✗ Error importing label2vectormerge: {e}")
            
            print("\nExiting test_model_training due to dataset creation error")
            return False
        
        # Create a small subset
        subset_size = min(20, len(dataset))
        indices = list(range(subset_size))
        subset = torch.utils.data.Subset(dataset, indices)
        
        dataloader = torch.utils.data.DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        print(f"Dataset loaded. Using {subset_size} samples for testing.")
    except Exception as e:
        print(f"Error setting up dataloader: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        return False
    
    # Create model
    print("\nCreating model...")
    try:
        model = MVNetwork(
            net_name="mvit_v2_s",  # Use MVITv2
            agr_type="perceiver"   # Use Perceiver for aggregation
        ).to(device)
        print("✓ Model created successfully.")
    except Exception as e:
        print(f"✗ Error creating model: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        
        # If the error is about "perceiver" agr_type
        if "perceiver" in str(e):
            print("\nThe error might be related to the 'perceiver' aggregation type.")
            print("Make sure you've implemented the PerceiverIO aggregation in model.py.")
            print("Try running with a standard aggregation type first to verify the dataset works:")
            print("Example: model = MVNetwork(net_name='mvit_v2_s', agr_type='max')")
        
        return False
    
    # Create loss function and optimizer
    criterion_offence = nn.CrossEntropyLoss()
    criterion_action = nn.CrossEntropyLoss()
    criterions = [criterion_offence, criterion_action]
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    print("\nStarting training test...")
    training_losses = []
    offence_losses = []
    action_losses = []
    
    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            running_offence_loss = 0.0
            running_action_loss = 0.0
            
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for i, (targets_offence, targets_action, mvclips, _) in enumerate(progress_bar):
                # Print shape information for debugging
                if i == 0:
                    print(f"\nInput shapes:")
                    print(f"  targets_offence: {targets_offence.shape}")
                    print(f"  targets_action: {targets_action.shape}")
                    print(f"  mvclips: {mvclips.shape}")
                
                # Move data to device
                targets_offence = targets_offence.to(device)
                targets_action = targets_action.to(device)
                mvclips = mvclips.to(device).float()
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                try:
                    outputs_offence, outputs_action, _ = model(mvclips)
                    
                    # Print output shapes for debugging
                    if i == 0:
                        print(f"\nOutput shapes:")
                        print(f"  outputs_offence: {outputs_offence.shape}")
                        print(f"  outputs_action: {outputs_action.shape}")
                except Exception as e:
                    print(f"\n✗ Error during forward pass: {e}")
                    print(f"  mvclips shape: {mvclips.shape}")
                    print("\nDetailed traceback:")
                    traceback.print_exc()
                    return False
                
                # Calculate loss
                try:
                    loss_offence = criterions[0](outputs_offence, targets_offence)
                    loss_action = criterions[1](outputs_action, targets_action)
                    loss = loss_offence + loss_action
                except Exception as e:
                    print(f"\n✗ Error calculating loss: {e}")
                    print(f"  outputs_offence shape: {outputs_offence.shape}")
                    print(f"  targets_offence shape: {targets_offence.shape}")
                    print(f"  outputs_action shape: {outputs_action.shape}")
                    print(f"  targets_action shape: {targets_action.shape}")
                    print("\nDetailed traceback:")
                    traceback.print_exc()
                    return False
                
                # Backward pass and optimize
                try:
                    loss.backward()
                    optimizer.step()
                except Exception as e:
                    print(f"\n✗ Error during backward pass: {e}")
                    print("\nDetailed traceback:")
                    traceback.print_exc()
                    return False
                
                # Update statistics
                running_loss += loss.item()
                running_offence_loss += loss_offence.item()
                running_action_loss += loss_action.item()
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': running_loss / (i + 1),
                    'offence_loss': running_offence_loss / (i + 1),
                    'action_loss': running_action_loss / (i + 1)
                })
            
            # Calculate average loss for the epoch
            epoch_loss = running_loss / len(dataloader)
            epoch_offence_loss = running_offence_loss / len(dataloader)
            epoch_action_loss = running_action_loss / len(dataloader)
            
            print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Offence Loss: {epoch_offence_loss:.4f}, Action Loss: {epoch_action_loss:.4f}")
            
            # Store losses for plotting
            training_losses.append(epoch_loss)
            offence_losses.append(epoch_offence_loss)
            action_losses.append(epoch_action_loss)
        
        print("\n✓ Training test completed successfully!")
        
        # Save model checkpoint
        checkpoint_path = os.path.join(output_dir, "test_model_checkpoint.pth")
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'training_losses': training_losses,
            'offence_losses': offence_losses,
            'action_losses': action_losses
        }, checkpoint_path)
        print(f"✓ Model checkpoint saved to {checkpoint_path}")
        
        # Plot and save loss curves
        plt.figure(figsize=(12, 6))
        epochs = range(1, num_epochs + 1)
        
        plt.subplot(1, 3, 1)
        plt.plot(epochs, training_losses, 'b-', label='Total Loss')
        plt.title('Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 2)
        plt.plot(epochs, offence_losses, 'r-', label='Offence Loss')
        plt.title('Offence Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 3, 3)
        plt.plot(epochs, action_losses, 'g-', label='Action Loss')
        plt.title('Action Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "training_loss_curves.png")
        plt.savefig(plot_path)
        print(f"✓ Loss curves saved to {plot_path}")
        
        # Print model statistics
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        
        return True
    
    except Exception as e:
        print(f"\n✗ Unexpected error during training: {e}")
        print("\nDetailed traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test model training")
    parser.add_argument("--path", required=True, type=str, help="Path to the dataset folder")
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size for training")
    parser.add_argument("--num_epochs", default=2, type=int, help="Number of epochs to train")
    parser.add_argument("--lr", default=1e-4, type=float, help="Learning rate")
    parser.add_argument("--no_cuda", action="store_true", help="Disable CUDA")
    parser.add_argument("--output_dir", default="./training_test_results", type=str, help="Output directory")
    parser.add_argument("--agr_type", default="perceiver", type=str, choices=["max", "mean", "attention", "perceiver"], 
                        help="Aggregation type for multi-view features")
    
    args = parser.parse_args()
    
    test_model_training(
        path_to_dataset=args.path,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.lr,
        use_cuda=not args.no_cuda,
        output_dir=args.output_dir
    )