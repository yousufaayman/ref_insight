# In model.py
import torch
from torch import nn
import timm
from transformers import PerceiverModel, PerceiverConfig
from mvaggregate import MVAggregate
from torchvision.models.video import r3d_18, R3D_18_Weights, MC3_18_Weights, mc3_18
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights, s3d, S3D_Weights
from torchvision.models.video import mvit_v2_s, MViT_V2_S_Weights
from utils import batch_tensor, unbatch_tensor

def load_pretrained_weights(model, pretrained_path, verbose=True):
    """
    Load pretrained weights from previous AI_VAR model to the enhanced model
    Handles potential architecture differences between MVIT and MVITv2
    """
    import os
    
    if not os.path.exists(pretrained_path):
        if verbose:
            print(f"Pretrained weights not found at {pretrained_path}")
        return False
    
    if verbose:
        print(f"Loading pretrained weights from {pretrained_path}")
    
    # Load pretrained weights
    pretrained_weights = torch.load(pretrained_path, map_location='cpu')
    
    # If it's a checkpoint with 'state_dict' key
    if 'state_dict' in pretrained_weights:
        pretrained_weights = pretrained_weights['state_dict']
    
    # Create a new state dict for the current model
    model_state_dict = model.state_dict()
    
    # Count loaded parameters
    loaded_params = 0
    mismatched_params = 0
    total_params = len(model_state_dict)
    
    # Keep track of parameter shapes for debugging
    shape_mismatches = []
    
    # Maps for converting parameter names from old to new model
    old_to_new_prefixes = {
        'mvnetwork.model.': 'base_network.',
        'base_network.': 'base_network.'
    }
    
    if verbose:
        print("Starting weight transfer...")
    
    # Try to load matching parameters
    for name, param in pretrained_weights.items():
        loaded = False
        
        # Try different prefix mappings
        for old_prefix, new_prefix in old_to_new_prefixes.items():
            if name.startswith(old_prefix):
                new_name = name.replace(old_prefix, new_prefix)
                
                # Check if parameter exists in new model
                if new_name in model_state_dict:
                    # Check if shapes match
                    if model_state_dict[new_name].shape == param.shape:
                        model_state_dict[new_name] = param
                        loaded_params += 1
                        loaded = True
                        break
                    else:
                        shape_mismatches.append((new_name, model_state_dict[new_name].shape, param.shape))
                        mismatched_params += 1
        
        # Also try direct name match
        if not loaded and name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name] = param
                loaded_params += 1
    
    # Load the updated state dict
    model.load_state_dict(model_state_dict, strict=False)
    
    if verbose:
        print(f"Loaded {loaded_params}/{total_params} parameters from pretrained model")
        print(f"Found {mismatched_params} parameters with shape mismatches")
        
        if shape_mismatches and verbose:
            print("\nDetails of shape mismatches (showing up to 10):")
            for i, (name, new_shape, old_shape) in enumerate(shape_mismatches[:10]):
                print(f"  {name}: model shape {new_shape}, pretrained shape {old_shape}")
            
            if len(shape_mismatches) > 10:
                print(f"  ...and {len(shape_mismatches) - 10} more mismatches")
    
    return True

# Add this new class for HRNet feature extraction
class HRNetFeatureExtractor(nn.Module):
    """
    HRNet-based feature extractor for each video frame
    """
    def __init__(self, model_name='hrnet_w48', pretrained=True):
        super().__init__()
        # Load HRNet from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[4]  # Use the highest resolution feature map
        )
        
        # Use AdaptiveAvgPool for variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension of hrnet_w48
        self.feature_dim = 2048
        
    def forward(self, x):
        """
        Args:
            x: Video tensor of shape (B, C, T, H, W)
        Returns:
            Features of shape (B, T, feature_dim)
        """
        B, C, T, H, W = x.shape
        
        # Reshape input to process each frame individually
        x = x.transpose(1, 2).contiguous()  # (B, T, C, H, W)
        x = x.view(B * T, C, H, W)  # (B*T, C, H, W)
        
        # Extract features for each frame
        features = self.backbone(x)[0]  # Using only the last output from features_only=True
        
        # Pool spatial dimensions
        features = self.pool(features).squeeze(-1).squeeze(-1)  # (B*T, feature_dim)
        
        # Reshape back to batch and time dimensions
        features = features.view(B, T, -1)  # (B, T, feature_dim)
        
        # Return temporal average pooling
        return features.mean(dim=1)  # (B, feature_dim)

# Modify the MVNetwork class to add our new components
class MVNetwork(torch.nn.Module):
    def __init__(self, net_name='mvit_v2_s', agr_type='perceiver', lifting_net=torch.nn.Sequential()):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        
        self.feat_dim = 512
        self.mvit_feat_dim = 400  # For MVITv2
        self.hrnet_feat_dim = 2048  # For HRNet

        # Primary video feature extractor
        if net_name == "mvit_v2_s" or net_name == "mvit":
            weights_model = MViT_V2_S_Weights.DEFAULT
            network = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "r3d_18":
            weights_model = R3D_18_Weights.DEFAULT
            network = r3d_18(weights=weights_model)
        elif net_name == "s3d":
            weights_model = S3D_Weights.DEFAULT
            network = s3d(weights=weights_model)
            self.feat_dim = 400
        elif net_name == "mc3_18":
            weights_model = MC3_18_Weights.DEFAULT
            network = mc3_18(weights=weights_model)
        elif net_name == "r2plus1d_18":
            weights_model = R2Plus1D_18_Weights.DEFAULT
            network = r2plus1d_18(weights=weights_model)
        else:
            # Default to MVITv2
            weights_model = MViT_V2_S_Weights.DEFAULT
            network = mvit_v2_s(weights=weights_model)
            self.feat_dim = 400
                
        network.fc = torch.nn.Sequential()  # Remove classification head
        
        # Secondary feature extractor (HRNet)
        self.hrnet = HRNetFeatureExtractor(model_name='hrnet_w48', pretrained=True)
        
        # Feature fusion layer
        self.fusion = nn.Sequential(
            nn.LayerNorm(self.mvit_feat_dim + self.hrnet_feat_dim),
            nn.Linear(self.mvit_feat_dim + self.hrnet_feat_dim, self.feat_dim),
            nn.ReLU(),
            nn.LayerNorm(self.feat_dim)
        )

        # Use perceiver or traditional aggregation
        if self.agr_type == "perceiver":
            # Define PerceiverIO configuration for multi-view aggregation
            num_latents = 32
            num_heads = 8
            
            self.perceiver_config = PerceiverConfig(
                d_model=self.feat_dim,
                num_latents=num_latents,
                d_latents=self.feat_dim,
                num_self_attends_per_block=4,
                num_cross_attention_heads=num_heads,
                num_self_attention_heads=num_heads,
                cross_attention_shape_for_attention="kv",
                self_attention_widening_factor=1,
                cross_attention_widening_factor=1,
            )
            
            self.perceiver = PerceiverModel(self.perceiver_config)
            self.output_proj = nn.Linear(self.feat_dim, self.feat_dim)
            
            # Classification heads
            self.fc_offence = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, 4)
            )
            
            self.fc_action = nn.Sequential(
                nn.LayerNorm(self.feat_dim),
                nn.Linear(self.feat_dim, self.feat_dim),
                nn.ReLU(),
                nn.Linear(self.feat_dim, 8)
            )
        else:
            # Use traditional aggregation
            self.mvnetwork = MVAggregate(
                model=network,
                agr_type=self.agr_type, 
                feat_dim=self.feat_dim, 
                lifting_net=self.lifting_net,
            )
        
        # Store the base network for feature extraction
        self.base_network = network

    # In model.py - Update the preprocess_for_mvit method
    def preprocess_for_mvit(self, mvimages):
        """Preprocess for MVITv2 with safer tensor operations"""
        B, V, C, T, H, W = mvimages.shape
        
        # Subsample frames if needed - MVITv2 typically works with 16 frames
        if T > 16:
            indices = torch.linspace(0, T-1, 16).long().to(mvimages.device)
            mvimages = mvimages[:, :, :, indices]
            _, _, _, T, _, _ = mvimages.shape  # Update T
        
        # Reshape differently to ensure correct shape
        mv_flat = mvimages.reshape(B * V, C, T, H, W)
        
        return mv_flat
    
    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
        
        if self.agr_type == "perceiver":
            # Preprocess input for MVITv2
            mv_flat = self.preprocess_for_mvit(mvimages)
            
            # Extract features
            mvit_features = self.base_network(mv_flat)  # (B*V, mvit_feat_dim)
            
            # Reshape back to batch and view dimensions
            mvit_features = mvit_features.view(B, V, -1)  # (B, V, mvit_feat_dim)
            
            # For HRNet, process each frame and then do temporal averaging
            hrnet_features = torch.zeros(B, V, self.hrnet_feat_dim, device=mvimages.device)
            for b in range(B):
                for v in range(V):
                    # Process with HRNet
                    hrnet_features[b, v] = self.hrnet(mvimages[b, v].unsqueeze(0)).squeeze(0)
            
            # Fuse features
            fused_features = torch.zeros(B, V, self.feat_dim, device=mvimages.device)
            for v in range(V):
                # Concatenate features from both networks
                combined = torch.cat([mvit_features[:, v], hrnet_features[:, v]], dim=-1)
                # Apply fusion
                fused_features[:, v] = self.fusion(combined)
            
            # Use Perceiver for multi-view aggregation
            outputs = self.perceiver(inputs=fused_features).last_hidden_state
            
            # Average the latent outputs
            pooled = outputs.mean(dim=1)
            
            # Project to output dimension
            aggregated = self.output_proj(pooled)
            
            # Apply classification heads
            pred_offence_severity = self.fc_offence(aggregated)
            pred_action = self.fc_action(aggregated)
            
            # Return attention weights for visualization
            attention_weights = torch.ones(B, V, device=mvimages.device)  # Placeholder
            
            return pred_offence_severity, pred_action, attention_weights
        else:
            # Use traditional aggregation
            return self.mvnetwork(mvimages)
    
    def load_pretrained(self, pretrained_path, freeze_mvit=True):
        """
        Load pretrained weights and optionally freeze MVIT layers
        """
        success = load_pretrained_weights(self, pretrained_path)
        
        if freeze_mvit and success:
            # Freeze MVIT weights initially to let new components adapt
            for name, param in self.base_network.named_parameters():
                param.requires_grad = False
            print("MVIT weights frozen. Only new components will be trained initially.")
        
        return success