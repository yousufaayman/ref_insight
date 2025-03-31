# In model.py
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint
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
    Memory-optimized HRNet feature extractor
    """
    def __init__(self, model_name='hrnet_w32', pretrained=True):
        super().__init__()
        # Load smaller HRNet model for efficiency
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[4]  # Use the highest resolution feature map
        )
        
        # Use AdaptiveAvgPool for variable input sizes
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Feature dimension of hrnet_w32 (smaller than w48)
        self.feature_dim = 1024 if 'w32' in model_name else 2048
        
    def forward(self, x):
        """
        Modified forward pass with memory optimization
        Only processes middle frame for video inputs
        """
        # Check if input is video or image
        if x.dim() == 5:  # Video input (B, C, T, H, W)
            B, C, T, H, W = x.shape
            
            # For video, just use the middle frame for spatial features
            mid_frame_idx = T // 2
            x = x[:, :, mid_frame_idx]  # (B, C, H, W)
        
        # Extract features with gradient checkpointing
        features = checkpoint.checkpoint(self._extract_features, x)
        
        return features  # (B, feature_dim)
    
    def _extract_features(self, x):
        """Separate function for checkpointing"""
        features = self.backbone(x)[0]
        features = self.pool(features).squeeze(-1).squeeze(-1)
        return features


# Modify the MVNetwork class to add our new components
class MVNetwork(torch.nn.Module):
    def __init__(self, net_name='mvit_v2_s', agr_type='attention', lifting_net=torch.nn.Sequential(), use_amp=True, use_checkpoint=True):
        super().__init__()

        self.net_name = net_name
        self.agr_type = agr_type
        self.lifting_net = lifting_net
        self.use_amp = use_amp  # Flag for mixed precision
        self.use_checkpoint = use_checkpoint  # Flag for gradient checkpointing
        
        self.feat_dim = 512
        self.mvit_feat_dim = 400  # For MVITv2
        self.hrnet_feat_dim = 1024  # For HRNet w32 (smaller model)

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
        
        # Apply gradient checkpointing to the backbone model
        if self.use_checkpoint and net_name == "mvit_v2_s":
            # Enable gradient checkpointing for MViT blocks
            for i, block in enumerate(network.blocks):
                network.blocks[i]._forward = block.forward
                network.blocks[i].forward = lambda x, thw, block=block: checkpoint.checkpoint(block._forward, x, thw)
        
        # Secondary feature extractor (HRNet)
        self.hrnet = HRNetFeatureExtractor(model_name='hrnet_w32', pretrained=True)
        
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
                use_checkpoint=self.use_checkpoint
            )
        
        # Store the base network for feature extraction
        self.base_network = network

    # Modified preprocess method to handle tensor shape issues
    def preprocess_for_mvit(self, mvimages):
        """Safer preprocessing for MVITv2"""
        B, V, C, T, H, W = mvimages.shape
        
        # Subsample frames if needed - MVITv2 typically works with 16 frames
        if T > 16:
            # Create indices for frame sampling
            step = T / 16
            indices = [min(int(i * step), T-1) for i in range(16)]
            indices = torch.tensor(indices, device=mvimages.device)
            # Sample frames
            mvimages = mvimages[:, :, :, indices]
            T = 16  # Update T after subsampling
        
        # Reshape to batch all views together
        mv_flat = mvimages.reshape(B * V, C, T, H, W)
        
        return mv_flat
    
    def forward(self, mvimages):
        B, V, C, T, H, W = mvimages.shape  # Batch, Views, Channel, Depth, Height, Width
        
        # Use mixed precision if enabled
        with torch.cuda.amp.autocast() if self.use_amp else torch.no_grad():
            if self.agr_type == "perceiver":
                # Preprocess input for MVITv2
                mv_flat = self.preprocess_for_mvit(mvimages)
                
                # Extract features using gradient checkpointing if enabled
                if self.use_checkpoint:
                    mvit_features = checkpoint.checkpoint(self.base_network, mv_flat)
                else:
                    mvit_features = self.base_network(mv_flat)
                
                # Reshape back to batch and view dimensions
                mvit_features = mvit_features.view(B, V, -1)  # (B, V, mvit_feat_dim)
                
                # Process with HRNet one view at a time to save memory
                hrnet_features = []
                for v in range(V):
                    # Process one view at a time
                    view_features = self.hrnet(mvimages[:, v])  # (B, hrnet_feat_dim)
                    hrnet_features.append(view_features.unsqueeze(1))  # (B, 1, hrnet_feat_dim)
                
                # Concatenate along view dimension
                hrnet_features = torch.cat(hrnet_features, dim=1)  # (B, V, hrnet_feat_dim)
                
                # Fuse features for each view
                fused_features = []
                for v in range(V):
                    # Concatenate features from both networks
                    combined = torch.cat([mvit_features[:, v], hrnet_features[:, v]], dim=-1)
                    # Apply fusion
                    fused_view = self.fusion(combined)  # (B, feat_dim)
                    fused_features.append(fused_view.unsqueeze(1))  # (B, 1, feat_dim)
                
                # Concatenate fused features from all views
                fused_features = torch.cat(fused_features, dim=1)  # (B, V, feat_dim)
                
                # Use Perceiver for multi-view aggregation
                if self.use_checkpoint:
                    outputs = checkpoint.checkpoint(self.perceiver, inputs=fused_features).last_hidden_state
                else:
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