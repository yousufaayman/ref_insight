from utils import batch_tensor, unbatch_tensor
import torch
from torch import nn
import torch.utils.checkpoint as checkpoint


class WeightedAggregate(nn.Module):
    def __init__(self, model, feat_dim, lifting_net=nn.Sequential(), use_checkpoint=False):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.feature_dim = feat_dim
        self.use_checkpoint = use_checkpoint

        r1 = -1
        r2 = 1
        self.attention_weights = nn.Parameter((r1 - r2) * torch.rand(feat_dim, feat_dim) + r2)

        self.normReLu = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.ReLU()
        )        

        self.relu = nn.ReLU()

    def _process_with_model(self, x):
        """Helper function for checkpointing"""
        return self.model(x)

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        
        # Apply batch tensor operation to prepare for backbone processing
        batched_input = batch_tensor(mvimages, dim=1, squeeze=True)
        
        # Apply backbone with optional checkpointing
        if self.use_checkpoint:
            features = checkpoint.checkpoint(self._process_with_model, batched_input)
        else:
            features = self.model(batched_input)
        
        # Unbatch tensor operation
        aux = unbatch_tensor(features, B, dim=1, unsqueeze=True)
        
        # Apply lifting net if provided
        if len(self.lifting_net) > 0:
            aux = self.lifting_net(aux)

        # Compute view attention weights
        aux = torch.matmul(aux, self.attention_weights)
        aux_t = aux.permute(0, 2, 1)
        prod = torch.bmm(aux, aux_t)
        relu_res = self.relu(prod)
        
        # Normalize attention weights
        aux_sum = torch.sum(torch.reshape(relu_res, (B, V*V)).T, dim=0).unsqueeze(0)
        final_attention_weights = torch.div(torch.reshape(relu_res, (B, V*V)).T, aux_sum.squeeze(0))
        final_attention_weights = final_attention_weights.T
        final_attention_weights = torch.reshape(final_attention_weights, (B, V, V))
        final_attention_weights = torch.sum(final_attention_weights, 1)

        # Apply attention weights to features
        output = torch.mul(aux.squeeze(), final_attention_weights.unsqueeze(-1))
        output = torch.sum(output, 1)

        return output.squeeze(), final_attention_weights


class ViewMaxAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential(), use_checkpoint=False):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.use_checkpoint = use_checkpoint

    def _process_with_model(self, x):
        """Helper function for checkpointing"""
        return self.model(x)

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        
        # Apply batch tensor operation
        batched_input = batch_tensor(mvimages, dim=1, squeeze=True)
        
        # Apply backbone with optional checkpointing
        if self.use_checkpoint:
            features = checkpoint.checkpoint(self._process_with_model, batched_input)
        else:
            features = self.model(batched_input)
        
        # Unbatch tensor and apply lifting net
        aux = unbatch_tensor(features, B, dim=1, unsqueeze=True)
        if len(self.lifting_net) > 0:
            aux = self.lifting_net(aux)
            
        # Max pooling across views
        pooled_view = torch.max(aux, dim=1)[0]
        return pooled_view.squeeze(), aux


class ViewAvgAggregate(nn.Module):
    def __init__(self, model, lifting_net=nn.Sequential(), use_checkpoint=False):
        super().__init__()
        self.model = model
        self.lifting_net = lifting_net
        self.use_checkpoint = use_checkpoint

    def _process_with_model(self, x):
        """Helper function for checkpointing"""
        return self.model(x)

    def forward(self, mvimages):
        B, V, C, D, H, W = mvimages.shape # Batch, Views, Channel, Depth, Height, Width
        
        # Apply batch tensor operation
        batched_input = batch_tensor(mvimages, dim=1, squeeze=True)
        
        # Apply backbone with optional checkpointing
        if self.use_checkpoint:
            features = checkpoint.checkpoint(self._process_with_model, batched_input)
        else:
            features = self.model(batched_input)
        
        # Unbatch tensor and apply lifting net
        aux = unbatch_tensor(features, B, dim=1, unsqueeze=True)
        if len(self.lifting_net) > 0:
            aux = self.lifting_net(aux)
            
        # Average pooling across views
        pooled_view = torch.mean(aux, dim=1)
        return pooled_view.squeeze(), aux


class MVAggregate(nn.Module):
    def __init__(self, model, agr_type="max", feat_dim=400, lifting_net=nn.Sequential(), use_checkpoint=False):
        super().__init__()
        self.agr_type = agr_type
        self.use_checkpoint = use_checkpoint

        # Feature transformation layers
        self.inter = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, feat_dim),
        )

        # Classification heads
        self.fc_offence = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 4)
        )

        self.fc_action = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, feat_dim),
            nn.Linear(feat_dim, 8)
        )

        if self.agr_type == "max":
            self.aggregation_model = ViewMaxAggregate(
                model=model, 
                lifting_net=lifting_net,
                use_checkpoint=use_checkpoint
            )
        elif self.agr_type == "mean":
            self.aggregation_model = ViewAvgAggregate(
                model=model, 
                lifting_net=lifting_net,
                use_checkpoint=use_checkpoint
            )
        else: 
            self.aggregation_model = WeightedAggregate(
                model=model, 
                feat_dim=feat_dim, 
                lifting_net=lifting_net,
                use_checkpoint=use_checkpoint
            )

    def forward(self, mvimages):
        # Get pooled features and attention weights
        pooled_view, attention = self.aggregation_model(mvimages)

        # Apply feature transformation
        inter = self.inter(pooled_view)
        
        # Make predictions
        pred_action = self.fc_action(inter)
        pred_offence_severity = self.fc_offence(inter)

        return pred_offence_severity, pred_action, attention