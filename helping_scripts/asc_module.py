import torch
import torch.nn as nn
import torch.nn.functional as F

class ASCModule(nn.Module):
    def __init__(self, feature_dim, num_heads=4):
        super(ASCModule, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # Transformer encoder for affinity computation
        self.transformer = nn.TransformerEncoderLayer(d_model=feature_dim, nhead=num_heads)
        
        # 1x1 convolution to predict initial depth from features
        self.initial_depth_predictor = nn.Conv2d(feature_dim, 1, kernel_size=1)
        
        # Correction confidence prediction
        self.confidence_predictor = nn.Sequential(
            nn.Conv2d(feature_dim + 2, feature_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(feature_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # Learnable query and key transformations for cross-attention
        self.query_transform = nn.Linear(feature_dim, feature_dim)
        self.key_transform = nn.Linear(feature_dim, feature_dim)
        
    def forward(self, features, depth_points, depth_values):
        # features: [B, C, H, W]
        # depth_points: List of depth point coordinates [B, N, 2]
        # depth_values: Corresponding depth values for the points [B, N, 1]

        B, C, H, W = features.size()

        # Predict initial depth map from features
        initial_depth = self.initial_depth_predictor(features)  # [B, 1, H, W]
        
        # Sample features at depth point locations
        sampled_features = []
        for b in range(B):
            sampled_features.append(F.grid_sample(features[b:b+1], depth_points[b:b+1]))  # [B, C, N]
        sampled_features = torch.cat(sampled_features, dim=0)  # [B, C, N]
        
        # Concatenate depth values with sampled features
        depth_features = torch.cat([sampled_features, depth_values], dim=2)  # [B, C+1, N]
        
        # Compute affinity via cross-attention
        query = self.query_transform(features.view(B, C, -1))  # [B, C, H*W]
        key = self.key_transform(depth_features)  # [B, C, N]
        attention_scores = torch.bmm(query.permute(0, 2, 1), key)  # [B, H*W, N]
        affinities = F.softmax(attention_scores, dim=-1)  # [B, H*W, N]

        # Shift correction
        depth_errors = depth_values - initial_depth
        shift_correction = torch.bmm(affinities, depth_errors)  # [B, H*W, 1]
        shift_correction = shift_correction.view(B, 1, H, W)  # Reshape back to [B, 1, H, W]

        # Apply shift correction
        corrected_depth = initial_depth + shift_correction  # [B, 1, H, W]

        # Compute confidence to fuse initial and corrected depth
        confidence_input = torch.cat([features, initial_depth, corrected_depth], dim=1)
        confidence = self.confidence_predictor(confidence_input)  # [B, 1, H, W]

        # Fuse initial and corrected depth using confidence
        final_depth = (1 - confidence) * initial_depth + confidence * corrected_depth
        
        return final_depth
