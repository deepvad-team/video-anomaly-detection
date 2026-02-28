"""
ETCL Model - Compatible with C2FPL codebase

Key differences from original C2FPL Model:
1. Evidential output (Dirichlet parameters) instead of sigmoid
2. Uncertainty quantification for robust training
3. Temporal consistency modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init


def weight_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)


class EvidentialLayer(nn.Module):
    """
    Evidential output layer that produces Dirichlet parameters
    instead of simple probabilities
    """
    def __init__(self, in_features, num_classes=2):
        super().__init__()
        self.num_classes = num_classes
        self.fc = nn.Linear(in_features, num_classes)
        
    def forward(self, x):
        # Evidence must be non-negative
        evidence = F.softplus(self.fc(x))
        
        # Dirichlet parameters
        alpha = evidence + 1.0
        
        # Total evidence (Dirichlet strength)
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Expected probability
        prob = alpha / S
        
        # Uncertainty (epistemic)
        uncertainty = self.num_classes / S.squeeze(-1)
        
        return {
            'evidence': evidence,
            'alpha': alpha,
            'prob': prob,
            'anomaly_score': prob[:, 1] if len(prob.shape) > 1 else prob[1],
            'uncertainty': uncertainty
        }


class Model_ETCL(nn.Module):
    """
    ETCL Model - Drop-in replacement for C2FPL Model
    
    Changes from original:
    - Evidential output instead of sigmoid
    - Returns both anomaly_score and uncertainty
    """
    def __init__(self, n_features, hidden_dim=512, dropout=0.6):
        super(Model_ETCL, self).__init__()
        
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 32)
        
        # Evidential output layer
        self.evidential = EvidentialLayer(32, num_classes=2)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.apply(weight_init)
    
    def forward(self, inputs, return_uncertainty=True):
        """
        Args:
            inputs: [batch, seq_len, features] - e.g., (1, 89, 2048)
                    or [batch, features] - e.g., (128, 2048)
        Returns:
            if return_uncertainty: dict with shape (batch, seq_len) or (batch,)
            else: anomaly_score tensor
        """
        original_shape = inputs.shape
    
        # Handle 3D input: (batch, seq_len, features)
        if len(inputs.shape) == 3:
            batch_size, seq_len, feat_dim = inputs.shape
            # Flatten to (batch * seq_len, features) for processing
            inputs = inputs.reshape(-1, feat_dim)  # (89, 2048)
        else:
            batch_size = inputs.shape[0]
            seq_len = 1
            feat_dim = inputs.shape[-1]
    
        # Handle 10-crop if present: (batch, seq_len, 10, features)
        if len(original_shape) == 4:
            batch_size, seq_len, num_crops, feat_dim = original_shape
            inputs = inputs.reshape(-1, feat_dim)  # (batch * seq_len * crops, features)
            # We'll reshape back after processing
    
        # Forward through layers
        x = self.relu(self.fc1(inputs))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
    
        # Evidential output
        output = self.evidential(x)
    
        # Reshape back to (batch, seq_len, ...) if input was 3D
        if len(original_shape) == 3:
            for key in output:
                if isinstance(output[key], torch.Tensor):
                    if output[key].dim() == 1:
                        # (batch*seq_len,) -> (batch, seq_len)
                        output[key] = output[key].view(batch_size, seq_len)
                    elif output[key].dim() == 2:
                        # (batch*seq_len, num_classes) -> (batch, seq_len, num_classes)
                        output[key] = output[key].view(batch_size, seq_len, -1)
    
        if return_uncertainty:
            return output
        else:
            return output['anomaly_score']


class Model_ETCL_Temporal(nn.Module):
    """
    ETCL Model with Temporal Consistency Module
    
    For video-level processing with temporal modeling
    """
    def __init__(self, n_features, hidden_dim=512, dropout=0.6, 
                 window_sizes=[3, 5, 7]):
        super(Model_ETCL_Temporal, self).__init__()
        
        # Feature extraction (same as base model)
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 128)
        self.fc3 = nn.Linear(128, 32)
        
        # Temporal convolutions for consistency
        self.temporal_convs = nn.ModuleList([
            nn.Conv1d(32, 32, kernel_size=w, padding=w//2)
            for w in window_sizes
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(32 * len(window_sizes), 32)
        
        # Evidential output
        self.evidential = EvidentialLayer(32, num_classes=2)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        
        self.apply(weight_init)
    
    def forward(self, inputs, return_temporal_features=False):
        """
        Args:
            inputs: [batch_size, seq_len, n_features] - video-level input
        """
        batch_size, seq_len, n_features = inputs.shape
        
        # Handle 10-crop if present
        if len(inputs.shape) == 4:
            inputs = inputs.mean(dim=2)  # Average over crops
        
        # Reshape for batch processing
        x = inputs.view(-1, n_features)
        
        # Feature extraction
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        
        # Reshape back to sequence
        x = x.view(batch_size, seq_len, -1)  # [B, T, 32]
        
        # Temporal processing
        x_t = x.transpose(1, 2)  # [B, 32, T]
        
        temporal_features = []
        for conv in self.temporal_convs:
            conv_out = self.relu(conv(x_t))
            temporal_features.append(conv_out)
        
        # Concatenate and fuse
        x_temporal = torch.cat(temporal_features, dim=1)  # [B, 32*3, T]
        x_temporal = x_temporal.transpose(1, 2)  # [B, T, 32*3]
        x_fused = self.relu(self.fusion(x_temporal))  # [B, T, 32]
        x_fused = self.dropout(x_fused)
        
        # Evidential output for each timestep
        x_flat = x_fused.view(-1, 32)
        output = self.evidential(x_flat)
        
        # Reshape outputs
        for key in output:
            if len(output[key].shape) == 1:
                output[key] = output[key].view(batch_size, seq_len)
            elif len(output[key].shape) == 2:
                output[key] = output[key].view(batch_size, seq_len, -1)
        
        if return_temporal_features:
            output['temporal_features'] = x_fused
        
        return output


class EvidentialLoss(nn.Module):
    """
    Loss function for Evidential Deep Learning
    
    Combines:
    1. Type II Maximum Likelihood loss
    2. KL divergence regularization
    3. Optional uncertainty-aware weighting
    """
    def __init__(self, num_classes=2, kl_weight=0.1, annealing_epochs=10):
        super().__init__()
        self.num_classes = num_classes
        self.kl_weight = kl_weight
        self.annealing_epochs = annealing_epochs
    
    def forward(self, alpha, targets, epoch=0, sample_weights=None):
        """
        Args:
            alpha: Dirichlet parameters [batch_size, num_classes]
            targets: Target labels [batch_size] (0 or 1)
            epoch: Current epoch for KL annealing
            sample_weights: Optional confidence weights
        """
        # One-hot encode targets
        if len(targets.shape) == 1:
            y = F.one_hot(targets.long(), self.num_classes).float()
        else:
            y = targets
        
        S = alpha.sum(dim=-1, keepdim=True)
        
        # Type II Maximum Likelihood Loss
        mle_loss = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
        
        # KL Divergence regularization
        alpha_tilde = y + (1 - y) * alpha
        kl_loss = self._kl_divergence(alpha_tilde)
        
        # Annealing coefficient
        annealing = min(1.0, epoch / self.annealing_epochs)
        
        # Apply sample weights if provided
        if sample_weights is not None:
            # Higher confidence = higher weight
            weights = sample_weights ** 2  # Squared for stronger effect
            mle_loss = (mle_loss * weights).sum() / (weights.sum() + 1e-8)
            kl_loss = (kl_loss * weights).sum() / (weights.sum() + 1e-8)
        else:
            mle_loss = mle_loss.mean()
            kl_loss = kl_loss.mean()
        
        total_loss = mle_loss + annealing * self.kl_weight * kl_loss
        
        return {
            'total': total_loss,
            'mle': mle_loss,
            'kl': kl_loss
        }
    
    def _kl_divergence(self, alpha):
        """KL divergence from uniform Dirichlet"""
        ones = torch.ones_like(alpha)
        sum_alpha = alpha.sum(dim=-1)
        sum_ones = ones.sum(dim=-1)
        
        kl = torch.lgamma(sum_alpha) - torch.lgamma(sum_ones) \
             - (torch.lgamma(alpha) - torch.lgamma(ones)).sum(dim=-1) \
             + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(sum_alpha.unsqueeze(-1)))).sum(dim=-1)
        
        return kl


class TemporalConsistencyLoss(nn.Module):
    """
    Temporal consistency regularization loss
    
    Penalizes abrupt changes in anomaly scores,
    weighted by prediction confidence
    """
    def __init__(self, weight=0.1):
        super().__init__()
        self.weight = weight
    
    def forward(self, anomaly_scores, uncertainty):
        """
        Args:
            anomaly_scores: [batch_size, seq_len]
            uncertainty: [batch_size, seq_len]
        """
        if len(anomaly_scores.shape) == 1:
            # Single video
            anomaly_scores = anomaly_scores.unsqueeze(0)
            uncertainty = uncertainty.unsqueeze(0)
        
        # Confidence = 1 - uncertainty
        confidence = (1.0 - uncertainty.clamp(0, 1))
        
        # Temporal differences
        diff = torch.abs(anomaly_scores[:, 1:] - anomaly_scores[:, :-1])
        
        # Weight by confidence of adjacent segments
        conf_weight = confidence[:, 1:] * confidence[:, :-1]
        
        # Weighted consistency loss
        loss = (diff * conf_weight).mean()
        
        return self.weight * loss


# Utility functions for compatibility with C2FPL
def convert_c2fpl_checkpoint(c2fpl_state_dict):
    """
    Convert C2FPL model weights to ETCL format
    """
    etcl_state_dict = {}
    
    for key, value in c2fpl_state_dict.items():
        if key.startswith('fc1') or key.startswith('fc2'):
            etcl_state_dict[key] = value
        elif key.startswith('fc3'):
            # fc3 in C2FPL outputs 1 dim, ETCL evidential outputs 2
            # Initialize new layer instead
            pass
    
    return etcl_state_dict
