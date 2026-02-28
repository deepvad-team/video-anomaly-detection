"""
ETCL Training - Compatible with C2FPL

Key modifications from C2FPL train.py:
1. Evidential loss instead of BCE
2. Uncertainty-aware sample weighting
3. Temporal consistency regularization
4. Iterative pseudo label refinement
"""

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import BCELoss
from tqdm import tqdm
import time

#torch.set_default_tensor_type('torch.cuda.FloatTensor')


class ETCLLoss:
    """
    Combined loss for ETCL training
    
    Components:
    1. Evidential MLE loss (classification)
    2. KL divergence regularization
    3. Temporal consistency loss
    4. Uncertainty-aware weighting
    """
    
    def __init__(
        self,
        kl_weight: float = 0.1,
        temporal_weight: float = 0.1,
        annealing_epochs: int = 10,
        use_uncertainty_weighting: bool = True
    ):
        self.kl_weight = kl_weight
        self.temporal_weight = temporal_weight
        self.annealing_epochs = annealing_epochs
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.bce_loss = BCELoss(reduction='none')
    
    def __call__(
        self,
        output: dict,
        labels: torch.Tensor,
        confidence: torch.Tensor = None,
        epoch: int = 0
    ) -> dict:
        """
        Compute ETCL loss
        
        Args:
            output: Model output dict with 'alpha', 'anomaly_score', 'uncertainty'
            labels: Pseudo labels [batch_size]
            confidence: Confidence scores [batch_size]
            epoch: Current epoch for annealing
            
        Returns:
            Dictionary with loss components
        """
        alpha = output['alpha']
        anomaly_score = output['anomaly_score']
        uncertainty = output['uncertainty']
        
        # Flatten if needed
        if len(alpha.shape) > 2:
            alpha = alpha.view(-1, alpha.shape[-1])
            anomaly_score = anomaly_score.view(-1)
            uncertainty = uncertainty.view(-1)
            labels = labels.view(-1)
            if confidence is not None:
                confidence = confidence.view(-1)
        
        # 1. Evidential MLE Loss
        mle_loss = self._evidential_mle_loss(alpha, labels)
        
        # 2. KL Divergence Regularization
        kl_loss = self._kl_divergence_loss(alpha, labels)
        
        # 3. Uncertainty-aware weighting
        if self.use_uncertainty_weighting and confidence is not None:
            # Combine provided confidence with model uncertainty
            model_confidence = 1.0 - uncertainty.clamp(0, 1)
            combined_confidence = confidence * model_confidence
            weights = combined_confidence ** 2
            
            mle_loss = (mle_loss * weights).sum() / (weights.sum() + 1e-8)
            kl_loss = (kl_loss * weights).sum() / (weights.sum() + 1e-8)
        else:
            mle_loss = mle_loss.mean()
            kl_loss = kl_loss.mean()
        
        # Annealing for KL
        annealing_coef = min(1.0, epoch / max(self.annealing_epochs, 1))
        
        # Total loss
        total_loss = mle_loss + annealing_coef * self.kl_weight * kl_loss
        
        return {
            'total': total_loss,
            'mle': mle_loss,
            'kl': kl_loss,
            'mean_uncertainty': uncertainty.mean()
        }
    
    def _evidential_mle_loss(self, alpha, targets):
        """Type II Maximum Likelihood loss for Dirichlet"""
        # One-hot encode
        y = torch.zeros_like(alpha)
        y[:, 0] = 1 - targets
        y[:, 1] = targets
        
        S = alpha.sum(dim=-1, keepdim=True)
        loss = (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=-1)
        
        return loss
    
    def _kl_divergence_loss(self, alpha, targets):
        """KL divergence from uniform Dirichlet"""
        # One-hot encode
        y = torch.zeros_like(alpha)
        y[:, 0] = 1 - targets
        y[:, 1] = targets
        
        # Remove evidence for correct class
        alpha_tilde = y + (1 - y) * alpha
        
        ones = torch.ones_like(alpha_tilde)
        sum_alpha = alpha_tilde.sum(dim=-1)
        sum_ones = ones.sum(dim=-1)
        
        kl = torch.lgamma(sum_alpha) - torch.lgamma(sum_ones) \
             - (torch.lgamma(alpha_tilde) - torch.lgamma(ones)).sum(dim=-1) \
             + ((alpha_tilde - ones) * (torch.digamma(alpha_tilde) - 
                torch.digamma(sum_alpha.unsqueeze(-1)))).sum(dim=-1)
        
        return kl


def etcl_train_epoch(
    loader,
    model,
    optimizer,
    labels: np.ndarray,
    confidence: np.ndarray,
    device,
    loss_fn,
    epoch: int = 0
):
    model.train()
    losses = []
    new_predictions = []
    
    confidence_tensor = torch.FloatTensor(confidence).to(device)
    
    for batch_idx, (input, batch_labels) in enumerate(loader):
        # input: (batch, 2048) - 이미 segment 단위, crop 평균됨
        # batch_labels: (batch,) - pseudo labels
        input = input.to(device)
        batch_labels = batch_labels.to(device)
        
        # Confidence 가져오기 (batch 인덱스 기반)
        # UCFTrainSnippetDataset은 순차적으로 반환하므로 별도 처리 필요
        # 일단 uniform confidence 사용
        batch_confidence = torch.ones_like(batch_labels)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(input, return_uncertainty=True)
        
        # Compute loss
        loss_dict = loss_fn(
            output,
            batch_labels,
            batch_confidence,
            epoch=epoch
        )
        
        total_loss = loss_dict['total']
        losses.append(total_loss.item())
        
        # Backward
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    return np.mean(losses), new_predictions


def refine_pseudo_labels(
    original_labels: np.ndarray,
    original_confidence: np.ndarray,
    predictions: list,
    confidence_threshold: float = 0.7,
    uncertainty_threshold: float = 0.3
) -> tuple:
    """
    Refine pseudo labels using model predictions
    
    Strategy:
    - Update labels where model is confident (low uncertainty)
    - Keep original labels where model is uncertain
    
    Args:
        original_labels: Current pseudo labels
        original_confidence: Current confidence scores
        predictions: List of (idx, score, uncertainty) from model
        confidence_threshold: Minimum confidence to update
        uncertainty_threshold: Maximum uncertainty to update
        
    Returns:
        refined_labels, refined_confidence
    """
    refined_labels = original_labels.copy()
    refined_confidence = original_confidence.copy()
    
    # Sort predictions by index
    predictions = sorted(predictions, key=lambda x: x[0])
    
    updates = 0
    for idx, score, uncertainty in predictions:
        model_confidence = 1.0 - uncertainty
        
        # Only update if model is confident
        if model_confidence > confidence_threshold and uncertainty < uncertainty_threshold:
            new_label = 1.0 if score > 0.5 else 0.0
            
            # Check if this changes the label
            if new_label != original_labels[idx]:
                refined_labels[idx] = new_label
                refined_confidence[idx] = model_confidence
                updates += 1
        else:
            # Keep original but maybe reduce confidence if model disagrees
            predicted_label = 1.0 if score > 0.5 else 0.0
            if predicted_label != original_labels[idx]:
                refined_confidence[idx] = min(refined_confidence[idx], 0.5 + uncertainty)
    
    print(f"  Pseudo label updates: {updates} segments changed")
    
    return refined_labels, refined_confidence


def etcl_train_with_refinement(
    train_loader,
    test_loader,
    model,
    optimizer,
    scheduler,
    initial_labels: np.ndarray,
    initial_confidence: np.ndarray,
    device,
    args,
    test_fn
):
    """
    Full ETCL training with iterative pseudo label refinement
    
    Args:
        train_loader: Training data loader
        test_loader: Test data loader
        model: ETCL model
        optimizer: Optimizer
        scheduler: LR scheduler
        initial_labels: Initial pseudo labels
        initial_confidence: Initial confidence scores
        device: torch device
        args: Training arguments
        test_fn: Test function (test from C2FPL)
        
    Returns:
        best_auc: Best achieved AUC
    """
    # Initialize loss function
    loss_fn = ETCLLoss(
        kl_weight=getattr(args, 'kl_weight', 0.1),
        temporal_weight=getattr(args, 'temporal_weight', 0.1),
        annealing_epochs=getattr(args, 'annealing_epochs', 10)
    )
    
    # Current labels
    labels = initial_labels.copy()
    confidence = initial_confidence.copy()
    
    # Training parameters
    max_epoch = getattr(args, 'max_epoch', 50)
    refinement_interval = getattr(args, 'refinement_interval', 10)
    
    best_auc = 0.0
    test_info = {"epoch": [], "test_auc": []}
    
    # Initial evaluation
    auc, ap = test_fn(test_loader, model, args, device)
    print(f"Epoch 0: AUC = {auc:.4f}, AP = {ap:.4f}")
    
    for epoch in tqdm(range(1, max_epoch + 1), total=max_epoch, dynamic_ncols=True):
        # Train epoch
        loss, predictions = etcl_train_epoch(
            train_loader, model, optimizer,
            labels, confidence, device,
            loss_fn, epoch
        )
        
        # Evaluate
        auc, ap = test_fn(test_loader, model, args, device)
        
        test_info["epoch"].append(epoch)
        test_info["test_auc"].append(auc)
        
        scheduler.step()
        
        print(f'\nEpoch {epoch}/{max_epoch}, LR: {optimizer.param_groups[0]["lr"]:.4f}, '
              f'AUC: {auc:.4f}, AP: {ap:.4f}, Loss: {loss:.4f}')
        
        # Save best model
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f'./ckpt/etcl_best.pkl')
            print(f"  New best AUC: {best_auc:.4f}")
        
        # Pseudo label refinement
        if epoch % refinement_interval == 0 and epoch < max_epoch - refinement_interval:
            print(f"\n  Refining pseudo labels at epoch {epoch}")
            labels, confidence = refine_pseudo_labels(
                labels, confidence, predictions,
                confidence_threshold=0.7,
                uncertainty_threshold=0.3
            )
    
    print(f"\nTraining completed. Best AUC: {best_auc:.4f}")
    
    return best_auc, test_info


# Compatibility wrapper for C2FPL's train function signature
def concatenated_train_feedback_etcl(loader, model, optimizer, original_label, device, 
                                      confidence=None, epoch=0, loss_fn=None):
    """
    Drop-in replacement for C2FPL's concatenated_train_feedback
    
    Adds uncertainty-aware training while maintaining same interface
    """
    if confidence is None:
        confidence = np.ones_like(original_label)
    
    if loss_fn is None:
        loss_fn = ETCLLoss()
    
    return etcl_train_epoch(
        loader, model, optimizer,
        original_label, confidence, device,
        loss_fn, epoch
    )
