import torch

from src.core.losses.loss_utils import intermediate_metric_calculation


class BCEAndIouLoss(torch.nn.Module):
    def __init__(self, smooth=1e-6, use_dice=False):
        super().__init__()
        self.smooth = smooth
        self.use_dice = use_dice

    def forward(self, predictions, targets):
        # predictions --> (B, #C, H, W) unnormalized
        # targets     --> (B, #C, H, W) one-hot encoded

        # Normalize model predictions
        predictions = torch.sigmoid(predictions)

        # Calculate pixel-wise loss for both channels. Shape: Scalar
        pixel_loss = torch.nn.functional.binary_cross_entropy(predictions, targets, reduction="mean")

        mask_loss = 1 - intermediate_metric_calculation(
            predictions, targets,
            use_dice=self.use_dice, smooth=self.smooth
        )
        total_loss = mask_loss + pixel_loss

        return total_loss
