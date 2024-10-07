import torch
import torch.nn.functional as F


def calculate_loss( mask_pred, diffusion_pred, mask_true,diffusion_true ):


    #results_pred = clean_zero(mask_pred,diffusion_pred)
    #results_true = clean_zero(mask_true, diffusion_true)
    mask_pred_prob = torch.sigmoid(mask_pred)


    threshold = 0.5
    binary_mask = (mask_pred_prob >= threshold).float() 
    results_pred = binary_mask * diffusion_pred
    results_true = mask_true * diffusion_true
    results_true = results_true.unsqueeze(1)  # This will add the channel dimension back if needed

    loss_mask = F.binary_cross_entropy(mask_pred_prob.squeeze(1), mask_true.float())
    loss_diffusion = F.mse_loss(results_pred, results_true)

    return loss_mask, loss_diffusion


def clean_zero(mask, diffusion):
    combined_matrix = mask  * diffusion
    non_zero_mask = combined_matrix != 0  # A mask for non-zero values
    non_zero_values = combined_matrix[non_zero_mask]

    return non_zero_values