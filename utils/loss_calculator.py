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

    # Only calculate the MSE based on the non-zero values: 
    results_true_list = results_true.flatten()
    results_pred_list = results_pred.flatten()
    result_combine = [[a,b] for a, b in zip(results_true_list,results_pred_list) if a * b != 0]

    if len(result_combine) == 0:
        # If there are no non-zero pairs, set diffusion loss to 0
        loss_diffusion = torch.tensor(0.0, device=mask_pred.device, requires_grad=True)
    else:
        results_true_nonzero = torch.stack([x[0] for x in result_combine]).to(mask_pred.device)
        results_pred_nonzero = torch.stack([x[1] for x in result_combine]).to(mask_pred.device)

        loss_diffusion = F.mse_loss(results_true_nonzero, results_pred_nonzero)

    return loss_mask, loss_diffusion


def clean_zero(mask, diffusion):
    combined_matrix = mask  * diffusion
    non_zero_mask = combined_matrix != 0  # A mask for non-zero values
    non_zero_values = combined_matrix[non_zero_mask]

    return non_zero_values