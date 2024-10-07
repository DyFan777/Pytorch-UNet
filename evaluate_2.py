import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.loss_calculator import calculate_loss

def evaluate_2(net, dataloader, device, amp):
    net.eval()
    total_loss1 = 0.0
    total_loss2 = 0.0
    num_batches = 0

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_batches, desc='Validation round', unit='batch', leave=False):
            images, true_masks, true_D = batch['image'], batch['mask'], batch['D']
            
            images = images.to(device)
            true_masks = true_masks.to(device)
            true_D = true_D.to(device)
            
            # Forward pass
            masks_pred, diffusion_pred = net(images)  # Ensure you unpack the outputs

            loss1,loss2 = calculate_loss(masks_pred,diffusion_pred,true_masks,true_D)


            # Accumulate the losses
            total_loss1 += loss1.item()
            total_loss2 += loss2.item()
            num_batches += 1

    # Average the losses over all batches
    avg_loss1 = total_loss1 / num_batches if num_batches > 0 else 0
    avg_loss2 = total_loss2 / num_batches if num_batches > 0 else 0

    return avg_loss1, avg_loss2
