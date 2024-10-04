""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet_2(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet_2, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        
        # Define the first output head (for mask)
        self.outc_mask = nn.Conv2d(64, 1, kernel_size=1)  # Single channel output for mask
        
        # Define the second output head (for spatial diffusion map)
        self.outc_diffusion = nn.Conv2d(64, 1, kernel_size=1)  # Single channel output for diffusion map        

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Two separate outputs
        mask_out = self.outc_mask(x)          # Output for mask
        diffusion_out = self.outc_diffusion(x)  # Output for diffusion map

        return mask_out, diffusion_out
    
    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc_mask = torch.utils.checkpoint(self.outc_mask)
        self.outc_diffusion = torch.utils.checkpoint(self.outc_diffusion)