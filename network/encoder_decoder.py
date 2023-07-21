import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, in_channels, n_channels, n_depth):
        super().__init__()

        self.input_layer = nn.ReflectionPad2d(3)

        layers = [nn.Conv2d(in_channels, n_channels, kernel_size=7, padding=0),
                  nn.BatchNorm2d(n_channels),
                  nn.ReLU(True)]
        
        for n in range(n_depth - 1):
            layers += [nn.Conv2d(n_channels * (2 ** n), n_channels * (2 ** (n + 1)), kernel_size=3, stride=2, padding=1),
                       nn.BatchNorm2d(n_channels * (2 ** (n + 1))),
                       nn.ReLU(True)]
            
        self.down_blocks= nn.ModuleList(layers)

    def forward(self, x):
        x = self.input_layer(x)

        for block in self.down_blocks:
            x = block(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, in_channels, n_channels, n_depth, n_pose_info):
        super().__init__()

        layers = []
        for n in reversed(range(n_depth - 1)):
            layers += [nn.Sequential(nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                                     nn.Conv2d(n_channels * (2 ** (n + 1)) + n_pose_info, n_channels * (2 ** n), kernel_size=3, stride=1, padding=1),
                                     nn.BatchNorm2d(n_channels * (2 ** n)),
                                     nn.ReLU(True))]
            
        self.up_blocks = nn.ModuleList(layers)
            
        self.output_layer = nn.Sequential(nn.ReflectionPad2d(3),
                                          nn.Conv2d(n_channels, in_channels, kernel_size=7, padding=0))
        
        self.pose_upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, x, pose_info=None):
        for block in self.up_blocks:
            if pose_info is not None:
                x = torch.cat([x, pose_info], dim=1)
                x = block(x)
                pose_info = self.pose_upsample(pose_info)
            
            else:
                x = block(x)

        x = self.output_layer(x)
        return x