import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp
    
class ApplyStyle(nn.Module):
    """
        @ref: https://github.com/lernapparat/lernapparat/blob/master/style_gan/pytorch_style_gan.ipynb
    """
    def __init__(self, latent_size, channels):
        super(ApplyStyle, self).__init__()
        self.linear = nn.Linear(latent_size, channels * 2)

    def forward(self, x, latent):
        style = self.linear(latent)  # style => [batch_size, n_channels*2]
        shape = [-1, 2, x.size(1), 1, 1]
        style = style.view(shape)    # [batch_size, 2, n_channels, ...]
        #x = x * (style[:, 0] + 1.) + style[:, 1]
        x = x * (style[:, 0] * 1 + 1.) + style[:, 1] * 1
        return x
    
class IIM(nn.Module):
    def __init__(self, n_channels, n_id_latent):
        super().__init__()

        self.activation = nn.ReLU(True)

        self.conv1 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0),
                                   InstanceNorm())
        self.style1 = ApplyStyle(n_id_latent, n_channels)

        self.conv2 = nn.Sequential(nn.ReflectionPad2d(1),
                                   nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0),
                                   InstanceNorm())
        self.style2 = ApplyStyle(n_id_latent, n_channels)
        
    def forward(self, x, latents):
        _x = self.conv1(x)
        _x = self.style1(_x, latents)
        _x = self.activation(_x)
        _x = self.conv2(_x)
        _x = self.style2(_x, latents)
        
        return x + _x