import torch
import torch.nn as nn

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels, n_channels, n_depth, n_discriminator, device="cpu"):
        super().__init__()

        self.discriminators = []
        for _ in range(n_discriminator):
            self.discriminators.append(NLayerDiscriminator(in_channels, n_channels, n_depth).to(device))

        self.downsample = nn.AvgPool2d(kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x, feature_idx=None):
        disc_output = []
        disc_features = []
        for disc in self.discriminators:
            output, features = disc(x, feature_idx=feature_idx)
            x = self.downsample(x)

            disc_output.append(output)
            disc_features.append(features)

        return disc_output, disc_features


class NLayerDiscriminator(nn.Module):
    def __init__(self, in_channels, n_channels, n_depth):
        super().__init__()

        self.input_layer = nn.Conv2d(in_channels, n_channels, kernel_size=4, padding=1)
        self.activation = nn.LeakyReLU(0.2, True)

        layers = []
        for n in range(n_depth):
            layers += [nn.Sequential(nn.Conv2d(n_channels * (2 ** n), n_channels * (2 ** (n + 1)), kernel_size=4, stride=2, padding=1),
                                     nn.BatchNorm2d(n_channels * (2 ** (n + 1))),
                                     nn.LeakyReLU(0.2, True))]
            
        layers += [nn.Sequential(nn.Conv2d(n_channels * (2 ** (n + 1)), n_channels * (2 ** (n + 1)), kernel_size=4, stride=1, padding=1),
                                 nn.BatchNorm2d(n_channels * (2 ** (n + 1))),
                                 nn.LeakyReLU(0.2, True))]
        self.backbone_blocks = nn.ModuleList(layers)

        self.output_layer = nn.Conv2d(n_channels * (2 ** (n + 1)), 1, kernel_size=4, stride=1, padding=1)

    def forward(self, x, feature_idx):
        x = self.input_layer(x)
        x = self.activation(x)

        features = []
        for idx, block in enumerate(self.backbone_blocks):
            x = block(x)

            if feature_idx is not None and idx in feature_idx:
                features.append(x)

        x = self.output_layer(x)

        return x, features
    
if __name__ == "__main__":
    disc = MultiScaleDiscriminator(3, 64, 3, 2)
    disc_output, disc_features = disc(torch.randn(1, 3, 256, 256), [2, 3])
    print(len(disc_output))
    print(disc_output[0].shape, disc_output[-1].shape)
    print(disc_features[1][0].shape, disc_features[1][1].shape)

    def hinge_loss(X, positive=True):
        if positive:
            return torch.relu(1-X)
        else:
            return torch.relu(X+1)
        
    adv_loss = 0
    for f_logit in disc_output:
        adv_loss += hinge_loss(f_logit, True).mean([1, 2, 3])
    adv_loss /= len(disc_output)
    print(adv_loss)
