import torch
import torch.nn as nn

from network.encoder_decoder import Encoder, Decoder
from network.id_injection_module import IIM

class FaceSwapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.encoder = Encoder(in_channels=cfg.network.in_channels,
                               n_channels=cfg.network.n_channels,
                               n_depth=cfg.network.n_ae_depth)
        
        self.decoder = Decoder(in_channels=cfg.network.in_channels,
                               n_channels=cfg.network.n_channels,
                               n_depth=cfg.network.n_ae_depth,
                               n_pose_info=cfg.network.n_pose_info)
        
        block = []
        for _ in range(cfg.network.n_iid_block):
            block.append(IIM(n_channels=cfg.network.n_channels * (2 ** (cfg.network.n_ae_depth - 1)),
                             n_id_latent=cfg.network.n_id_latent))
        self.iim_blocks = nn.ModuleList(block)
        
    def forward(self, x, id_latent, pose_info=None):
        x = self.encoder(x)

        for block in self.iim_blocks:
            x = block(x, id_latent)

        x = self.decoder(x, pose_info)

        return x