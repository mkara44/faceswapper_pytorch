import torch
from pytorch_lightning import LightningModule

from network.face_swapper import FaceSwapper
from utils.arcface import ArcFace
from utils.loss import SwapperLoss
from utils.dataset import VGGFace2
from torch.utils.data import DataLoader


class FaceSwapperTrain(LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.cfg = cfg
        self.batch_size = self.cfg.train.batch_size
        self.learning_rate = self.cfg.train.learning_rate

        self.face_swapper = FaceSwapper(self.cfg)
        self.arc_face = ArcFace(self.cfg)
        self.loss = SwapperLoss(self.cfg)

    def get_id_latent(self, img):
        with torch.no_grad():
            id_latent = self.arc_face(img)

        return id_latent

    def training_step(self, batch):
        source_img, target_img, same_person = batch

        source_id_latent = self.get_id_latent(source_img)

        opt_g, opt_d = self.optimizers()

        # Train G
        self.toggle_optimizer(opt_g)
        swapped_img = self.face_swapper(target_img, source_id_latent)
        swapped_id_latent = self.get_id_latent(swapped_img)
        g_loss, g_loss_dict = self.loss(source_img, target_img, swapped_img,
                                        source_id_latent, swapped_id_latent, same_person, 0)
        self.log_dict(g_loss_dict, on_epoch=True)
        self.manual_backward(g_loss)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Train D
        self.toggle_optimizer(opt_d)
        swapped_img = self.face_swapper(target_img, source_id_latent)
        d_loss, d_loss_dict = self.loss(source_img, target_img, swapped_img, optimizer_idx=1)
        self.log_dict(d_loss_dict, on_epoch=True)
        self.manual_backward(d_loss)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.face_swapper.parameters(), lr=self.learning_rate, betas=(0, 0.99), eps=1e-8)

        disc_parameters = []
        for disc in self.loss.discriminator.discriminators:
            disc_parameters += disc.parameters()

        opt_d = torch.optim.Adam(disc_parameters, lr=self.learning_rate, betas=(0, 0.99), eps=1e-8)
        return [opt_g, opt_d], []

    def train_dataloader(self):
        dataset = VGGFace2(self.cfg)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True)