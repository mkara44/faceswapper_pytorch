import dlib
import torch
import torch.nn as nn

from network.discriminator import MultiScaleDiscriminator
from utils.shape_predictor import InsightFaceShapePredictor


def hinge_loss(X, positive=True):
    if positive:
        return torch.relu(1-X)
    else:
        return torch.relu(X+1)

class SwapperLoss(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.rec_loss_weight = cfg.loss.rec_loss_weight
        self.attribute_loss_weight = cfg.loss.attribute_loss_weight
        self.id_loss_weight = cfg.loss.id_loss_weight
        self.weak_fm_loss_weight = cfg.loss.weak_fm_loss_weight
        self.adv_loss_weight = cfg.loss.adv_loss_weight

        self.discriminator = MultiScaleDiscriminator(in_channels=cfg.network.discriminator.in_channels,
                                                     n_channels=cfg.network.discriminator.n_channels,
                                                     n_depth=cfg.network.discriminator.n_depth,
                                                     n_discriminator=cfg.network.discriminator.n_discriminator)
        self.weak_fm_loss_layer_idx = cfg.network.discriminator.weak_fm_loss_layer_idx

        self.shape_predictor = InsightFaceShapePredictor(cfg=cfg)

        self.rec_criterion = nn.L1Loss()
        self.weak_fm_criterion = nn.L1Loss()
        self.attribute_criterion = nn.MSELoss()

    def forward(self, source_img, target_img, swapped_img, 
                source_id_latent=None, swapped_id_latent=None, same_person=None, optimizer_idx=None):
        loss = 0
        log_dict = {}
    
        if optimizer_idx == 0:
            if self.rec_loss_weight > 0:
                n = 0
                rec_loss = 0
                for idx in range(same_person.shape[0]):
                    if not same_person[idx, ...]:
                         continue
                    
                    n += 1
                    rec_loss += self.rec_criterion(swapped_img[idx, ...], target_img[idx, ...])
                rec_loss = rec_loss / n if n != 0 else torch.tensor(0, dtype=torch.float)

                log_dict["rec_loss"] = rec_loss.detach().clone()
                loss += self.rec_loss_weight * rec_loss

            if self.attribute_loss_weight > 0:
                att_loss = 0
                for idx in range(source_img.shape[0]):
                    source_att = self.shape_predictor(source_img[idx, ...], single_face=True, preprocess=True)
                    swapped_att = self.shape_predictor(swapped_img[idx, ...], single_face=True, preprocess=True)
                    att_loss += self.attribute_criterion(swapped_att, source_att)
                att_loss = att_loss / source_img.shape[0]

                log_dict["att_loss"] = att_loss.detach().clone()
                loss += self.attribute_loss_weight * att_loss

            if self.id_loss_weight > 0:
                id_loss = (1 - torch.cosine_similarity(source_id_latent, swapped_id_latent, dim=1)).mean()

                log_dict["id_loss"] = id_loss.detach().clone()
                loss += self.id_loss_weight * id_loss

            if self.weak_fm_loss_weight > 0 or self.adv_loss_weight > 0:
                _, real_features = self.discriminator(target_img, self.weak_fm_loss_layer_idx)
                fake_logits, fake_features = self.discriminator(swapped_img, self.weak_fm_loss_layer_idx)

                adv_loss = 0
                for f_logit in fake_logits:
                    adv_loss += hinge_loss(f_logit, True).mean()
                adv_loss = adv_loss / len(fake_logits)

                weak_fm_loss = 0
                for r_feat, f_feat in zip(real_features, fake_features):
                    _weak_fm_loss = 0
                    for idx in range(len(r_feat)):
                        _weak_fm_loss += self.weak_fm_criterion(f_feat[idx], r_feat[idx])
                    weak_fm_loss = _weak_fm_loss / len(r_feat)
                weak_fm_loss = weak_fm_loss / len(real_features)

                log_dict["adv_loss"] = adv_loss.detach().clone()
                log_dict["weak_fm_loss"] = weak_fm_loss.detach().clone()
                loss += self.adv_loss_weight * adv_loss + self.weak_fm_loss_weight * weak_fm_loss

        elif optimizer_idx == 1:
                real_logits, _ = self.discriminator(target_img)
                fake_logits, _ = self.discriminator(swapped_img)

                real_disc_loss = 0
                fake_disc_loss = 0
                for r_logit, f_logit in zip(real_logits, fake_logits):
                    real_disc_loss += hinge_loss(r_logit, True).mean()
                    fake_disc_loss += hinge_loss(f_logit, False).mean()

                disc_loss = real_disc_loss / len(real_logits) + fake_disc_loss / len(fake_logits)

                log_dict["disc_loss"] = disc_loss.detach().clone()
                loss += disc_loss

        return loss, log_dict