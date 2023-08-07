import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.iresnet import iresnet100


class ArcFace(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        
        self.model = self.load_model(model_path=cfg.network.arcface.model_path)

    def load_model(self, model_path):
        model = iresnet100(fp16=False)
        model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
        for param in model.parameters():
            param.requires_grad = False
        model.eval()

        return model
    
    def __preprocess__(self, img):
        return F.interpolate(img, size=(112, 112), mode="bilinear", align_corners=True)

    def __call__(self, img):
        img = self.__preprocess__(img)
        return self.model(img)
