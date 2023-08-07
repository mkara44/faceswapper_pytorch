import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="5"

import torch
from omegaconf import OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from trainer import FaceSwapperTrain
from utils.utils_func import get_parser, save_model

def main(args):
    cfg = OmegaConf.load(args.config_path)

    if args.train:
        face_swapper_model = FaceSwapperTrain(cfg).to(cfg.train.device)

        sd = torch.load("ckpt/mavlast.ckpt")["state_dict"]
        face_swapper_model.load_state_dict(sd, strict=False)

        trainer = Trainer(max_epochs=500,
                          accelerator="cuda",
                          devices=1,
                          logger=CSVLogger("logs", name="exp02"),
                          callbacks=[ModelCheckpoint(dirpath="ckpt",
                                                     save_last=True,
                                                     verbose=True)])
        trainer.fit(face_swapper_model)
        save_model(face_swapper_model, cfg.train.model_name)

if __name__ == "__main__":
    args = get_parser()
    main(args)
