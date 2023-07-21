from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from trainer import FaceSwapperTrain
from utils.utils_func import get_parser, save_model

def main(args):
    cfg = OmegaConf.load(args.config_path)

    if args.train:
        face_swapper_model = FaceSwapperTrain(cfg)

        trainer = Trainer(max_epochs=500,
                          accelerator="mps",
                          devices=1)
        trainer.fit(face_swapper_model)
        save_model(face_swapper_model, cfg.train.model_name)

if __name__ == "__main__":
    args = get_parser()
    main(args)