import os
import torch
import argparse
import datetime


def get_parser():
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--train", type=str2bool, const=True, default=False, nargs="?")
    args = parser.parse_args()

    return args

def save_model(face_swapper_model, model_name):
    now = datetime.datetime.now().strftime("%d%b-%H:%M:%S")
    folder_path = f"models/{now}"
    os.makedirs(folder_path, exist_ok=True)

    torch.save(face_swapper_model.face_swapper.state_dict(), f"{folder_path}/{model_name}")