import os
import cv2
import torch
import random
import albumentations as A
from torch.utils.data import Dataset

def bgr2rgb(image, **kwargs):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def to_tensor(image, **kwargs):
    return torch.from_numpy(image)


def transpose(image, **kwargs):
    return image.transpose(2, 0, 1)

def to_float(image, **kwargs):
    return image.type(torch.float)


class VGGFace2(Dataset):
    same_person = True

    def __init__(self, cfg):
        super().__init__()

        self.img_list, self.folder_img_dict = self.__get_data__(cfg.data.dataset_path)
        self.transform = A.Compose([A.Lambda(name="BGR2RGB", image=bgr2rgb),
                                    A.Resize(height=cfg.data.img_height, width=cfg.data.img_width),
                                    A.Normalize(mean=cfg.data.norm_mean, std=cfg.data.norm_std, max_pixel_value=255.),
                                    A.Lambda(name="ChannelTranspose", image=transpose),
                                    A.Lambda(name="ToTensor", image=to_tensor),
                                    A.Lambda(name="ToFloat", image=to_float)])

    def __len__(self):
        return len(self.img_list)

    def __get_data__(self, dataset_path):
        img_list = []
        folder_img_dict = {}
        folder_list = os.listdir(dataset_path)
        for folder in folder_list:
            if not os.path.isdir(os.path.join(dataset_path, folder)):
                continue

            folder_img_dict[folder] = []
            for img in os.listdir(os.path.join(dataset_path, folder)):
                full_path = os.path.join(dataset_path, folder, img)
                img_list.append(full_path)
                folder_img_dict[folder].append(full_path)

        return img_list, folder_img_dict
    
    def random_choice_and_read(self, folder_name):
        img_path = random.choice(self.folder_img_dict[folder_name])
        img = cv2.imread(img_path)
        return img
    
    def get_target_img(self, source_folder_name):
        while True:
            target_folder_name = random.choice(self.folder_img_dict.keys())
            if target_folder_name != source_folder_name:
                break

        target_img = self.random_choice_and_read(target_folder_name)
        return target_img
    
    def __getitem__(self, item):
        source_img_path = self.img_list[item]
        source_folder_name = source_img_path.split("/")[-2]
        source_img = cv2.imread(source_img_path)

        # same_person = random.choices([0, 1], [1-self.same_prob, self.same_prob], k=1)[0]
        if self.same_person:
            target_img = self.random_choice_and_read(source_folder_name)
            self.same_person = False

        else:
            target_img = self.get_target_img(source_folder_name)
            self.same_person = True

        source_img = self.transform(image=source_img)["image"]
        target_img = self.transform(image=target_img)["image"]

        return source_img, target_img, self.same_person