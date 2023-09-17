import cv2
import dlib
import torch
import numpy as np
from insightface.app import FaceAnalysis


class InsightFaceShapePredictor():
    def __init__(self, cfg):
        self.predictor = FaceAnalysis(allowed_modules=cfg.loss.insightface_allowed_modules)
        self.predictor.prepare(ctx_id=0, det_size=tuple(cfg.loss.insightface_input_size))

    def __preprocess__(self, img):
        img = img.permute(1, 2, 0).detach().cpu().numpy()

        img = (img + 1) * .5
        img = (img * 255).astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img
    
    def shape_to_torch(self, faces):
        coords = np.zeros((len(faces),) + (faces[0].landmark_2d_106.shape))

        for face_idx, face in enumerate(faces):
            att = face.landmark_2d_106
            att = np.round(att).astype(np.int32)
        
            for att_idx in range(att.shape[0]): 
                coords[face_idx][att_idx] = att[att_idx]

        coords = torch.from_numpy(coords)
        return coords

    def __call__(self, img, single_face=False, preprocess=True):
        if preprocess:
            img = self.__preprocess__(img)
        with torch.no_grad():
            faces = self.predictor.get(img)
        if len(faces) == 0:
            return torch.zeros([106, 2], dtype=torch.int)
        return self.shape_to_torch(faces if not single_face else [faces[0]])
    

class DLibShapePredictor:
    def __init__(self, cfg):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(cfg.loss.face_attribute_detector)

    def __preprocess__(self, img):
        img = img.permute(1, 2, 0).detach().cpu().numpy()

        img = (img + 1) * .5
        img = (img * 255).astype("uint8")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        return img
    
    def shape_to_torch(self, img, faces):
        coords = np.zeros((len(faces),) + (68, 2))

        for face_idx, face in enumerate(faces):
            att = self.predictor(img, face)
            for i in range(0, 68):
                coords[face_idx][i][0] = att.part(i).x
                coords[face_idx][i][1] = att.part(i).y

        coords = torch.from_numpy(coords)
        return coords

    def __call__(self, img, single_face=False, preprocess=True):
        if preprocess:
            img = self.__preprocess__(img)
        faces = self.detector(img)
        if len(faces) == 0:
            return torch.zeros([68, 2], dtype=torch.int)
        return self.shape_to_torch(img, faces if not single_face else [faces[0]])
