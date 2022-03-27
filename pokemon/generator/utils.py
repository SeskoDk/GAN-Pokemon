import torch
import shutil
import datetime
import numpy as np
from PIL import Image
from pathlib import Path
from ..model.cnn import CNNGenerator

def create_new_folder(dir):
    if dir.exists():
        shutil.rmtree(dir)
        dir.mkdir(parents=True, exist_ok=True)
    else:
        dir.mkdir(parents=True, exist_ok=True)


def saveModel(model):
    modelDir = Path('../trained_models')
    modelDir.mkdir(parents=True, exist_ok=True)
    modelName = '{:%Y-%m-%d_%H_%M_%S}.pt'.format(datetime.datetime.now())
    torch.save(model.state_dict(), modelDir / modelName)


def loadModel(modelPath: str, device: torch.device):
    model = CNNGenerator()
    model.load_state_dict(torch.load(modelPath, map_location=device))
    model.eval()
    return model


def get_shape(root_dir: str, idx: str):
    path = "{root_dir}/{image}.png".format(root_dir=root_dir, image=f"{idx+1}".zfill(3))
    return np.array(Image.open(path)).shape
