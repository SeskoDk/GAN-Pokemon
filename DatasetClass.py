import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from torchvision import transforms
from torch.utils.data import Dataset


class PokemonDataset(Dataset):
    def __init__(self, root_dir, transform=None, RGB=True):
        self.root_dir = root_dir
        self.transform = transform
        self.RGB = RGB
        self.files = self._load_images()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx) -> Tuple:
        idx = "{0:03d}".format(idx+1)
        filename = "{root_dir}/{idx}.png".format(root_dir=self.root_dir, idx=idx)
        image = Image.open(filename)

        if self.transform:
            image = self.transform(image)
        return image, 1

    def _load_images(self) -> np.ndarray:
        path = Path(self.root_dir).rglob("*.png")
        files = np.array([np.array(Image.open(img)) for img in tqdm(path, desc="Load images")])
        if self.RGB:
            files = files[:, :, :, :3]
        return files


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    dataset = PokemonDataset(root_dir="data/pokemon_small", transform=transform, RGB=True)
    pass


