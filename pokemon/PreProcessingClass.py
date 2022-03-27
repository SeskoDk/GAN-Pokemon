import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path


class PreProcessing():
    def __init__(self, root_dir: str, new_dir: str, transform=None, RGB=True):
        self.root_dir = root_dir
        self.new_dir = new_dir
        self.transform = transform
        self.RGB = RGB

    def pre_process_images(self):
        self._create_new_folder()
        self._store_images()

    def _create_new_folder(self):
        file = Path(self.new_dir)
        if file.exists():
            shutil.rmtree(file)
            file.mkdir()
        else:
            file.mkdir()

    def _load_images(self) -> np.ndarray:
        path = Path(self.root_dir).rglob("*.png")
        files = np.array([np.array(Image.open(img)) for img in tqdm(path, desc="Load images")])
        return files[:, :, :, :3] if self.RGB else files

    def _store_images(self):
        data = self._load_images()
        for i, image in tqdm(enumerate(data), desc="Preprocess images"):
            name = "{0:03d}.png".format(i + 1)
            path = Path(self.new_dir, name)
            image = Image.fromarray(image)
            image = self.transform(image)
            image.save(path)
        print("Done!")
