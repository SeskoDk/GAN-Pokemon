import torch
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from .generator import utils


def generate_new_images(model, folder, n_images):
    utils.create_new_folder(folder)
    for i in tqdm(range(n_images), desc="Generate new images"):
        z = torch.randn(size=(1, 100, 1, 1), dtype=torch.float32)
        images = model(z)
        images = (0.5 * images + 0.5)
        img = images[0].reshape(3, 128, 128)
        img = img.permute(1, 2, 0).detach().numpy()

        plt.imsave(Path(folder, f"{i}.png"), img)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_new_images", help="path of the generated images", type=str, default="data/generated_images")
    parser.add_argument("--n_images", help="Number of generated images", type=int, default=10)
    parser.add_argument("--model", help="path of the trained generator", type=str, default="pokemon/trained_models/2022-03-08_15_49_59.pt")
    args = parser.parse_args()

    path_new_images = Path(args.path_new_images)
    model_path = args.model
    n_images = args.n_images

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = utils.loadModel(model_path, device)

    generate_new_images(model=generator,  folder=path_new_images,  n_images=n_images)
