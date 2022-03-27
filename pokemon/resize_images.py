from torchvision.transforms import transforms
from pokemon.PreProcessingClass import PreProcessing


def main(path_of_dataset, path_of_new_dataset, height, width):
    transform = transforms.Compose([
        transforms.Resize(size=(height, width), interpolation=transforms.InterpolationMode.BILINEAR)
    ])
    preprocess = PreProcessing(root_dir=path_of_dataset, new_dir=path_of_new_dataset, transform=transform, RGB=True)
    preprocess.pre_process_images()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--height", help="height of the new image", type=int, default=128)
    parser.add_argument("--width",  help="width of the new image",  type=int, default=128)
    parser.add_argument("--path_of_dataset", help="path of the original dataset", type=str, default="pokemon/data/pokemon_dataset")
    parser.add_argument("--path_of_preprocessed_pokemon", help="path of the preprocessed pokemon", type=str, default="pokemon/data/pokemon_preprocessed")
    args = parser.parse_args()

    main(path_of_dataset=args.path_of_dataset,
         path_of_new_dataset=args.path_of_preprocessed_pokemon,
         height=args.height,
         width=args.width)
