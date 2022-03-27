import json
import torch
from . import utils
import torchvision
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ..model.cnn import CNNGenerator, CNNDiscriminator
from .DatasetClass import PokemonDataset
from ..diff_augmentation.DiffAugmentation import diff_augmentation


def train(G, D, lr, betas, num_epochs, dataloader, device, shape):
    # shape of images
    H, W, C = shape

    # optimizer
    optimizerG = optim.Adam(params=G.parameters(), lr=lr, betas=betas)
    optimizerD = optim.Adam(params=D.parameters(), lr=lr, betas=betas)

    # loss function
    criterion = nn.BCELoss()

    # terminal -> tensorboard --logdir=runs
    n_th_image = 4
    global_step = 0
    log_every_n_th_step = 50
    writer = SummaryWriter()

    G.train()
    D.train()

    for _ in tqdm(range(num_epochs)):
        for batch in dataloader:
            images_real, label_real = batch
            b_size = images_real.size(0)

            # input for the generator: z~N(0, 1)
            z = torch.randn(size=(b_size, 100, 1, 1), dtype=torch.float32, device=device)

            # B x C x H x W
            images_real = images_real.to(device)
            images_fake = G(z)

            # differential augmentation
            images_real, images_fake = diff_augmentation(images_real, images_fake)

            # B x 1
            label_real = label_real.to(device).to(torch.float32)
            label_fake = torch.zeros(b_size).to(device)

            # train discriminator: forward + backward + optimize
            optimizerD.zero_grad(set_to_none=True)
            real_part = criterion(D(images_real).squeeze(), label_real)
            fake_part = criterion(D(images_fake.detach()).squeeze(), label_fake)
            lossD = (real_part + fake_part).mean()
            lossD.backward()
            optimizerD.step()

            # train generator: forward + backward + optimize
            optimizerG.zero_grad(set_to_none=True)
            # Non-Saturating GAN Loss: log(D(G(z))
            lossG = (criterion(D(images_fake).squeeze(), label_real)).mean()
            # Minimax Gan Loss: (1-log(D(G(z))
            # lossG = -criterion(D(images_fake.detach()).squeeze(), label_fake)

            lossG.backward()
            optimizerG.step()

            # update tensorboard
            if global_step % log_every_n_th_step == 0:
                writer.add_scalar('lossD', lossD.detach().cpu().item(), global_step)
                writer.add_scalar('lossG', lossG.detach().cpu().item(), global_step)

                # [-1, 1] -> [0, 1]
                images = (0.5 * images_fake[: n_th_image] + 0.5)
                # images.clamp(0, 1).detach().cpu().reshape(log_every_n_th_step, C, H, W)
                grid = torchvision.utils.make_grid(images)
                writer.add_image('generator images', grid, global_step)

            global_step += 1

    utils.saveModel(G)
    writer.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="path of the config file", type=str, default="pokemon/generator/train_config.json")
    args = parser.parse_args()
    config_file = args.config_file

    with open(config_file, 'r') as fh:
        config = json.load(fh)

    # Main directory
    root_dir = config["root_dir"]

    # Hyperparameters
    batch_size = config["batch_size"]
    num_epochs = config["num_epochs"]
    num_workers = config["num_workers"]
    learning_rate = config["learning_rate"]
    betas = config["betas"]
    mean = config["mean"]
    std = config["std"]
    shape = utils.get_shape(root_dir=config["root_dir"], idx=0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # [0,1] -> [-1, 1]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])

    pokemon_dataset = PokemonDataset(root_dir=root_dir, transform=transform)
    poke_loader = DataLoader(pokemon_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    G = CNNGenerator().to(device)
    D = CNNDiscriminator().to(device)

    train(G, D, lr=learning_rate, betas=betas, num_epochs=num_epochs,
          dataloader=poke_loader, device=device, shape=shape)


if __name__ == "__main__":
    main()
