import torch.nn as nn


class CNNGenerator(nn.Sequential):
    def __init__(self, in_channels=100, out_channels=3, hidden_channels=16):
        super().__init__(
            # Input: B x 100 x 1 x 1
            nn.ConvTranspose2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LayerNorm((hidden_channels, 2, 2)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 2 x 2
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LayerNorm((hidden_channels, 4, 4)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 4 x 4
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LayerNorm((hidden_channels, 8, 8)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 8 x 8
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LayerNorm((hidden_channels, 16, 16)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 16 x 16
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LayerNorm((hidden_channels, 32, 32)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 32 x 32
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.LayerNorm((hidden_channels, 64, 64)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 64 x 64
            nn.ConvTranspose2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
            nn.Tanh()
            # Output: B x 3 x 128 x 128
        )


class CNNDiscriminator(nn.Sequential):
    def __init__(self, in_channels=3, out_channels=1, hidden_channels=16):
        # B x 3 x 128 x 128 -> B x 3 x 1x1
        super().__init__(
            # Input: B x 3 x 128 x 128
            nn.Conv2d(in_channels=in_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm((hidden_channels, 64, 64)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 64 x 64
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm((hidden_channels, 32, 32)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 32 x 32
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm((hidden_channels, 16, 16)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 16 x 16
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm((hidden_channels, 8, 8)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 8 x 8
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm((hidden_channels, 4, 4)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 4 x 4
            nn.Conv2d(in_channels=hidden_channels, out_channels=hidden_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.LayerNorm((hidden_channels, 2, 2)),
            # nn.BatchNorm2d(num_features=hidden_channels),
            nn.LeakyReLU(0.2, inplace=True),

            # Input: B x H_C x 2 x 2
            nn.Conv2d(in_channels=hidden_channels, out_channels=out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
            nn.Sigmoid()
            # Output: B x 1 x 1 x 1
        )
