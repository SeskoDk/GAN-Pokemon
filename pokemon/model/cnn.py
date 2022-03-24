import math
import torch
import torch.nn as nn
from typing import Type


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



class CNNGenerator_2(nn.Sequential):
    def __init__(
        self,
        num_in_chan: int = 100,
        num_out_chan: int = 3,
        num_h_chan=16,
        target_size: int = 128,
    ):
        num_layers = int(math.log2(target_size))
        in_sizes = [2 ** i for i in range(num_layers)]
        in_channels = [num_in_chan] + [num_h_chan] * (num_layers - 1)
        out_channels = [num_h_chan] * (num_layers - 1) + [num_out_chan]
        norms = [True] * (num_layers - 1) + [False]
        leaky_args = dict(negative_slope=0.2, inplace=True)
        activations = [(nn.LeakyReLU, leaky_args)] * (num_layers - 1) + [(nn.Tanh, {})]
        super().__init__(
            *[
                self._layer(inc, outc, ins, norm, act[0], **act[1])
                for inc, outc, ins, norm, act in zip(
                    in_channels, out_channels, in_sizes, norms, activations
                )
            ]
        )

    def _layer(
        self,
        num_in_chan: int,
        num_out_chan: int,
        in_size: int,
        norm: bool = True,
        act_klass: Type = nn.LeakyReLU,
        **act_params
    ) -> nn.Sequential:
        """Returns a CNN-Generator standard layer block."""
        layers = []
        layers.append(
            nn.ConvTranspose2d(
                in_channels=num_in_chan,
                out_channels=num_out_chan,
                kernel_size=(4, 4),
                stride=(2, 2),
                padding=(1, 1),
                bias=False,
            )
        )
        if norm:
            layers.append(nn.LayerNorm((num_out_chan, in_size * 2, in_size * 2)))
        if act_klass is not None:
            layers.append(act_klass(**act_params))
        return nn.Sequential(*layers)


# gen = CNNGenerator_2(num_in_chan=100, num_out_chan=3, num_h_chan=16, target_size=128)
# print(gen)
# print(gen(torch.rand(1, 100, 1, 1)).shape)

# Ramp-up
# gen = CNNGenerator_2(num_in_chan=100, num_out_chan=3, num_h_chan=32, target_size=2048)
# print(gen)
# print(gen(torch.rand(1, 100, 1, 1)).shape)