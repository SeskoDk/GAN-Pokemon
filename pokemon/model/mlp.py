import torch.nn as nn


class MLPGenerator(nn.Module):
    def __init__(self, n_input, n_output):
        super(MLPGenerator, self).__init__()
        self.main = nn.Sequential(nn.Linear(n_input, 200),
                                  nn.LeakyReLU(0.1),
                                  nn.Linear(200, 400),
                                  nn.LeakyReLU(0.1),
                                  nn.Linear(400, 600),
                                  nn.LeakyReLU(0.1),
                                  nn.Linear(600, n_output),
                                  nn.Tanh())

    def forward(self, x):
        return self.main(x)


class MLPDiscriminator(nn.Module):
    def __init__(self, n_input):
        super(MLPDiscriminator, self).__init__()
        self.main = nn.Sequential(nn.Linear(n_input, 100),
                                  nn.LeakyReLU(0.1),
                                  nn.Linear(100, 25),
                                  nn.LeakyReLU(0.1),
                                  nn.Linear(25, 1),
                                  nn.Sigmoid())

    def forward(self, x):
        return self.main(x)

