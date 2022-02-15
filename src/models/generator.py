import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_size, output_size):
        super().__init__()

        self.noise_size = noise_size

        self.network = nn.Sequential(
            nn.Linear(noise_size, 128),
            nn.LeakyReLU(0.2),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),

            nn.Linear(1024, output_size),
            nn.Tanh(),
        )

    def forward(self, noise):
        return self.network(noise)

    def generate(self, count, output_shape=None):
        with torch.no_grad():
            noise = self.sample_noise(count)
            output = self.forward(noise)
            if output_shape:
                return output.reshape(count, *output_shape)
            return output

    def sample_noise(self, count):
        return torch.randn(count, self.noise_size)

    def loss(self, d_probs):
        eps = 1e-6
        return -torch.mean(torch.log(d_probs+eps))