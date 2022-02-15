import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, input_size):
        super().__init__()

        self.input_size = input_size

        self.network = nn.Sequential(
            nn.Flatten(),
            
            nn.Linear(input_size, 512),
            nn.LeakyReLU(0.2),

            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),

            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, images):
        return self.network(images)

    def loss(self, real, generated):
        eps = 1e-6
        real_probs = self(real)
        generated_probs = self(generated)
        return -torch.mean(torch.log(real_probs+eps) + torch.log(1 - generated_probs+eps))