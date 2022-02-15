import torch
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms

from src.gan import GAN
from src.models.generator import Generator
from src.models.discriminator import Discriminator
from utils import plot_loss, visualize

def main():
    """
    Application of GAN on the MNIST dataset
    """

    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
        ]
    )
    train_ds = MNIST(root=".data", train=True, download=True, transform=transform)
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    num_examples, height, width = train_ds.data.shape
    image_size = height * width
    noise_size = 100
    
    generator = Generator(noise_size, image_size)
    discriminator = Discriminator(image_size)
    g_optim = torch.optim.Adam(generator.parameters(), eps=5e-4)
    d_optim = torch.optim.Adam(discriminator.parameters(), eps=5e-4)

    gan = GAN(generator, discriminator, k=1)
    gan.compile(g_optim, d_optim)
    g_losses, d_losses = gan.train(train_dl, 1, 64)

    plot_loss(g_losses, d_losses)
    visualize(generator, (5, 4), (height, width))

if __name__ == "__main__":
    main()