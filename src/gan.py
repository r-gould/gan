import math
import torch
import matplotlib.pyplot as plt
from statistics import mean

from .models.generator import Generator
from .models.discriminator import Discriminator

class GAN:
    def __init__(self, generator, discriminator, k=1):
        self.generator = generator
        self.discriminator = discriminator
        self.k = k
        self._compiled = False

    def compile(self, g_optim, d_optim):
        self.g_optim = g_optim
        self.d_optim = d_optim
        self._compiled = True

    def train(self, train_dl, epochs, batch_size):
        if not self._compiled:
            raise RunTimeError("GAN must be compiled before training.")

        def sample_batch():
            while True:
                for images, _ in train_dl:
                    yield images

        data_gen = sample_batch()
        batches_done, epochs_done = 0, 0
        epoch_g_losses, epoch_d_losses = [], []
        g_losses, d_losses = [], []
        done_epoch = False

        while epochs_done < epochs:

            for _ in range(self.k):
                """
                1. Sample batch from dataset.
                2. Generate random noise.
                3. Generate images from random noise.
                4. Calculate discriminator loss.
                5. Update weights of discriminator based on the loss.
                """

                real = next(data_gen)
                batch_size = real.shape[0]

                noise = self.generator.sample_noise(batch_size)
                generated = self.generator(noise)

                loss_d = self.discriminator.loss(real, generated.detach())
                epoch_d_losses.append(loss_d.item())

                self.d_optim.zero_grad()
                loss_d.backward()
                self.d_optim.step()

                batches_done += 1

                if batches_done >= len(train_dl):
                    done_epoch = True

            """
            1. Generate random noise.
            2. Generate images from random noise.
            3. Calculate discriminator probabilities (how likely the generated images are dataset images).
            4. Calculate generator loss.
            5. Update weights of generator based on the loss.
            """

            noise = self.generator.sample_noise(batch_size)
            generated = self.generator(noise)
            d_probs = self.discriminator(generated)

            loss_g = self.generator.loss(d_probs)
            epoch_g_losses.append(loss_g.item())
            
            self.g_optim.zero_grad()
            loss_g.backward()
            self.g_optim.step()

            """
            Stores loss values after each epoch for plotting.
            """

            if done_epoch:
                done_epoch = False
                epochs_done += 1
                batches_done = 0

                print("Epoch:", epochs_done)
                
                g_losses.append(mean(epoch_g_losses))
                d_losses.append(mean(epoch_d_losses))
                epoch_g_losses, epoch_d_losses = [], []

        return g_losses, d_losses
