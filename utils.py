import matplotlib.pyplot as plt

def plot_loss(g_losses, d_losses):
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.plot(g_losses, color="green", label="Generator")
    plt.legend(loc="upper right")

    plt.plot(d_losses, color="red", label="Discriminator")
    plt.legend(loc="upper right")

    plt.show()

def visualize(generator, size, output_shape):
    rows, cols = size
    count = rows * cols
    generated = generator.generate(count, output_shape)
    figure, axis = plt.subplots(rows, cols)

    for r in range(rows):
        for c in range(cols):
            image = generated[c + r*cols]
            axis[r, c].imshow(image, cmap="gray")
    plt.show()