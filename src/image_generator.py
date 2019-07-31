import matplotlib.pyplot as plt
import numpy as np


def sample_images(name, data):
    fig, axs = plt.subplots(data.shape[0])

    for row_index in range(data.shape[0]):
        axs[row_index].imshow(data[row_index], cmap='winter')
        axs[row_index].axis('off')

    fig.savefig("images/%s.png" % name)
    plt.close()


def sample_image(name, data):
    fig, axs = plt.subplots()
    axs.imshow(data, cmap='winter')
    axs.axis('off')

    fig.savefig("images/%s.png" % name)
    plt.close()
