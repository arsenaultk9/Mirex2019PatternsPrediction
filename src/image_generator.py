import matplotlib.pyplot as plt
import numpy as np


def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 128, 64))

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(noise[cnt, :], cmap='winter')
            axs[i, j].axis('off')
            cnt += 1

    fig.savefig("images/%d.png" % epoch)
    plt.close()
