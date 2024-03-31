import numpy as np
import os
import matplotlib.pyplot as plt
import trimesh


def plot_points(path):
    ax = plt.figure().add_subplot(projection="3d")
    obj = trimesh.load(path)
    x, y, z = obj.vertices[:, 0], obj.vertices[:, 1], obj.vertices[:, 2]
    mask = obj.colors[:, 1] == 255
    ax.scatter(
        x[mask], y[mask], zs=z[mask], zdir="y", alpha=1, c=obj.colors[mask] / 255
    )
    ax.scatter(
        x[~mask], y[~mask], zs=z[~mask], zdir="y", alpha=0.01, c=obj.colors[~mask] / 255
    )
    plt.show()


def download_data():
    import gdown

    if not os.path.exists("./data"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/1EKWU_daQL3pxFkjFUomGs25_qekyfeAd",
            quiet=False,
        )

    if not os.path.exists("./processed"):
        gdown.download_folder(
            "https://drive.google.com/drive/folders/175_LtuWh1LknbbMjUumPjGzeSzgQ4ett",
            quiet=False,
        )
