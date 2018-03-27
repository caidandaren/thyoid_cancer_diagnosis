import numpy as np
from matplotlib import pyplot as plt
import __init__
from data_loader.path_manager import PATH

import pandas as pd

def fuse(id):
    fname = PATH.raw_info
    dataset = pd.read_csv(fname)
    npz_loc = dataset.loc[id, 'npz_loc']
    x = dataset.loc[id, 'X']
    y = dataset.loc[id, 'Y']
    z = dataset.loc[id, 'Z']
    img = np.load(npz_loc)
    plt.figure(figsize=(15, 15))
    plt.subplot(2, 2, 1)
    plt.imshow(img['pet'][z])
    plt.subplot(2, 2, 2)
    plt.imshow(img['ct'][z])
    plt.subplot(2, 2, 3)
    plt.imshow(img['ct'][z], plt.cm.gray)
    plt.imshow(img['pet'][z], plt.cm.afmhot, alpha=0.5)
    plt.scatter(x, y, color='', marker='o', edgecolors='w', s=200)
    plt.title(1)
    plt.show()

fuse(194)