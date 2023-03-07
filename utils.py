import os
import pandas as pd
import numpy as np


def merge(dir):

    for (_, dirs, files) in os.walk(dir, topdown=True):
        print()

    x = pd.read_csv(os.path.join(dir, files[0]))
    x = x.to_numpy()

    for i in range(len(files)-1):
        f2 = pd.read_csv(os.path.join(dir, files[i+1]))
        f2 = f2.to_numpy()
        f2[f2 == 1] = i+2
        x = np.concatenate((x, f2))

    return x


if __name__ == "__main__":
    x = merge('data')
