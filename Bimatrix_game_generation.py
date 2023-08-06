# Import packages
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
from torch.autograd import grad
import nashpy as nash
import pickle
import time

np.set_printoptions(suppress=True)
CUDA = torch.cuda.is_available()
mse = nn.MSELoss()


# Generation of a bimatrix game
def BG_generate(GS, PD, *args):
    size = (2, GS[0], GS[1])
    BG = PD(*args, size=size)
    return BG

# Obtain the optimal value of the bimatrix game
def BG_solve(BG):
    GS = BG.shape[1:]
    BG = nash.Game(BG[0], BG[1])
    equilibria = BG.lemke_howson_enumeration()

    for eq in equilibria:
        x, y = eq
        if ~np.isnan(x).all() and ~np.isnan(y).all():
            if (x>=0).all() and (y>=0).all():
                if x.sum()==1 and y.sum()==1:
                    if x.shape[0]==GS[0] and y.shape[0]==GS[1]:
                        return x, y
    return np.nan

# Generate a batch of non-degenerate bimatrix games
def BGs_generate(batch_size, GS, PD, *args):
    BGs = np.zeros((batch_size, 2, *GS))
    TOVs = np.zeros((batch_size, 2))
    i=0
    while True:
        BG = BG_generate(GS, PD, *args)
        try:
            NE = BG_solve(BG)
        except RuntimeError as error:
            continue
        if NE is np.nan:
            continue
        if np.isnan(NE[0]).any() or np.isnan(NE[1]).any():
            continue

        x, y = NE
        BGs[i] = BG
        TOVs[i] = x.T@(BGs[i][0]@y), x.T@(BGs[i][1]@y)

        i += 1
        if i==batch_size:
            break
        print(i)
    return BGs, TOVs
