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

# Hyperprams
lr = 0.0001
MAX_ITERS = 100000
batch_size = 16

# GSs and PDs
GSs = [(20, 20)]
PDs = [np.random.uniform]
ARGS = [(0, 100)]

GSs_testing = [(25, 25), (35, 35)]
PDs_testing = [np.random.normal]
ARGS_testing = [(75, 25)]

# Loss
LOSS = dict()
for GS in GSs:
    LOSS[GS]=dict()
    for args in ARGS:
        LOSS[GS][args]=list()

LOSS_testing = dict()
for GS in GSs_testing:
    LOSS_testing[GS]=dict()
    for args in ARGS_testing:
        LOSS_testing[GS][args]=list()

class BichannelCNN(nn.Module):
    def __init__(self):
        super(BichannelCNN, self).__init__()
        self.conv1 = nn.Conv2d(2, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)

        self.convx1 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)
        self.convx2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn12 = nn.BatchNorm2d(16)
        self.convx3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn13 = nn.BatchNorm2d(16)
        self.convx4 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn14 = nn.BatchNorm2d(8)

        self.convy1 = nn.Conv2d(64, 32, 3, padding=1)
        self.bn21 = nn.BatchNorm2d(32)
        self.convy2 = nn.Conv2d(32, 16, 3, padding=1)
        self.bn22 = nn.BatchNorm2d(16)
        self.convy3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn23 = nn.BatchNorm2d(16)
        self.convy4 = nn.Conv2d(16, 8, 3, padding=1)
        self.bn24 = nn.BatchNorm2d(8)

    def forward(self, bg):
        bg = F.leaky_relu(self.bn1(self.conv1(bg)))
        bg = F.leaky_relu(self.bn2(self.conv2(bg)))
        bg = F.leaky_relu(self.bn3(self.conv3(bg)))
        bg = F.leaky_relu(self.bn4(self.conv4(bg)))

        x = F.leaky_relu(self.bn11(self.convx1(bg)))
        x = F.leaky_relu(self.bn12(self.convx2(x)))
        x = F.leaky_relu(self.bn13(self.convx3(x)))
        x = F.leaky_relu(self.bn14(self.convx4(x)))
        x = x.mean((1, 3))
        x = F.softmax(x, 1)

        y = F.leaky_relu(self.bn21(self.convy1(bg)))
        y = F.leaky_relu(self.bn22(self.convy2(y)))
        y = F.leaky_relu(self.bn23(self.convy3(y)))
        y = F.leaky_relu(self.bn24(self.convy4(y)))
        y = y.mean((1, 2))
        y = F.softmax(y, 1)

        return x, y

def loss_batch(BGs, TOVs, GS):
    xs, ys = net(BGs)
    xs = xs.reshape((batch_size, 1, GS[0]))
    ys = ys.reshape((batch_size, GS[1], 1))
    POV1s = torch.bmm(torch.bmm(xs, BGs[:, 0, :, :]), ys).reshape((batch_size, 1))
    POV2s = torch.bmm(torch.bmm(xs, BGs[:, 1, :, :]), ys).reshape((batch_size, 1))
    POVs = torch.cat((POV1s, POV2s), 1)
    loss = mse(POVs, TOVs)

    return loss

def train(GSs, PDs, ARGS):
    iter = 0
    while True:
        for GS in GSs:
            for i in range(len(PDs)):
                PD, args = PDs[i], ARGS[i]
                # Generate a batch of data
                BGs, TOVs = BGs_generate(GS, PD, *args)

                BGs = torch.tensor(BGs).reshape((batch_size, 2, *GS)).float()
                TOVs = torch.tensor(TOVs).reshape((batch_size, 2)).float()
                if CUDA:
                    BGs, TOVs = BGs.cuda(), TOVs.cuda()

                # Train the model
                optimizer.zero_grad()
                loss = loss_batch(BGs, TOVs, GS)
                loss.backward()
                optimizer.step()
                LOSS[GS][args].append(round(loss.item(), 3))

                # Test on untrained GSs and PDs
                if iter%120==0:
                    for GS_tesing in GSs_testing:
                        for j in range(len(PDs_testing)):
                            PD_testing, args_testing = PDs_testing[j], ARGS_testing[j]
                            BGs, TOVs = BGs_generate(GS_tesing, PD_testing, *args_testing)
                            BGs = torch.tensor(BGs).reshape((batch_size, 2, *GS_tesing)).float()
                            TOVs = torch.tensor(TOVs).reshape((batch_size, 2)).float()
                            if CUDA:
                                BGs, TOVs = BGs.cuda(), TOVs.cuda()
                            loss = loss_batch(BGs, TOVs, GS_tesing)
                            LOSS_testing[GS_tesing][args_testing].append(round(loss.item(), 3))
                            print(f'-------------------------Testing: Loss: {loss.item():.2f}, GS: {GS_tesing}, PD: {args_testing}-------------------------')

                            f = open('/content/drive/MyDrive/No.3  Phd Day/results/ablation study/LOSS_testing.pkl',"wb")
                            pickle.dump(LOSS_testing,f)
                            f.close()

                # Save resutls
                if iter%100==0:
                    torch.save(net.state_dict(), '/content/drive/MyDrive/No.3  Phd Day/results/ablation study/net.pt')
                    f = open('/content/drive/MyDrive/No.3  Phd Day/results/ablation study/LOSS.pkl',"wb")
                    pickle.dump(LOSS,f)
                    f.close()

                print(f"Iteration: {iter}, Loss: {loss.item():.2f}, GS: {GS}, PD: {args}")
                iter += 1
                if iter>MAX_ITERS:
                    return
