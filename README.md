# IEEETAI-BiCNN

This repository is dedicated to paper "Predicting Nash Equilibria in Bimatrix Games using a Robust Bi-channel Convolutional Neural Network" submitted to IEEE Transactions on Artificial Intelligence.

------

## BiCNN_Training.py
It shows how to implement the BiCNN model proposed in Section 3, and how to train the model with the proposed algorithm in Section 4.

## Bimatrix_game_generation.py
It shows how to generate a bimatrix game used for training. It implements the LH algorithm described in Section 2.2

## Abalation_study.py
It relates to Section 5.3 of the manuscript.
This script is similar to BiCNN_Training.py, except that here we train using only one pair of GS-PDs.

## TS_DFM.py
It relates to Section 5.4 of the manuscript, where we implement the TS and DFM algorithms.
