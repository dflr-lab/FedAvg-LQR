#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7


import numpy as np

import torch
import matplotlib.pyplot as plt

from update import LocalUpdate
from options import args_parser
from update import test_inference
from models import DynamicsModel


if __name__ == '__main__':
    args = args_parser()
    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'

    # Initialize parameters of dynamics model (A, B, K).
    A = np.ones((2, 2))
    B = np.ones((2, 1))
    K = np.ones((1, 2))

    # Initialize the model with the correct device and  set the model to train
    model = DynamicsModel(A, B, K, device)
    model.train()

    # Set training parameters
    robot_id = 1
    torch.autograd.set_detect_anomaly(True)

    # Initialize local update class
    local_update = LocalUpdate(args=args, robot_id=robot_id, device=device)

    # Train the model locally
    _, avg_train_loss, epoch_loss = local_update.update_weights(model=model)
    print(f'Average Training Loss: {avg_train_loss}')

    for name, param in model.named_parameters():
        print(name, param.data)

    # Inference on test data
    avg_test_loss = local_update.inference(model=model)
    print(f'Average Test Loss: {avg_test_loss}')

    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Training LQR loss')
    plt.savefig('../save/baseline_dynamics_model_{}.png'.format(args.local_ep))

    # # testing
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)
    # print('Test on', len(test_dataset), 'samples')
    # print("Test Accuracy: {:.2f}%".format(100*test_acc))
