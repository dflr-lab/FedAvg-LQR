#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7


import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from sklearn.model_selection import train_test_split
from dataset import SyntheticDataset
from loss import LQRLoss, LQRLossU, LQRLossX


class LocalUpdate:
    """LocalUpdate for Client."""
    def __init__(self, args, robot_id, device='cuda'):
        self.args = args
        self.device = device
        self.robot_id = robot_id

        # Splitting the dataset into training and testing
        dataset = SyntheticDataset(robot_id)
        idxs = list(range(len(dataset)))
        idxs_train, idxs_test = train_test_split(idxs, test_size=0.2)  # Assuming 20% test split, 40:10

        self.trainloader = DataLoader(SyntheticDataset(robot_id, idxs_train), batch_size=args.local_bs, shuffle=True)
        self.testloader = DataLoader(SyntheticDataset(robot_id, idxs_test), batch_size=args.local_bs, shuffle=False)

    def update_weights(self, model):
        model.train()

        # Set optimizer for the local updates
        if self.args.optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.args.lr,
                                        momentum=0.5)
        elif self.args.optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr,
                                         weight_decay=1e-4)

        scheduler = StepLR(optimizer, step_size=1, gamma=0.98) # schedule learning rate

        epoch_loss = []

        for epoch in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (x, u) in enumerate(self.trainloader):
                x, u = x.to(self.device), u.to(self.device) # x: (batch_size x T x 2), u: (batch_size x T x 1)

                initial_states = x[:, 0]
                x_pred, u_pred = model(initial_states) # x_pred: (batch_size x T x 2), u_pred: (batch_size x T x 1)
                loss = LQRLoss(x_pred, x, u_pred, u)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0) # clip gradient avoiding gradient explosion
                optimizer.step()

                batch_loss.append(loss.item())

                if self.args.verbose and batch_idx % 2 == 0:
                    print('Robot {}, Local Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        self.robot_id,
                        epoch, batch_idx * len(x), len(self.trainloader.dataset),
                        100. * batch_idx / len(self.trainloader), loss.item()))

            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            scheduler.step() # schedule learning rate

        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), epoch_loss

    def inference(self, model):
        """ Returns the inference average loss."""
        model.eval()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for x, u in self.testloader:
                x, u = x.to(self.device), u.to(self.device)

                initial_states = x[:, 0]
                x_pred, u_pred = model(initial_states)
                loss = LQRLoss(x_pred, x, u_pred, u)
                total_loss += loss.item() * x.size(0) * x.size(1)  # Multiply by the batch size and time steps
                total_samples += x.size(0) * x.size(1)

        avg_loss = total_loss / total_samples
        return avg_loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss.
    """

    model.eval()
    loss, total, correct = 0.0, 0.0, 0.0

    device = 'cuda' if args.gpu else 'cpu'
    criterion = nn.NLLLoss().to(device)
    testloader = DataLoader(test_dataset, batch_size=128,
                            shuffle=False)

    for batch_idx, (images, labels) in enumerate(testloader):
        images, labels = images.to(device), labels.to(device)

        # Inference
        outputs = model(images)
        batch_loss = criterion(outputs, labels)
        loss += batch_loss.item()

        # Prediction
        _, pred_labels = torch.max(outputs, 1)
        pred_labels = pred_labels.view(-1)
        correct += torch.sum(torch.eq(pred_labels, labels)).item()
        total += len(labels)

    accuracy = correct/total
    return accuracy, loss
