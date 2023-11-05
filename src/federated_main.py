#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.7


import os
import copy
import time
import pickle
import numpy as np
from tqdm import tqdm

import torch
from tensorboardX import SummaryWriter

from options import args_parser
from update import LocalUpdate, test_inference
from models import DynamicsModel
from utils import average_weights, exp_details


if __name__ == '__main__':
    start_time = time.time()

    # define paths
    path_project = os.path.abspath('..')
    logger = SummaryWriter('../logs')

    args = args_parser()
    exp_details(args)

    if args.gpu:
        torch.cuda.set_device(int(args.gpu))
    device = 'cuda' if args.gpu else 'cpu'


    # BUILD MODEL
    if args.model == 'dynamics model':
         # Initialize parameters of dynamics model (A, B, K).
        A = np.ones((2, 2))
        B = np.ones((2, 1))
        K = np.ones((1, 2))

        # Initialize the model with the correct device and  set the model to train
        global_model = DynamicsModel(A, B, K, device)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    print(global_model)

    torch.autograd.set_detect_anomaly(True)

    # copy weights
    global_weights = global_model.state_dict()

    # Training
    train_loss, val_loss = [], []
    print_every = 2

    for epoch in tqdm(range(args.epochs)):
        local_weights, local_losses = [], []
        print(f'\n | Global Training Round : {epoch+1} |\n')

        global_model.train()
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(1, args.num_users + 1), m, replace=False)

        # Construct all the local clients with its local dataset
        local_updates = [LocalUpdate(args=args, robot_id=robot_id, device=device) for robot_id in idxs_users]

        # Start local update on clients' training data
        for i in range(len(local_updates)):
            local_update = local_updates[i]
            
            # Train the model locally
            w, loss, _ = local_update.update_weights(model=copy.deepcopy(global_model)) # distribute global model

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss))

        # update global weights
        global_weights = average_weights(local_weights)

        # update global weights
        global_model.load_state_dict(global_weights)

        loss_avg_train = sum(local_losses) / len(local_losses)
        train_loss.append(loss_avg_train)

        # Calculate avg test loss over all users at every epoch
        list_loss = []
        
        global_model.eval()
        for i in range(len(local_updates)):
            local_update = local_updates[i]

            loss = local_update.inference(model=global_model)
            list_loss.append(copy.deepcopy(loss))
        
        loss_avg_test = sum(list_loss) / len(list_loss)
        val_loss.append(loss_avg_test)

        # print global training loss after every 'i' rounds
        if (epoch+1) % print_every == 0:
            print(f' \nTraining Stats after {epoch+1} global rounds:')
            print(f'Training Loss : {np.array(loss_avg_train)}')
            print(f'Testing Loss : {np.array(loss_avg_test)}')

    # # Test inference after completion of training
    # test_acc, test_loss = test_inference(args, global_model, test_dataset)

    # print(f' \n Results after {args.epochs} global rounds of training:')
    # print("|---- Avg Train Accuracy: {:.2f}%".format(100*train_accuracy[-1]))
    # print("|---- Test Accuracy: {:.2f}%".format(100*test_acc))

    # # Saving the objects train_loss and train_accuracy:
    # file_name = '../save/objects/{}_{}_{}_C[{}]_iid[{}]_E[{}]_B[{}].pkl'.\
    #     format(args.dataset, args.model, args.epochs, args.frac, args.iid,
    #            args.local_ep, args.local_bs)

    # with open(file_name, 'wb') as f:
    #     pickle.dump([train_loss, train_accuracy], f)

    print('\n Total Run Time: {0:0.4f}'.format(time.time()-start_time))

    # PLOTTING (optional)
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use('Agg')

    # Plot Loss curve
    plt.figure()
    plt.title('Training Loss vs Communication rounds')
    plt.plot(range(len(train_loss)), train_loss, color='r')
    plt.ylabel('Training loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_E[{}]_B[{}]_training_loss.png'.
                 format(args.dataset, args.model, args.epochs, args.frac,
                         args.local_ep, args.local_bs))

     # Plot Testing Loss vs Communication rounds
    plt.figure()
    plt.title('Testing Loss vs Communication rounds')
    plt.plot(range(len(val_loss)), val_loss, color='k')
    plt.ylabel('Testing Loss')
    plt.xlabel('Communication Rounds')
    plt.savefig('../save/fed_{}_{}_{}_C[{}]_E[{}]_B[{}]_testing_loss.png'.
                format(args.dataset, args.model, args.epochs, args.frac,
                       args.local_ep, args.local_bs))
