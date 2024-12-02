#!/usr/bin/env python

from alpha_net import ChessNet, train
import os
import pickle
import numpy as np
import torch
import torch.multiprocessing as mp


if __name__=="__main__":
    net_to_train="current_net_trained_iter2.pth.tar"; save_as="current_net_trained_iter3.pth.tar"
    # gather data
    data_path = "./datasets/iter1/"
    datasets = []
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))

    data_path = "./datasets/iter2/"
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))

    data_path = "./datasets/mcts/"
    for idx,file in enumerate(os.listdir(data_path)):
        filename = os.path.join(data_path,file)
        with open(filename, 'rb') as fo:
            datasets.extend(pickle.load(fo, encoding='bytes'))



    # initiate Net
    mp.set_start_method("spawn",force=True)
    net = ChessNet()
    cuda = torch.cuda.is_available()
    if cuda:
        net.cuda()
    net.share_memory()
    net.train()
    print("hi")
    current_net_filename = os.path.join("./model_data/",\
                                    net_to_train)
    checkpoint = torch.load(current_net_filename,weights_only=False)
    net.load_state_dict(checkpoint['state_dict'])
    
    processes = []
    for i in range(3):
        p = mp.Process(target=train,args=(net,datasets,0,100,i))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    # save results
    torch.save({'state_dict': net.state_dict()}, os.path.join("./model_data/",\
                                    save_as))
