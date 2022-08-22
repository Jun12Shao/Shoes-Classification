# -*- coding:utf-8 -*-
import numpy as np
import argparse
from torch.backends import cudnn
import pickle
import torch
import torch.onnx
import torch.nn as nn
from torch.autograd import Variable
from data_loader import get_loader
import time
from models.CNNs import CNN, CNN2
from models.Resnet import resnet18


def compute_precision_recall_and_F1(y, prec):
    results=np.zeros((3,3))
    N=len(y)
    a= float(sum((y==0)&(prec==0)))
    b= float(sum((y==1)&(prec==0)))
    c= float(sum((y==2)&(prec==0)))
    d= float(sum((y==0)&(prec==1)))
    e= float(sum((y==1)&(prec==1)))
    f= float(sum((y==2)&(prec==1)))
    g= float(sum((y==0)&(prec==2)))
    h= float(sum((y==1)&(prec==2)))
    l= float(sum((y==2)&(prec==2)))

    ### precision, recall and F1 for adidas class
    if a+b+c>0 and a+d+g>0:
        p=a/(a+b+c)
        r = a / (a + d + g)

        results[0, 0]=p
        results[0, 1]=r
        results[0,2]=2*p*r/(p+r)

    ### precision, recall and F1 for converse class
    if d + e + f > 0 and b + e + h > 0:
        p = e / (d + e + f)
        r = e / (b + e + h)
        results[1, 0] = p
        results[1, 1] = r
        results[1, 2] = 2 * p * r / (p + r)

    ### precision, recall and F1 for nike class
    if g + h + l > 0 and c + f + l > 0:
        p = l / (g + h + l)
        r = l / (c + f + l)
        results[2, 0] = p
        results[2, 1] = r
        results[2, 2] = 2 * p * r / (p + r)

    return results


def get_n_params(model):
    N=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        N += nn
    size = N / 1e6
    print("The number of weights is %f M" % size)

    return N


def update_learning_rate(config, optimizer,lr, lr_decay):
    # updated learning rate G
    lr2 =lr*lr_decay
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr2
    print('update G learning rate: %f -> %f' %  (lr, lr2 ))
    config.learning_rate=lr2



def main():
    parser = argparse.ArgumentParser(description='Train Fully Convolutional Neural Networks')

    # train args
    parser.add_argument('--epochs', type=int, default=150, metavar='NEPOCHS',   help='number of epochs to train (default: 10000)')
    parser.add_argument('--optim_method', type=str, default='Adam', metavar='OPTIM',  help='optimization method (default: Adam)')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',  help='learning rate (default: 0.0001)')
    parser.add_argument('--decay_every', type=int, default=20, metavar='LRDECAY',  help='number of epochs after which to decay the learning rate')
    default_weight_decay = 0.5
    parser.add_argument('--weight_decay', type=float, default=default_weight_decay, metavar='WD', help="weight decay (default: {:f})".format(default_weight_decay))
    parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size')
    parser.add_argument('--serial_batches', default=False, help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    parser.add_argument('--method', type=str, default='ResNet_CNN',choices=['Basic_CNN','ResNet_CNN'], help='Basic_CNN: a fully convolutional Neural network.'
                        'Resent_CNN: A pretrained Resnet18 for feature extraction + shallow CNN for classification. ')
    parser.add_argument('--best_model', type=str, default='ResNet-cnn-', help='The name of the best model')

    # Directories.
    parser.add_argument('--root', type=str, default='./data', help='path to dataset')
    parser.add_argument('--ids_file', type=str, default='train_valid_id.pkl', help='file containing samples id')
    parser.add_argument('--labels_file', type=str, default='train_labels.pkl', help='file containing labes dictionary')

    config = parser.parse_args()


    # For fast training.
    cudnn.benchmark = True

    time1=time.time()

    # Data loader.
    loader_train = get_loader(config, 'Shoes', 'train')
    loader_valid = get_loader(config, 'Shoes', 'valid')

    ## model loading
    if config.method=='Basic_CNN':
        cnn = CNN()
    else:
        res = resnet18(pretrained=True)
        res = res.cuda()
        res.eval()

        cnn = CNN2()

    cnn = cnn.cuda()


    ## Set loss function and optimizer
    loss_func = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(cnn.parameters(), lr=config.learning_rate)  # optimize all cnn parameters

    train_logs = {'train_accuracy': [], 'valid_accuracy': [],'train_c1_f1':[],'train_c2_f1':[],'train_c3_f1':[], 'train_loss': [], 'valid_loss': []
                  ,'valid_c1_f1':[],'valid_c2_f1':[],'valid_c3_f1':[]}
    val_max_f1 = 0

    ## model training
    for epoch in range(config.epochs):
        loss_value = 0
        accuracy = 0
        total_y =None
        total_pred=None
        for step, (X, y) in enumerate(loader_train):
            optimizer.zero_grad()
            X = Variable(X).cuda()
            y = Variable(y).cuda()
            if config.method=='Basic_CNN':
                outputs = cnn(X)
            else:
                outputs = res(X)
                outputs = cnn(outputs)

            loss = loss_func(outputs, y)
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            predictions = torch.argmax(outputs, dim=1)

            ## calculate accuracy
            acc = np.mean((y == predictions).cpu().numpy())
            accuracy += acc

            if total_y==None:
                total_y=y
            else:
                total_y=torch.cat([total_y,y],dim=0)

            if total_pred==None:
                total_pred=predictions
            else:
                total_pred=torch.cat([total_pred,predictions],dim=0)

        ## calculate precision, recall and f1
        p_r_f=compute_precision_recall_and_F1(total_y,total_pred)

        print("Epoch:", epoch)
        print('Train_loss:',loss_value/ (step + 1),'   Train_accuracy:', accuracy/ (step + 1))
        print('Train_f1:',p_r_f[0,2], p_r_f[1,2] ,p_r_f[2,2])

        train_logs['train_accuracy'].append(accuracy / (step + 1))
        train_logs['train_loss'].append(loss_value / (step + 1))
        train_logs['train_c1_f1'].append(p_r_f[0,2])
        train_logs['train_c2_f1'].append(p_r_f[1,2] )
        train_logs['train_c3_f1'].append(p_r_f[2,2] )

        ## validation loss and accuracy
        val_p_r_f = np.zeros((3, 3))
        valid_loss = 0
        valid_accuracy = 0
        for step,(X, y) in enumerate(loader_valid):
            X = X.cuda()
            y = y.cuda()

            if config.method=='Basic_CNN':
                outputs = cnn.forward(X)
            else:
                outputs = res.forward(X)
                outputs = cnn(outputs)

            loss = loss_func(outputs, y)
            valid_loss += loss.item()

            predictions = torch.argmax(outputs, dim=1)
            acc = np.mean((y == predictions).cpu().numpy())
            valid_accuracy += acc

            ## calculate p_r_f1
            val_p_r_f += compute_precision_recall_and_F1(y, predictions)

        # valid_loss, valid_accuracy = compute_loss_and_accuracy(cnn, loader_valid)
        print('Valid_loss:', valid_loss , '   Valid_accuracy', valid_accuracy)
        print('Valid_f1:', val_p_r_f[0, 2] , val_p_r_f[1, 2] , val_p_r_f[2, 2])

        train_logs['valid_accuracy'].append(valid_accuracy/ (step + 1))
        train_logs['valid_loss'].append(valid_loss/ (step + 1))
        train_logs['valid_c1_f1'].append(val_p_r_f[0, 2] )
        train_logs['valid_c2_f1'].append(val_p_r_f[1, 2] )
        train_logs['valid_c3_f1'].append(val_p_r_f[2, 2] )

        ## Save the best model
        v_f1=val_p_r_f[:,2].mean()
        if v_f1>=val_max_f1:
            val_max_f1=v_f1
            ## save model
            input_names = ["actual_input_1"]
            output_names = ["output1"]

            if config.method=='Basic_CNN':
                dummy_input = torch.randn(1, 3, 240, 240, device='cuda')
            else:
                dummy_input = torch.randn(1, 512, 8, 8, device='cuda')

            torch.onnx.export(cnn, dummy_input, "./checkpoints/"+config.best_model+str(epoch)+".onnx", verbose=False,
                              input_names=input_names, output_names=output_names)
            print("Save model_...")

        ## learning rate decay
        if (epoch + 1) % config.decay_every == 0:
            update_learning_rate(config, optimizer, config.learning_rate, config.weight_decay)


    time2=time.time()

    print('Time for model training:',time2-time1)
    with open('./results/train_logs_'+config.best_model+'.pkl', 'wb') as f:
        pickle.dump(train_logs, f)
        f.close()


if __name__ == '__main__':
    main()

