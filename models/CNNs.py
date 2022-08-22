# -*- coding:utf-8 -*-
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.cnn=nn.Sequential(
            nn.Conv2d(3,32,3,padding=1),        ##(32,240,240)
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),                 ##(32,120,120)
            nn.ReLU(),

            nn.Conv2d(32, 64, 3, padding=1),    ##(64,120,120)
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2, 2),                 ##(64,60,60)
            nn.ReLU(),


            nn.Conv2d(64, 128, 3, padding=1),   ##(128,60,60)
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2, 2),                 ##(128,30,30)
            nn.ReLU(),

            nn.Conv2d(128, 256, 3, padding=1),  ##(256,30,30)
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2, 2),                 ##(256,15,15)
            nn.ReLU(),

            nn.Conv2d(256, 512, 3, padding=1),  ##(512,15,15)
            nn.BatchNorm2d(512),
            nn.MaxPool2d(2, 2),                 ##(512,7,7)
            nn.ReLU(),

            nn.Conv2d(512, 1024, 3, padding=1),  ##(1024,7,7)
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),                 ##(1024,3,3)
            nn.ReLU(),

            nn.Conv2d(1024,1024, 3, padding=1),  ##(1024,3,3)
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),                 ##(1024,1,1)
            nn.ReLU(),
        )

        self.result=nn.Sequential(
            nn.Conv2d(1024, 100, 1),            ##(100,1,1)
            nn.Conv2d(100, 3, 1),               ##(3,1,1)
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        features=self.cnn(x)
        out=self.result(features)
        return out.view(-1,3)

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2,self).__init__()

        self.cnn=nn.Sequential(
            nn.Conv2d(512, 800, 3, padding=1),  ##(800,8,8)
            nn.BatchNorm2d(800),
            nn.MaxPool2d(2, 2),                 ##(800,4,4)
            nn.ReLU(),

            nn.Conv2d(800, 1000, 3, padding=1),  ##(800,4,4)
            nn.BatchNorm2d(1000),
            nn.MaxPool2d(2, 2),                 ##(800,2,2)
            nn.ReLU(),

            nn.Conv2d(1000,1024, 3, padding=1),  ##(1024,2,2)
            nn.BatchNorm2d(1024),
            nn.MaxPool2d(2, 2),                 ##(1024,1,1)
            nn.ReLU(),
        )

        self.result=nn.Sequential(
            nn.Conv2d(1024, 100, 1),            ##(100,1,1)
            nn.Conv2d(100, 3, 1),               ##(3,1,1)
            nn.Softmax(dim=1)
        )


    def forward(self, x):
        features=self.cnn(x)
        out=self.result(features)
        return out.view(-1,3)
