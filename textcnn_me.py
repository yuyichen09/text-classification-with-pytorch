# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:44:32 2018

@author: cmei
"""

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import numpy as np;
import os
import jieba
import gensim.models.word2vec as w2v
from sklearn.model_selection import train_test_split
import pandas as pd
import collections
from torch.utils.data import Dataset

from tqdm import tqdm

import math


# def getw2v():
#    model_file_name = 'new_model_big.txt'
#    # 模型训练，生成词向量
#    '''
#    sentences = w2v.LineSentence('trainword.txt')
#    model = w2v.Word2Vec(sentences, size=20, window=5, min_count=5, workers=4)
#    model.save(model_file_name)
#    '''
#    model = w2v.Word2Vec.load(model_file_name)
#    return model;


class TextCNN(nn.Module):
    def __init__(self, embedding, sentence_len, filter_size, latent_size, n_class):
        super(TextCNN, self).__init__()
        self.embed = embedding
        self.sentence_len = sentence_len
        self.filter_size = filter_size
        self.latent_size = latent_size
        self.n_class = n_class

        # 想要con2d卷积出来的图片尺寸没有变化, padding=(kernel_size-1)/2
        # sentence_len = 100
        # filter_size = 16

        self.conv1_1 = nn.Sequential(
            nn.Conv2d(1, self.filter_size, (5, self.embed.weight.size()[1]), stride=1, padding=(2,0)),
            nn.BatchNorm2d(self.filter_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))  # (16,50,50)
        )

        self.conv1_2 = nn.Sequential(
            nn.Conv2d(1, self.filter_size, (3, self.embed.weight.size()[1]), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.filter_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))  # (16,50,50)
        )

        self.conv1_3 = nn.Sequential(
            nn.Conv2d(1, self.filter_size, (2, self.embed.weight.size()[1]), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.filter_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))  # (16,50,50)
        )

        self.conv2_1 = nn.Sequential(
            nn.Conv2d(self.filter_size, self.filter_size * 2, (5, 1), stride=1, padding=(2,0)),
            nn.BatchNorm2d(self.filter_size*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))  # (32,25,25)
        )

        self.conv2_2 = nn.Sequential(
            nn.Conv2d(self.filter_size, self.filter_size * 2, (3, 1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.filter_size*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))  # (32,25,25)
        )

        self.conv2_3 = nn.Sequential(
            nn.Conv2d(self.filter_size, self.filter_size * 2, (2, 1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.filter_size*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,1))  # (32,25,25)
        )

        self.conv3_1 = nn.Sequential(
            nn.Conv2d(self.filter_size * 2, self.latent_size, (5, 1), stride=1, padding=(2,0)),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2) # (32,25,25)
        )

        self.conv3_2 = nn.Sequential(
            nn.Conv2d(self.filter_size * 2, self.latent_size, (3, 1), stride=1, padding=(1,0)),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2) # (32,25,25)
        )

        self.conv3_3 = nn.Sequential(
            nn.Conv2d(self.filter_size * 2, self.latent_size, (1, 1), stride=1, padding=(0,0)),
            nn.BatchNorm2d(self.latent_size),
            nn.ReLU()
            # nn.MaxPool2d(kernel_size=2) # (32,25,25)
        )

        self.out = nn.Linear(self.latent_size*3 * (self.sentence_len / 4) , self.n_class)

    def forward(self, x):
        
        x = self.embed(x)
       

        x = x.view(x.size(0), 1, x.size()[1], x.size()[2])
        
        # print(x.size())
        x1 = self.conv1_1(x)
        x1 = self.conv2_1(x1)
        x1 = self.conv3_1(x1)
        
        

        x2 = self.conv1_2(x)
        x2 = self.conv2_2(x2)
        x2 = self.conv3_2(x2)
        

        x3 = self.conv1_3(x)
        x3 = self.conv2_3(x3)
        x3 = self.conv3_3(x3)
        

        in_put = torch.cat((x1, x2), 1)
        in_put = torch.cat((in_put, x3), 1)
        
        print(in_put.size())

        in_put = in_put.view(in_put.size(0), -1)  # 将（batch，outchanel,w,h）展平为（batch，outchanel*w*h）
        # print(x.size())

        output = self.out(in_put)
        return output
