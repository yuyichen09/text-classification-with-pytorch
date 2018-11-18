# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:05:29 2018

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


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def is_number(uchar):
    """判断一个unicode是否是数字"""
    if uchar >= u'\u0030' and uchar <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(uchar):
    """判断一个unicode是否是英文字母"""
    if (uchar >= u'\u0041' and uchar <= u'\u005a') or (uchar >= u'\u0061' and uchar <= u'\u007a'):
        return True
    else:
        return False


def is_legal(uchar):
    """判断是否非汉字，数字和英文字符"""
    if not (is_chinese(uchar) or is_number(uchar) or is_alphabet(uchar)):
        return False
    else:
        return True


def extract_chinese(line):
    res = ""
    for word in line:
        if is_legal(word):
            res = res + word
    return res;


def words2line(words):
    line = ""
    for word in words:
        line = line + " " + word
    return line


class deal_with_data():
    def __init__(self, file_path, stop_path, sentence_len, rare_len, rare_word, transform=None):
        self.file_path = file_path
        self.stop_path = stop_path
        self.sentence_len = sentence_len
        self.transform = transform
        self.rare_word = []
        self.rare_len = rare_len
#        self.num_lines_start = num_lines_start
#        self.num_lines_end = num_lines_end

    # jieba.load_userdict('userdict.txt')  
    # 创建停用词list  
    #    def stopwordslist():
    #        stopwords = [line.strip() for line in open('./stop_words.txt', 'r', encoding='utf-8').readlines()]
    #        return stopwords

    # 对句子进行分词  
    def word_cut(self):
        print('deal with data')
        texts = []
        labels = []
        file = open(self.file_path, 'r', encoding='utf-8')
        lines = file.readlines()
        stopwords = [line.strip() for line in open(self.stop_path, 'r', encoding='utf-8').readlines()]  # 这里加载停用词的路径
        for line in lines:
            line, label = line.split('\t')
            sentence_seged = jieba.lcut(line, cut_all=False, HMM=True)
            outstr = []
            for word in sentence_seged:
                if word not in stopwords:
                    if word != '\t':
                        outstr.append(word)
            texts.append(outstr)
            if int(float(label))<=3:
                label = 0
            else:
                label = 1
            labels.append(label)
        
        labels_all = np.array([np.array([int(label)]) for label in labels], dtype=np.int32)
        
        return texts, labels_all
        

    ###将词转换为id，并计算词出现的个数
    def word_to_id(self):
        word2index = {"<PAD>": 0, "<UNK>": 1}
        index2word = {0: "<PAD>", 1: "<UNK>"}
        n_words = 2
        data, labels = self.word_cut()
        word_dict = {}
        for sentence in data:
            for word in sentence:
                if word in word_dict.keys():
                    word_dict[word] += 1
                else:
                    word_dict[word] = 1
        
        for sentence in data:
            for word in sentence:
                if word_dict[word] <= self.rare_len:
                    self.rare_word.append(word)
                elif word not in word2index:
                    word2index[word] = n_words
                    index2word[n_words] = word
                    n_words += 1
                    # Transform to idx
        transform_data = np.array([[word2index[word]
                                    if word not in self.rare_word
                                    else word2index["<UNK>"] for word in sentence]
                                   for sentence in data])

        ###规定好句子长度，不足补零
        temp_list = []
        for sentence in transform_data:
            if len(sentence) > self.sentence_len:
                # truncate sentence if sentence length is longer than `sentence_len`
                temp_list.append(np.array(sentence[:self.sentence_len]))
            else:
                # pad sentence  with '<PAD>' token if sentence length is shorter than `sentence_len`
                sent_array = np.lib.pad(np.array(sentence),
                                        (0, self.sentence_len - len(sentence)),
                                        "constant",
                                        constant_values=(0, 0))
                temp_list.append(sent_array)
        transform_data = np.array(temp_list, dtype=np.int32)

        return transform_data, labels, self.rare_word, word2index, index2word
        
        
class data_set(Dataset):
    def __init__(self,data,labels,transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sentence = self.data[idx]
        label =self.labels[idx]
        #print(label)
        sample = {"sentence": sentence, "label": label}

        if self.transform:
            sample = {"sentence": self.transform(sample["sentence"]),
                      "label": self.transform(sample["label"])}

        return sample

    def vocab_length(self):
        return self.n_words


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        return torch.from_numpy(data).type(torch.LongTensor)

    # a = deal_with_data(file_path = './comment_data_sample.txt',stop_path = './stop_words.txt',sentence_len = 50,rare_len = 0,rare_word = [],transform =ToTensor())
