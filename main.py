# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 17:23:53 2018

@author: cmei
"""



import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import util
import textcnn_me
from datasets import deal_with_data, ToTensor,data_set
#from train import train_classification
from torch.autograd import Variable




def main(file_train_path,stop_path,rare_len,epochs,embed_dim,batch_size,shuffle,sentence_len,filter_size,latent_size,n_class,LR,save_interval,save_dir,use_cuda):
    data_all,labels,rare_word, word2index, index2word = deal_with_data(file_path = file_train_path,stop_path = stop_path,sentence_len = sentence_len,rare_len =rare_len,rare_word = []).word_to_id()   
    
    ###一共有多少个词
    counts_words_len = len(word2index)
    
    ###一共多少个样本
    sample_len = len(labels)
    
    train_data = data_set(data_all[0:int(0.7*sample_len)],labels[0:int(0.7*sample_len)],transform =ToTensor())
    test_data = data_set(data_all[int(0.7*sample_len):sample_len],labels[int(0.7*sample_len):sample_len],transform =ToTensor())
    
    
    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)
    test_loader = DataLoader(test_data, batch_size=len(test_data)/100, shuffle=shuffle)
               
    # 做embedding                       
    embedding = nn.Embedding(counts_words_len, embed_dim, max_norm=1.0, norm_type=2.0)
    
    #构建textCNN模型
    cnn = textcnn_me.TextCNN(embedding = embedding, sentence_len = sentence_len, filter_size = filter_size, latent_size = latent_size,n_class = n_class)
    
    cnn_opt = torch.optim.Adam(cnn.parameters(), lr=LR)
    
    #损失函数
    loss_function = nn.CrossEntropyLoss()
    
    steps = 0
    for epoch in range(1, epochs+1):
        
        
        print("=======Epoch========")
        print(epoch)
        for batch in data_loader:
            feature, target = Variable(batch["sentence"]), Variable(batch["label"])
            if use_cuda:
                cnn.cuda()
                feature, target = feature.cuda(), target.cuda()
            
            cnn_opt.zero_grad()
            
            output = cnn(feature)
            
            #print(output)
            #print(target.view(target.size()[0]))
            loss = loss_function(output, target.view(target.size()[0]))
            loss.backward()
            cnn_opt.step()
           
    
            steps += 1
            print("Epoch: {}".format(epoch))
            print("Steps: {}".format(steps))
            print("Loss: {}".format(loss.data[0]))
            
    
            if epoch % save_interval == 0:
                util.save_models(cnn, save_dir, "cnn", epoch)
        
            for batch in test_loader:
                test_feature,test_target = Variable(batch["sentence"]),Variable(batch["label"])
                test_output =cnn(test_feature)    
                pred_y = torch.max(test_output,1)[1]
                acc = (test_target.view(test_target.size()[0]) == pred_y)
                acc = acc.numpy().sum()
                accuracy = acc / (test_target.size(0))
                print(len(pred_y))
                print('test_acc:{}'.format(accuracy))
        
        
    
    print("Finish!!!")
    return cnn
    


epochs = 1
embed_dim = 20
batch_size = 1280
shuffle = True
sentence_len = 160
rare_len = 10
filter_size = 16
latent_size = 32
n_class = 2
LR = 0.001
save_interval = 5
save_dir = './model/'
use_cuda = False
file_train_path = './comment_data_train.txt'
stop_path = './stop_words.txt'





if __name__ == '__main__':
    cnn = main(file_train_path,stop_path,rare_len,epochs,embed_dim,batch_size,shuffle,sentence_len,filter_size,latent_size,n_class,LR,save_interval,save_dir,use_cuda)





























