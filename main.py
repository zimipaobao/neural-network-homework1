import math
import random

import pandas as pd
from mlxtend.data import loadlocal_mnist
import matplotlib.pyplot as plt
import numpy as np
import copy
import xlwt

init_learning_rate=0.001
final_learning_rate=0.0005
hidden_size=100#the number of the hidden nodes is 20
output_size=10#the number of the output nodes is 10, indicating the 10 classes
p=1#for l2 regularization
itera=50
stop_criterion=100
bs=100
lr1=init_learning_rate
lr2=final_learning_rate


###########################################################################################
# load the data set
train_image, train_label = loadlocal_mnist(
        images_path='train-images.idx3-ubyte',
        labels_path='train-labels.idx1-ubyte')
test_image, test_label = loadlocal_mnist(
        images_path='t10k-images.idx3-ubyte',
        labels_path='t10k-labels.idx1-ubyte')
print('Dimensions: %s x %s' % (train_image.shape[0], train_image.shape[1]))
temp=train_image[0].reshape(28,28)
plt.imshow(temp)# test if the data set is correctly loaded
ave=np.mean(train_image)
sd=np.std(train_image)
train_image=(train_image-ave)/sd
test_image=(test_image-ave)/sd
#########################################################################################
# some of the functions needed.Including loss function, gradient function, relu function,

def gradient_g2(target,prob):
        #input:the prob vector,target
        #output:the gradient vector corresponding to score vector,which is w2*h1
        #reference:https://zhuanlan.zhihu.com/p/35709485
        temp=copy.deepcopy(prob)
        temp[0,target]=prob[0,target]-1
        return temp

def gradient_w2(g2,w2,h1):
        #input:g2 is the result from gradient_g2 function.w2 is weight2.h1 is the result after relu
        #output:gradient of weight2
        temp=np.dot(h1.T,g2)+p*w2
        return temp

def gradient_w1(g2,w2,data,w1):
        #input:g2 is the result from gradient_g2 function.w2 is weight2.data is the 784*1 input data.w1 is weight1.h1 is h1 after relu
        #output:gradient of weight1
        temp=np.dot(g2,w2.T)
        temp=np.maximum(0,temp)
        temp=np.dot(data.T,temp)
        temp=temp+p*w1
        return temp

def activate_function(input):
        #input of the function is a ndarray.
        #output is also a ndarray,same size with input.
        #this is relu function.
        res=np.maximum(0,input)
        return res

def softmax(score):
        temp=copy.deepcopy(score)
        total=np.sum(np.exp(score))
        temp=temp/total
        return temp

def loss(prob,target):
        return - np.log(prob[0,target])

#########################################################################################
# define the neural network,use the network to get a predicted value.
def nn(data,weight1,weight2):
        #data is a 784*1 vector
        #weight1 is a 100*784 weight matrix
        #weight2 is a 20*10 weight matrix
        #output is the 10*1 vector
        h1=np.dot(data,weight1)
        h1=activate_function(h1)
        score=np.dot(h1,weight2)
        prob=softmax(score)
        predicted=np.argmax(prob)
        return predicted

#########################################################################################
#define the train process

# the initial parameters
init_weight1=np.random.normal(0.0, 1, (784,hidden_size))#the initial weight of the hidden nodes is from a N(0,1) distribution.
init_weight2=np.random.normal(0.0, 1, (hidden_size,output_size))#the initial weight of the output nodes is from a N(0,1) distribution.

image=train_image
label=train_label
#def train(init_weight1,init_weight2,image=train_image,label=train_label,lr1=init_learning_rate,lr2=final_learning_rate,itera=25,bs=50,stop_criterion=100):
record = xlwt.Workbook(encoding='utf-8', style_compression=0)
sheet = record.add_sheet('record1', cell_overwrite_ok=True)
col = ('iteration', 'accuracy','loss')
for i in range(0, 2):
        sheet.write(0, i, col[i])
n=range(60000)
weight1=copy.deepcopy(init_weight1)
weight2=copy.deepcopy(init_weight2)
for i in range(itera):
        tloss = 0
        for m in range(stop_criterion):
                index=random.sample(n,bs)
                batch_image=image[index,]
                batch_label=label[index,]
                grad1=np.zeros(np.shape(weight1))
                grad2=np.zeros(np.shape(weight2))
                for j in range(np.shape(batch_label)[0]):
                        sample_image=batch_image[j,].reshape(1,784)
                        sample_label=batch_label[j,]
                        h1 = np.dot(sample_image,weight1)
                        ha1 = activate_function(h1)
                        score = np.dot(ha1,weight2)
                        prob=softmax(score)
                        g2=gradient_g2(sample_label,prob)
                        gw2=gradient_w2(g2,weight2,ha1)
                        gw1=gradient_w1(g2=g2,w2=weight2,data=sample_image,w1=weight1)
                        grad1 += (1/bs) * gw1
                        grad2 += (1/bs) * gw2
                        tloss += loss(prob,sample_label)
                weight1=weight1-((1-i/itera)*lr1+(i/itera)*lr2)*grad1
                weight2=weight2-((1-i/itera)*lr1+(i/itera)*lr2)*grad2
                if np.linalg.norm(grad1)<=0.1 and np.linalg.norm(grad2<=0.1):
                        tloss=tloss/(bs*(m+1))
                        break
        pred=np.zeros(np.shape(test_label))
        for h in range(np.shape(test_label)[0]):
                data=test_image[h,].reshape(1,784)
                res=nn(data,weight1,weight2)
                pred[h,]=res
        accuracy=np.sum(pred==test_label)/np.shape(test_label)[0]
        tloss += p*(np.linalg.norm(weight1)+np.linalg.norm(weight2))
        sheet.write(i+1,0,i)
        sheet.write(i+1,1,accuracy)
        sheet.write(i+1,2,tloss)
        print("iteration:",i,'   ','accuracy:',accuracy,'   ','loss:',tloss)
savepath = 'D:/learning/neural_network/neural_network_hw1_classifier/result.xls'
record.save(savepath)
#return weight1,weight2

writer1=pd.ExcelWriter('weight1.xlsx')
w1pd=pd.DataFrame(weight1)
w1pd.to_excel(writer1)
writer1.save()
writer1.close()

writer2=pd.ExcelWriter('weight2.xlsx')
w2pd=pd.DataFrame(weight2)
w2pd.to_excel(writer2)
writer2.save()
writer2.close()
#res=train(init_weight1,init_weight2,image=train_image,label=train_label,lr1=init_learning_rate,lr2=final_learning_rate)

plt.imshow(weight1)
plt.imshow(weight2)