# -*- coding: utf-8 -*-
"""
Created on Wed May 16 20:45:43 2018

@author: tph
"""

import numpy as np

class SVM(object):
    def __init__(self):
        #W为加上偏置的权重（D,num_class)
        self.W=None

    def svm_loss_naive(self,x,y,reg):
        """
        功能：非矢量化版本的损失函数
        
        输入：
        -x：(numpy array)样本数据（N,D)
        -y：(numpy array)标签（N，）
        -reg：(float)正则化强度
        
        输出：
        (float)损失函数值loss
        (numpy array)权重梯度dW
        """
        num_train=x.shape[0]
        num_class=self.W.shape[1]

        #初始化
        loss=0.0
        dW=np.zeros(self.W.shape)

        for i in range(num_train):
            scores=x[i].dot(self.W)
            #计算边界,delta=1
            margin=scores-scores[y[i]]+1
            #把正确类别的归0
            margin[y[i]]=0

            for j in range(num_class):
                #max操作
                if j==y[i]:
                    continue
                if margin[j]>0:
                    loss+=margin[j]
                    dW[:,y[i]]+=-x[i]
                    dW[:,j]+=x[i]

        #要除以N
        loss/=num_train
        dW/=num_train
        #加上正则项
        loss+=0.5*reg*np.sum(self.W*self.W)
        dW+=reg*self.W

        return loss,dW


    def svm_loss_vectorized(self, x, y, reg):
        loss=0.0
        dW=np.zeros(self.W.shape)
    
        num_train=x.shape[0]
        scores=x.dot(self.W)
        margin=scores-scores[np.arange(num_train),y].reshape(num_train,1)+1
        margin[np.arange(num_train),y]=0.0
        margin=(margin>0)*margin
        loss+=margin.sum()/num_train

        loss+=0.5*reg*np.sum(self.W*self.W)

        margin=(margin>0)*1
        row_sum=np.sum(margin,axis=1)
        margin[np.arange(num_train),y]=-row_sum
        dW=x.T.dot(margin)/num_train+reg*self.W
    
        return loss,dW
    
    
    def train(self,x,y,reg=1e-5,learning_rate=1e-3,num_iters=100,batch_size=200,verbose=False):
        num_train,dim=x.shape
        num_class=np.max(y)+1
        if self.W is None:
            self.W=0.005*np.random.randn(dim,num_class)
        
        batch_x=None
        batch_y=None
        history_loss=[]
        for i in range(num_iters):
            mask=np.random.choice(num_train,batch_size,replace=False)
            batch_x=x[mask]
            batch_y=y[mask]
            loss,grad=self.svm_loss_vectorized(batch_x,batch_y,reg)
            self.W+=-learning_rate*grad
            history_loss.append(loss)
            if verbose==True and i%100==0:
                print("iteratons:%d/%d,loss:%f"%(i,num_iters,loss))
            
        return history_loss
        
    def predict(self,x):
        y_pre=np.zeros(x.shape[0])
        scores=x.dot(self.W)
        y_pre=np.argmax(scores,axis=1)
        
        return y_pre


























