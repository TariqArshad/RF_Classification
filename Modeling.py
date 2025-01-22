import torch
import numpy as np
import pandas as pd
import random
import time
from sklearn import linear_model
import os
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
from reservoirpy.nodes import ScikitLearnNode
from reservoirpy.nodes import Reservoir, Ridge, Input, ESN, NVAR
from reservoirpy.observables import rmse, rsquare
import pickle
from tqdm import tqdm
import sys
import logging



class Conv1D_RF_Classifier(torch.nn.Module):
    def __init__(self, classes = 3, seq_len = 1024):
        super(Conv1D_RF_Classifier, self).__init__()
        self.loss_fn = torch.nn.CrossEntropyLoss();
        
        self.classes = classes;
        
        self.conv_0 = torch.nn.Conv1d(in_channels = 4, out_channels = 64, kernel_size = 9, stride = 1);
        self.batch_norm_0 = torch.nn.BatchNorm1d(64);
        self.max_pool_0 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.conv_1 = torch.nn.Conv1d(in_channels = 64, out_channels = 32, kernel_size = 9, stride = 1);
        self.batch_norm_1 = torch.nn.BatchNorm1d(32)
        self.max_pool_1 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.conv_2 = torch.nn.Conv1d(in_channels = 32 , out_channels = 16, kernel_size = 9, stride = 1);  
        self.batch_norm_2 = torch.nn.BatchNorm1d(16);
        self.max_pool_2 = torch.nn.MaxPool1d(kernel_size = 2 , stride = 2)
        
        self.linear_0 = torch.nn.Linear(1936, 256); #for 1024 seq len
        self.linear_1 = torch.nn.Linear(256,classes);
        self.Relu = torch.nn.ReLU()
        self.Conv_Dropout = torch.nn.Dropout(0.3);
        self.Linear_Dropout = torch.nn.Dropout(0.5);
         

    def forward(self, data_in, y_true = None):
        """
        data_in shape:(batch_sz,channel_sz,seq_len), where channel_sz = 4, seq_ken = 1024
        """
        logits = self.conv_0(data_in)
        logits = self.Relu(logits)
        logits = self.batch_norm_0(logits)
        logits = self.max_pool_0(logits)
        logits = self.Conv_Dropout(logits)
        
        logits = self.conv_1(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_1(logits)
        logits = self.max_pool_1(logits)
        logits = self.Conv_Dropout(logits)
        
        logits = self.conv_2(logits)
        logits = self.Relu(logits)
        logits = self.batch_norm_2(logits)
        logits = self.max_pool_2(logits)
        logits = self.Conv_Dropout(logits)
    
        logits = logits.flatten(1, 2)
        logits = self.linear_0(logits)
        logits = self.Relu(logits)
        logits = self.Linear_Dropout(logits)

        logits = self.linear_1(logits);

        if y_true == None: return logits;
        else:
            logits = logits.reshape((-1, logits.shape[1]));
            loss = self.loss_fn(logits, y_true);
            return logits, loss;
            

class RC_Seq_Classifier():
    def __init__(self, seq_sub_size = 16, savedir = "",reservoir = Reservoir(**{"units": 512, "lr":0.2, "sr":0.9, "rc_connectivity":0.5})  ,rc_params = {"units": 256, "lr":0.5, "sr":0.9}, nvar_params = None):
        self.reservoir = reservoir;
        #self.reservoir = Reservoir(**{"units": 512, "lr":0.2, "sr":0.9, "rc_connectivity":0.5}) >> Reservoir(**{"units": 512, "lr":0.2, "sr":0.9, "rc_connectivity":0.5});
        if nvar_params != None:
            self.nvar = NVAR(**nvar_params);
        self.seq_sub_size = seq_sub_size;
        if savedir[-1] != "/": savedir += "/";
        if os.path.exists(savedir) == False: os.mkdir(savedir);
        self.savedir = savedir;
        #variable is only relevent when saving and visulaizing last states
        self.train = True;

    def get_reservoir_laststate(self, samples, savestates = False):
        last_states = [];
        for seq in tqdm(samples):
            
            sub_seqs =np.stack(np.split(seq, len(seq)//self.seq_sub_size));
            
            last_state = self.reservoir.run(sub_seqs, reset = True);
            #last_state = torch.from_numpy(states);
            #last_state = torch.nn.functional.scaled_dot_product_attention(last_state, last_state, last_state);
            #last_state = last_state.numpy();
            
            last_states.append(last_state[-1]);
        last_states = np.stack(last_states);

        if savestates:
            if self.train: Savepath = self.savedir + "last_states_train.npy";
            else:  Savepath = self.savedir + "last_states_test.npy";

            with open(Savepath, "wb") as savefile:
                np.save(savefile, last_states);
        return last_states;


    def Visualize_reservoir_laststate2D(self, xsamples, ysamples, label_encoder = None, load_laststates = False):
        """
        Function visualizes the last states of signals using t-SNE
        :param samples:
        :return:
        """
        from sklearn.manifold import TSNE

        if load_laststates:
            if self.train: Loadpath = self.savedir + "last_states_train.npy";
            else: Loadpath = self.savedir + "last_states_test.npy";
            with open(Loadpath, "rb") as loadfile:
                last_states = np.load(loadfile);
        else: last_states = self.get_reservoir_laststate(xsamples);

        SNE_embedds = TSNE(n_components=2, learning_rate='auto', init = 'random').fit_transform(last_states)

        fig, ax = plt.subplots()

        if label_encoder != None:
            ysamples = label_encoder.inverse_transform(ysamples);
        else:
            ysamples = np.argmax(ysamples, axis=1);

        yclasses = np.unique(ysamples);

        for y_class in yclasses:
            class_index = (ysamples == y_class).flatten();
            embedds = SNE_embedds[class_index];

            ax.scatter(embedds[:, 0], embedds[:, 1], label = y_class);
        ax.legend();
        plt.show();


        return SNE_embedds;


    def train_Ridge(self, xtrain, ytrain, load_laststates = False, savemodel = False, ridge = 1e-5):
        self.Ridge = Ridge(output_dim=ytrain.shape[1], ridge=ridge);
        if load_laststates:
            Loadpath = self.savedir + "last_states_train.npy";
            with open(Loadpath, "rb") as loadfile:
                last_states = np.load(loadfile);
        else: last_states = self.get_reservoir_laststate(samples = xtrain);
        self.Ridge.fit(last_states, ytrain);

        if savemodel:
            model = {"reservoir":self.reservoir, "Ridge":self.Ridge};
            savepath = self.savedir + "RC_ridgemodel.npy";
            with open(savepath, "wb") as savefile:
                np.save(savefile, model);

        return None

    def eval_Ridge(self, xtest, ytest, load_laststates = False):
        if load_laststates:
            Loadpath = self.savedir + "last_states_test.npy";
            with open(Loadpath, "rb") as loadfile:
                last_states = np.load(loadfile);
        else:
            last_states = self.get_reservoir_laststate(samples=xtest);
        ypred = self.Ridge.run(last_states);

        Y_pred_class = np.array([np.argmax(y_p) for y_p in ypred]);
        Y_test_class = np.array([np.argmax(y_t) for y_t in ytest]);
        score = accuracy_score(Y_test_class, Y_pred_class)
        print("accuracy score:", score);

        cm = confusion_matrix(Y_test_class, Y_pred_class);
        ConfusionMatrixDisplay(cm).plot();
        plt.show()

        return None;
    def train_MultiLog(self, xtrain, ytrain, max_iter = 1000):
        from sklearn.linear_model import LogisticRegression
        ytrain = np.argmax(ytrain, axis = 1);
        last_states =  self.get_reservoir_laststate(samples=xtrain)

        self.MultiLog = LogisticRegression(max_iter=max_iter).fit(last_states, ytrain);
        return None;
    def eval_MultiLog(self, xtest, ytest):
        last_states = self.get_reservoir_laststate(samples=xtest)
        score = self.MultiLog.score(last_states, np.argmax(ytest, axis = 1));
        print(score);
        return None;


def main():
   
    return 0



if __name__ == '__main__':
    main();
