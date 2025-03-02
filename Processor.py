import os
import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import json
import torch
import random
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm


class POWDERRF_Processor():

    def __init__(self, datadir = "", sample_len = 1024, samples_per_sig = 10, max_samples = 1000,savedir = None, data_days = [1, 2], transmitters = ["bes", "browning", "honors", "meb"],protocols = ["5G", "4G", "WiFi"]):
        """
        Data processor for the POWDER RF fingerprint dataset
        datadir = dirctory holding powder
        sample_len = length of signal to be sampled
        loaddatadict = load previously processed dictionary
        Data_Day = The day the data was taken, where 0 or None=both, 1=day1, 2=day2
        Protocol = Protocol used in transmisson between 4G(4G), Wifi(802.11a), and 5G(5G NR) 
        """
        
        if datadir[-1] != "/": datadir += "/";
        if savedir!= None and savedir[-1] != "/": savedir += "/";
        self.savedir = savedir;
        self.datadir = datadir;
        self.sample_len = sample_len;
        self.sample_count = 0;
        self.max_samples = max_samples;
        self.samples_per_sig = samples_per_sig;
        self.data_days = data_days;
        self.transmitters = transmitters;
        self.protocols = protocols;
        self.ytrain = None;
        self.datadict = None;

    def __call__(self,train_test_split = .8, signal_type = "All",task = "transmitter"  , normalize = True ,loaddatadict = False):
        """
        train_test_split = percetnage of overall dataset to be training
        signal_type = if signal should be All,real, imag, Mag, or Phase
        task = if task should be to classify transmitter or protocol
        """
        if loaddatadict:
            loadpath = self.savedir + "POWDERRF.json"
            with open(loadpath, "r") as loadfile:
                self.datadict = json.load(loadfile); 
        else:    
            self.datadict = self.parse_labels();
        
        x_data = [];
        y_data = [];

        for file_key in tqdm(self.datadict.keys()):
            if task == "protocol":y_samples = self.datadict[file_key]["protocol"]
            elif task == "transmitter": y_samples = self.datadict[file_key]["transmitter"].split("__")[0];  
            elif task == "joint": y_samples = self.datadict[file_key]["transmitter"].split("__")[0] +"_"+ self.datadict[file_key]["protocol"];
                
            I_x_samples = self.datadict[file_key]["I_signal"];
            Q_x_samples = self.datadict[file_key]["Q_signal"];
            
            if signal_type == "Complex":
                x_samples = np.array(I_x_samples) +np.array( Q_x_samples)*1.0j;
            elif signal_type == "Real":
                x_samples = I_x_samples;
            elif signal_type == "Imag":
                x_samples = Q_x_samples;
            elif signal_type == "Mag":
                x_samples = np.array(I_x_samples) + np.array(Q_x_samples)* 1.0j;
                x_samples = np.abs(x_samples).tolist();
            elif signal_type == "Phase":
                x_samples = np.arctan2(np.array(Q_x_samples)/np.array(I_x_samples));
            elif signal_type == "All":
                Mag = np.abs(np.array(I_x_samples) + np.array(Q_x_samples)* 1.0j)
                Mag = np.expand_dims(Mag, axis = 1);
                Phase =  np.arctan2(np.array(Q_x_samples), np.array(I_x_samples))
                Phase = np.expand_dims(Phase, axis = 1);
                I_ = np.array(I_x_samples);
                I_ = np.expand_dims(I_, axis = 1);
                Q_ = np.array( Q_x_samples)
                Q_ = np.expand_dims(Q_, axis = 1)
                x_samples = np.concatenate([I_, Q_, Mag, Phase] ,axis = 1);

                
            # normalization, should make into own function
            if normalize:
                x_data_mean = np.mean(x_samples, axis = 0);
                x_data_std = np.std(x_samples, axis = 0);
                x_samples = (x_samples - x_data_mean)/x_data_std;
            
            y_data.extend([y_samples]*len(x_samples));
            x_data.extend(x_samples);

        y_data = np.array(y_data);
        x_data = np.array(x_data);

        classes = np.unique(y_data);
        label_encoder = OneHotEncoder();
        label_encoder.fit(classes.reshape((-1, 1)));
        self.label_encoder = label_encoder;
        y_data = label_encoder.transform(y_data.reshape(-1, 1)).toarray().astype(int);

        shuffle_index = list(range(0, x_data.shape[0]));

        random.shuffle(shuffle_index);

        y_data = y_data[shuffle_index];
        x_data = x_data[shuffle_index];

        train_split_index = int(np.floor(x_data.shape[0]*train_test_split));

        x_train = x_data[:train_split_index];
        x_test = x_data[train_split_index:];
        y_train = y_data[:train_split_index];
        y_test = y_data[train_split_index:];

        self.x_train = x_train;
        self.x_test = x_test;
        self.y_train = y_train;
        self.y_test = y_test;

        return x_train, y_train, x_test, y_test;
        
    def parse_labels(self, save = False):
        datadict = {}
        files = [];
        for file in os.listdir(self.datadir):
            #so we do not iterate over the .bin files
            if ".json" not in file: continue;
            #filter files by protocol and day
            filename_split = file.split(sep = "_");
            transmitter = filename_split[3];
            protocol = filename_split[0];
            day = int(filename_split[2]);
            if transmitter not in self.transmitters: continue;
            if protocol not in self.protocols: continue;
            if day not in self.data_days: continue;
            files.append(file);
        random.shuffle(files);
            
        N_files = self.max_samples//self.samples_per_sig;
        
        while N_files > len(files):
            copy_files = files;
            random.shuffle(copy_files);
            files = files + copy_files;
        if len(files) > N_files:
            files = files[:N_files];
            
        
        #in case the number of samples exceeds the number of available files go back and reshuffle files and sample again
        for file in tqdm(files):
            dict_key = file.split(sep=".")[0];
            binfilepath = self.datadir + dict_key + ".bin";
            jsonfilepath = self.datadir + dict_key + ".json"
            
            with open(jsonfilepath, "rb") as jsonfile:
                sig_metadata = json.load(jsonfile);
              
            if dict_key not in datadict.keys():
                datadict[dict_key] = {"I_signal":[],"Q_signal":[], "transmitter":None, "protocol":None};
                datadict[dict_key]["transmitter"] = sig_metadata["annotations"]["transmitter"]["core:location"] + "__" + sig_metadata["annotations"]["transmitter"]["core:radio"].replace(" ", "_") + "__" + sig_metadata["annotations"]["transmitter"]["core:antenna"].replace("-", "_");
                datadict[dict_key]["protocol"] = sig_metadata["annotations"]["core:protocol"];
                
            sig_data = np.fromfile(file = binfilepath, dtype=np.complex64);
            sig_data_list = self.parition_signal(sig_data);
            datadict[dict_key]["I_signal"].extend(np.real(sig_data_list).astype(np.float32).tolist());
            datadict[dict_key]["Q_signal"].extend(np.imag(sig_data_list).astype(np.float32).tolist());
        
        if save == True:
            if self.savedir != None:
                savepath = self.savedir + "POWDERRF.json"
                with open(savepath, "w") as savefile:
                    json.dump(datadict, savefile);
                
        return datadict;
        
    def parition_signal(self, signal):
        signal_list = [];
        
        for count in range(self.samples_per_sig):
            i = random.randint(0, len(signal) - 1 - self.sample_len);
            signal_sample = signal[i: i + self.sample_len];
            signal_list.append(signal_sample.tolist());
        
        random.shuffle(signal_list);
        return signal_list;

    def get_labeldist(self):
        if self.label_encoder == None:
            print("No Dataset Found!!!");
            return None;

        train_lables = self.label_encoder.inverse_transform(self.y_train);
        test_lables = self.label_encoder.inverse_transform(self.y_test);

        train_classes, train_counts = np.unique(train_lables, return_counts=True);
        test_classes, test_counts = np.unique(test_lables, return_counts=True);

        fig, axs = plt.subplots(1, 1);
        axs.bar(train_classes, train_counts);
        axs.bar(test_classes, test_counts);
        plt.show();

        return None;



def main():

    return 0;
if __name__ == "__main__":
    main();