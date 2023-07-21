import matplotlib.pyplot as plt
import os
import numpy as np
import librosa
from tqdm import tqdm
import math
# import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
# from MyTool.featureExtraction import featureExtracting
from utils import featureExtracting
import utils
from utils import MyDataset
from spafe.utils.preprocessing import SlidingWindow

def read_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the contents of the file
        lines = file.readlines()
    
    # Remove newline characters and whitespace from each line
    lines = [line.strip() for line in lines]
    
    return lines

config_path = 'config.json'
hps = utils.get_hparams_from_file(config_path)
data_paths = read_file(hps.data.raw_data)


for data_path in tqdm(data_paths):
    type_list = os.listdir(data_path)

    datasetName = data_path.split('/')[-1]
    datasetPath = os.path.join(hps.data.downsample_data,datasetName)
    if not os.path.exists(datasetPath):
        os.mkdir(datasetPath)
    count_train=0
    count_test=0
    for t in type_list:
        if 'train' in t:
            save_path = os.path.join(hps.data.downsample_data,datasetName,'train_data_mfcc_lfcc.npz')
        else:
            save_path = os.path.join(hps.data.downsample_data,datasetName,'test_data_mfcc_lfcc.npz')
        data_list = os.listdir(os.path.join(data_path,t))
        X_data = []
        y_data = []

        for i in tqdm(range(len(data_list))):
            try:
                wav,sr = librosa.load(os.path.join(data_path,t,data_list[i]),sr=8000)
            except:
                continue
            
            if 'original' in data_list[i]:
                y_data.append(0)
            else:
                y_data.append(1)

#==================================== 特征提取=====================================
            #提取mfcc特征,将原语音和纯语音mfcc相减获取本底特征
            data_feature_mfcc = featureExtracting(wav,sr,hps.data.feature_type[0],
                        pre_emph=1,
                        pre_emph_coeff=0.97,
                        num_ceps= hps.data.n_mel_channels[0],
                        window=SlidingWindow(hps.data.win_length,hps.data.hop_length , hps.data.window_type),
                        nfilts=hps.data.nfilts,
                        nfft=hps.data.nfft,
                        low_freq=hps.data.mel_fmin,
                        normalize="mvn")
            data_feature_lfcc = featureExtracting(wav,sr,hps.data.feature_type[1],
                        pre_emph=1,
                    pre_emph_coeff=0.97,
                    num_ceps= hps.data.n_mel_channels[1],
                    window=SlidingWindow(hps.data.win_length,hps.data.hop_length , hps.data.window_type),
                    nfilts=hps.data.nfilts,
                    nfft=hps.data.nfft,
                    low_freq=hps.data.mel_fmin,
                    normalize="mvn")

            #concat mfcc and lfcc
            data_feature = np.vstack((data_feature_mfcc,data_feature_lfcc))
            if data_feature.shape[1] < 300:
                arr_0 = np.zeros((32,300-data_feature.shape[1]))
                data_feature = np.concatenate((data_feature,arr_0),axis=1)

            X_data.append(data_feature)

#==================================================================================
            if len(X_data) >= 2000:#每2000个保存一个npz文件
                X_data = np.array(X_data)
                y_data = np.array(y_data)
                
                if 'train' in save_path:
                    save_name = save_path.split('.')[0]+str(count_train)+'.npz'
                    while os.path.exists(save_name):
                        count_train+=1
                        save_name = save_path.split('.')[0]+str(count_train)+'.npz'
                    print('save train file:',save_name)
                    count_train+=1
                else:
                    save_name = save_path.split('.')[0]+str(count_test)+'.npz'
                    while os.path.exists(save_name):
                        count_test+=1
                        save_name = save_path.split('.')[0]+str(count_test)+'.npz'
                    print('save test file:',save_name)
                    count_test+=1
                np.savez(save_name,matrix=X_data,labels=y_data)

                X_data = []
                y_data = []
        if len(X_data) >=50:
            X_data = np.array(X_data)
            y_data = np.array(y_data)
            
            if 'train' in save_path:
                save_name = save_path.split('.')[0]+str(count_train)+'.npz'
                while os.path.exists(save_name):
                    count_train+=1
                    save_name = save_path.split('.')[0]+str(count_train)+'.npz'
                print('save train file:',save_name)
                count_train+=1
            else:
                save_name = save_path.split('.')[0]+str(count_test)+'.npz'
                while os.path.exists(save_name):
                    count_test+=1
                    save_name = save_path.split('.')[0]+str(count_test)+'.npz'
                print('save test file:',save_name)
                count_test+=1
            np.savez(save_name,matrix=X_data,labels=y_data)
