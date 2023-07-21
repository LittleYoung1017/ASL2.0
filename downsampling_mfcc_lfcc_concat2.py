import matplotlib.pyplot as plt
import os
import numpy as np
import librosa
from tqdm import tqdm
import math
# import pandas as pd
import math
from torch.utils.data import Dataset, DataLoader
from MyTool.featureExtraction import featureExtracting
import utils
from utils import MyDataset

from spafe.utils.preprocessing import SlidingWindow


config_path = 'config.json'
hps = utils.get_hparams_from_file(config_path)
data_paths = hps.data.raw_data
X_data = []
y_data = []

for data_path in data_paths:
    data_list = os.listdir(data_path)
    save_path_x = hps.data.prepared_train_data_x
    save_path_y = hps.data.prepared_train_data_y
    datasetName = data_path.split('/')[-1]
    datasetPath = os.path.join(hps.data.downsample_data,datasetName + '2')
    if not os.path.exists(datasetPath):
        os.mkdir(datasetPath)
    count_train=0
    count_test=0

    for i in tqdm(range(len(data_list))):
        
        if i > len(data_list)*0.2:
            save_path = os.path.join(hps.data.downsample_data,datasetName +'2','train_data_mfcc_lfcc.npz')
        else:
            save_path = os.path.join(hps.data.downsample_data,datasetName + '2','test_data_mfcc_lfcc.npz')

        try:
            wav,sr = librosa.load(os.path.join(data_path,data_list[i]),sr=8000)
        except:
            continue
        if len(wav)/sr <=1:
            continue
        if len(data_list[i].split('.')[0].split('_'))==2:
            cut_point = int(data_list[i].split('.')[0].split('_')[1])
        else:
            cut_point = int(data_list[i].split('.')[0].split('_')[2])
        if 'original' in data_list[i]:
            y_data.append(0)
        else:
            y_data.append(1)


        #小波转换
        # datarec = utils.wavelet(data_path=data_path+data_list[i])

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
        

        # print(len(X_data))
 
        if len(X_data) >= 200:
            
            X_data = np.array(X_data)
            y_data = np.array(y_data)
            if 'train' in save_path:
                save_name = save_path.split('.')[0]+str(count_train)+'.npz'
                while os.path.exists(save_name):
                    count_train+=1
                    save_name = save_path.split('.')[0]+str(count_train)+'.npz'
                np.savez(save_name,matrix=X_data,labels=y_data)

                print('save train file:',save_path.split('.')[0]+str(count_train))
                count_train+=1
            else:
                save_name = save_path.split('.')[0]+str(count_test)+'.npz'
                while os.path.exists(save_name):
                    count_test+=1
                    save_name = save_path.split('.')[0]+str(count_test)+'.npz'
                np.savez(save_name,matrix=X_data,labels=y_data)
                
          
                print('save test file:',save_path.split('.')[0]+str(count_test))
                count_test+=1

            X_data = []
            y_data = []

        
        

