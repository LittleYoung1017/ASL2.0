import os
import glob
import sys
import argparse
import logging
import librosa
import json
import pywt
import subprocess
import numpy as np
import pandas as pd
from scipy.io.wavfile import read
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import random
import soundfile as sf
import math
import shutil
from spafe.features.mfcc import mfcc
from spafe.features.lfcc import lfcc
from spafe.features.cqcc import cqcc
from spafe.features.mfcc import mel_spectrogram
from spafe.utils.preprocessing import SlidingWindow

MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging


def read_file(file_path):
    # Open the file in read mode
    with open(file_path, 'r') as file:
        # Read the contents of the file
        lines = file.readlines()
    
    # Remove newline characters and whitespace from each line
    lines = [line.strip() for line in lines]
    
    return lines


def get_hparams_from_file(config_path):
  with open(config_path, "r") as f:
    data = f.read()
  config = json.loads(data)

  hparams =HParams(**config)
  return hparams

def get_logger(model_dir, filename="train.log"):
  global logger
  logger = logging.getLogger(os.path.basename(model_dir))
  logger.setLevel(logging.DEBUG)
  
  formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
  if not os.path.exists(model_dir):
    os.makedirs(model_dir)
  h = logging.FileHandler(os.path.join(model_dir, filename))
  h.setLevel(logging.DEBUG)
  h.setFormatter(formatter)
  logger.addHandler(h)
  return logger


class HParams():
  def __init__(self, **kwargs):
    for k, v in kwargs.items():
      if type(v) == dict:
        v = HParams(**v)
      self[k] = v
    
  def keys(self):
    return self.__dict__.keys()

  def items(self):
    return self.__dict__.items()

  def values(self):
    return self.__dict__.values()

  def __len__(self):
    return len(self.__dict__)

  def __getitem__(self, key):
    return getattr(self, key)

  def __setitem__(self, key, value):
    return setattr(self, key, value)

  def __contains__(self, key):
    return key in self.__dict__

  def __repr__(self):
    return self.__dict__.__repr__()

  
class MyDataset(Dataset):

    def __init__(self,X_data,y_data):
        self._x = torch.from_numpy(X_data)
        self._y = torch.from_numpy(y_data)
        self._len = y_data.shape[0]

    def __getitem__(self, item):  # 每次循环的时候返回的值
        return self._x[item], self._y[item]

    def __len__(self):
        return self._len

def wavelet(data_path=None,data=None,sr=8000,wt_type='sym8',ts_type='soft',maxlev=5):
    if data_path:
        data, sr = librosa.load(data_path, sr=sr)
    #加载信号
    index = [i/sr for i in range(len(data))]

    # Create wavelet object and define parameters
    w = pywt.Wavelet(wt_type)  # 选用sym8小波
    # maxlev = pywt.dwt_max_level(len(data), w.dec_len)
    # print("maximum level is " + str(maxlev))
    threshold = 0.04  #0.04 Threshold for filtering

    # Decompose into wavelet components, to the level selected:
    coeffs = pywt.wavedec(data, wt_type, level=maxlev)  # 将信号进行小波分解

    #plt.figure()
    for i in range(1, len(coeffs)):
        coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]),mode=ts_type)  # 将噪声滤波

    datarec = pywt.waverec(coeffs, wt_type)  # 将信号进行小波重构

    if len(data) % 2 == 1:
        datarec = datarec[:-1]
    
    return datarec

def featureExtracting(wav,sr,feature_name,**params):
    if feature_name=='mfcc':
        mfccs = mfcc(wav,fs=sr,
                pre_emph=1,
                pre_emph_coeff=0.97,
                num_ceps=params['num_ceps'],
                window=SlidingWindow(0.03, 0.015, "hamming"),
                nfft=512,
                low_freq=0,
                normalize="mvn")
        return mfccs.T
    elif feature_name=='lfcc':
        lfccs= lfcc(wav,fs=sr,
                pre_emph=1,
            pre_emph_coeff=0.97,
            num_ceps=params['num_ceps'],
            window=SlidingWindow(0.03, 0.015, "hamming"),
            nfilts=128,
            nfft=512,
            low_freq=0,
            normalize="mvn")
        return lfccs.T
    elif feature_name=='cqcc':
        cqccs = cqcc(wav,
                    fs=sr,
                    pre_emph=1,
                    pre_emph_coeff=0.97,
                    window=SlidingWindow(0.03, 0.015, "hamming"),
                    nfft=512,
                    low_freq=0,
                    high_freq=sr/2,
                    normalize="mvn")
        return cqccs.T
    elif feature_name=='mel_spectrogram':
        mel_spec,_ = mel_spectrogram(wav,
                                            fs=sr,
                                            pre_emph=0,
                                            pre_emph_coeff=0.97,
                                            window=SlidingWindow(0.03, 0.015, "hamming"),
                                            nfilts=24,
                                            nfft=512,
                                            low_freq=0,
                                            high_freq=sr/2)
        return mel_spec
 
def saving_feature():
    config_path = 'config.json'
    hps = get_hparams_from_file(config_path)
    print('extracting features...')
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
            # for i in tqdm(range(int(len(data_list)/4))):
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
            if len(X_data) >=2:
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
    



def load_data_npz(hps):   #load test data
    print('Loading data...')
    test_data_x = np.empty((1,32,300))
    test_data_y = np.empty((1,))
    setList = os.listdir(hps.data.downsample_data)
    for s in setList:
      setpath = os.path.join(hps.data.downsample_data,s)
      print(setpath)
      npyList = os.listdir(setpath)
    
      count = sum(1 for item in npyList if 'test' in item)
      for i in tqdm(range(count)):
          npyName = os.path.join(setpath,'test_data_mfcc_lfcc.npz')
          data = np.load(npyName.split('.')[0]+str(i)+'.npz')
          matrix = data['matrix']
          labels = data['labels']
          test_data_x = np.concatenate((test_data_x,matrix),axis=0)
          test_data_y = np.concatenate((test_data_y,labels),axis=0)


    test_data_x = test_data_x[1:]
    test_data_y = test_data_y[1:]
    test_data_y = test_data_y.astype('int64')
    
    return test_data_x, test_data_y


def preprocess_data(batch_data):
    train_data_x = np.empty((1,32,300))
    train_data_y = np.empty((1,))
    for name in tqdm(batch_data):
      data = np.load(name)
      matrix = data['matrix']
      labels = data['labels']
      train_data_x = np.concatenate((train_data_x,matrix),axis=0)
      train_data_y = np.concatenate((train_data_y,labels),axis=0)
    train_data_x = train_data_x[1:]
    train_data_y = train_data_y[1:]
    train_data_y = train_data_y.astype('int64')
    return train_data_x, train_data_y


def load_data_new(hps):

    print('Loading data...')
    train_data_x = np.empty((1,32,300))
    train_data_y = np.empty((1,))
    test_data_x = np.empty((1,32,300))
    test_data_y = np.empty((1,))
    setList = os.listdir(hps.data.downsample_data)
    for s in setList:
        
        setpath = os.path.join(hps.data.downsample_data,s)
        print(setpath)
        npyList = os.listdir(setpath)
        npyList = [ i for i in npyList if '.npz' in i]
        for item in tqdm(npyList):
            npyName = os.path.join(setpath,item)
            data = np.load(npyName)
            matrix = data['matrix']
            labels = data['labels']
            if 'train' in npyName:
                train_data_x = np.concatenate((train_data_x,matrix),axis=0)
                train_data_y = np.concatenate((train_data_y,labels),axis=0)
            elif 'test' in npyName:
                test_data_x = np.concatenate((test_data_x,matrix),axis=0)
                test_data_y = np.concatenate((test_data_y,labels),axis=0)


    train_data_x = train_data_x[1:]
    train_data_y = train_data_y[1:]
    test_data_x = test_data_x[1:]
    test_data_y = test_data_y[1:]
    train_data_y = train_data_y.astype('int64')
    test_data_y = test_data_y.astype('int64')
    print("train_data_x:",train_data_x.shape)
    print("train_data_y:",train_data_y.shape)
    print("test_data_x:",test_data_x.shape)
    print("test_data_y:",test_data_y.shape)

    return train_data_x, train_data_y, test_data_x, test_data_y  


def resample(file_path,save_path,sr,sr_re=8000):
    file_list = os.listdir(file_path)
    sr=sr
    sr_re = sr_re 
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    for i in tqdm(range(len(file_list))):
        try:
            data,rate = librosa.load(file_path+file_list[i],sr=sr)
            data = librosa.resample(y=data, orig_sr=sr, target_sr=sr_re)

            sf.write(os.path.join(save_path,file_list[i]),data,sr_re)
        except:
            continue

"""
    usage:
        python data_splicing_utils.py \
        --type resample \
        --drop_last True \
        --s_path /home/yangruixiong/dataset/ESC-50/ESC \
        --t_path /home/yangruixiong/dataset/ESC-50/ESC-8k \
        --s_sr 44100 \
        --t_sr 8000
"""
#=================================================================================

        
def data_splicing(data_path,save_path,sr,cutting_time,drop_last):
    data_list = os.listdir(data_path)
    
    dividing_len = int(cutting_time * sr)
    count=0
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    for i in tqdm(range(len(data_list))):

        data, sr = librosa.load(os.path.join(data_path,data_list[i]),sr=sr)
        time = librosa.get_duration(y=data,sr=sr)
 
        if time < 2.9:
            continue
        while(len(data)>dividing_len):
            cut = dividing_len
            # cut = int(random.uniform(0.95,1.05) * dividing_len)
            split = data[0:cut]
            data = data[cut:]
            # path = save_path + data_list[i].split('.')[0]+ '_'+str(count)+'.wav'
            path = os.path.join(save_path,'original_' + str(count) + '.wav')
            sf.write(path,split,sr)
            count+=1
        if drop_last == 0:    #save the last part of the audio 
            if len(data)>0:
                path = os.path.join(save_path,data_list[i].split('.')[0]+ '_'+str(count)+'.wav')
                print(path)
                sf.write(path,data,sr)
"""
    usage:
        python data_splicing_utils.py \
        --type splicing \
        --drop_last 1 \
        --s_path /home/yangruixiong/dataset/new_splicing_detection_dataset/southern-power-grid/val \
        --t_path /home/yangruixiong/dataset/new_splicing_detection_dataset/southern-power-grid-3s-8k/val \
        --sr 8000 \
        --cutting_time 3
"""
#=================================================================================

def resample_and_splicing(data_path,save_path,sr,sr_re,cutting_time,drop_last):

    sr=sr
    sr_re = sr_re 
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    dividing_len = int(cutting_time * sr_re)
    count=0
    data_list = os.listdir(data_path)
    for i in tqdm(range(len(data_list))):
        try:
            data,rate = librosa.load(os.path.join(data_path,data_list[i]),sr=sr)
        except:
            continue
        data = librosa.resample(y=data, orig_sr=sr, target_sr=sr_re)
        time = librosa.get_duration(y=data,sr=sr_re)
        print(time)

        while(len(data)>=dividing_len):
            cut = dividing_len
            # cut = int(random.uniform(0.95,1.05) * dividing_len)
            split = data[0:cut]
            data = data[cut:]
            # path = save_path + data_list[i].split('.')[0]+ '_'+str(count)+'.wav'
            path = os.path.join(save_path,'original_' + str(count) + '.wav')
            sf.write(path,split,sr_re)
            count+=1
        print(len(data))
        print(drop_last)
        if drop_last == 0:
            if len(data)>0:
                path = os.path.join(save_path,data_list[i].split('.')[0]+ '_'+str(count)+'.wav')
                print(path)
                sf.write(path,data,sr_re)

"""
    usage:
        python data_splicing_utils.py \
            --type resample_splicing \
            --drop_last False \
            --s_path /home/yangruixiong/dataset/ESC-50/ESC \
            --t_path /home/yangruixiong/dataset/ESC-50/ESC-3s-8k \
            --s_sr 44100 \
            --t_sr 8000 \
            --cutting_time 3
"""
#==============================================================================

def concat(data_path,data_path_2,save_path,sr):
    file_path_1 = data_path
    file_path_2 = data_path_2
    save_path = save_path
    file_list1 = os.listdir(file_path_1)
    file_list2 = os.listdir(file_path_2)
    count = 1
    random.shuffle(file_list1)
    random.shuffle(file_list2)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    for i in tqdm(range(min(len(file_list1),len(file_list2)))):
        data,rate = librosa.load(os.path.join(file_path_1,file_list1[i]),sr=sr)
        data2,rate2 = librosa.load(os.path.join(file_path_2,file_list2[i]),sr=sr)

        time1 = librosa.get_duration(y=data,sr=sr)
        time2= librosa.get_duration(y=data2,sr=sr)
        Type = 0
        if Type == 0:  #音频两部分进行两两拼凑
            cut_pos = random.uniform(0.1,0.9)
            #split
            start1 = int(cut_pos * len(data))
            #print(start1)
            start2 = int(cut_pos* len(data2))

            #stop = start + duration
            split_left, split_right = data[:start1],data[start1:]
            split_left2, split_right2 = data2[:start2],data2[start2:]

            #concat
            audio1 = np.hstack((split_left, split_right2))
            audio2 = np.hstack((split_left2, split_right))

            sf.write(os.path.join(save_path,'tampered_' + str(count) + '.wav'),audio1,sr)
            count+=1
            sf.write(os.path.join(save_path,'tampered_' + str(count) + '.wav'),audio2,sr)
            count+=1
            #save file with splicing position
            # sf.write(os.path.join(save_path,str(count) +  '_tampered_' + str(start1) + '.wav'),audio1,sr)
            # count+=1
            # sf.write(os.path.join(save_path,str(count) +  '_tampered_' + str(start2) + '.wav'),audio2,sr)
            # count+=1
        elif Type == 1:  #两个音频拼接
            splicing_data1 = np.hstack((data,data2))
            splicing_data2 = np.hstack((data2,data))
            
            sf.write(os.path.join(save_path, str(count) +  '_tampered_' + str(len(data)) + '.wav'),splicing_data1,sr)
            count+=1
            sf.write(os.path.join(save_path, str(count) +  '_tampered_' + str(len(data2)) + '.wav'),splicing_data2,sr)
            count+=1

"""
    usage example:
        python data_splicing_utils.py \
            --type concat \
            --drop_last 1 \
            --s_path /home/yangruixiong/dataset/new_splicing_detection_dataset/guangda-bank-3s-8k/train \
            --s_path2 /home/yangruixiong/dataset/new_splicing_detection_dataset/southern-power-grid-3s-8k/train \
            --t_path /home/yangruixiong/dataset/new_splicing_detection_dataset/concat-gdbank-spg-3s-8k/train \
            --s_sr 8000 \
            --t_sr 8000 \
            --cutting_time 3
"""

#================================================================================
def dividing_train_test_resample_spliting(data_path,save_path,s_sr,t_sr,cutting_time,split_ratio=0.8,feature_extraction=1):
    #get dataset name
    dataset_Name = data_path.split('/')
    if dataset_Name[-1] == '':
        dataset_Name = dataset_Name[-2]
    else:
        dataset_Name = dataset_Name[-1]
        
    #划分train和test的音频保存位置    
    dividing_dataset_path = os.path.join(save_path,dataset_Name)
    
    train_dir = os.path.join(dividing_dataset_path, 'train')
    val_dir = os.path.join(dividing_dataset_path, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

     # Get list of all files in the dataset directory
    file_list = os.listdir(data_path)
    data_path = [i for i in data_path if '.wav' in i]
    random.shuffle(file_list)
     # Calculate number of files for training and validation
    num_files = len(file_list)
    num_train = int(num_files * split_ratio)
    num_val = num_files - num_train
    
    dividing_len = int(cutting_time * t_sr)

    count=0

     # Move files to training set
    for file_name in tqdm(file_list):   #训练集部分    
        # if '.wav' in file_name: 
        src_path = os.path.join(data_path, file_name)   #初始文件路径
        try:
            data,rate = librosa.load(src_path,sr=s_sr)
        except:
            continue
        train_dividing_count=0
        test_dividing_count=0
        while(len(data)>=dividing_len):
            cut = dividing_len
            # cut = int(random.uniform(0.95,1.05) * dividing_len)
            split = data[0:cut]
            data = data[cut:]

            if count<num_train:
            # path = save_path + data_list[i].split('.')[0]+ '_'+str(count)+'.wav'
                path = os.path.join(train_dir,'original_' + str(train_dividing_count) + '.wav')
                train_dividing_count+=1
            else:
                path = os.path.join(val_dir,'original_' + str(test_dividing_count) + '.wav')
                test_dividing_count+=1
            sf.write(path,split,t_sr)
        count+=1

    print(f"Dataset split complete. {num_train} files moved to training set, {num_val} files moved to validation set.")
    return dividing_dataset_path,dataset_Name

"""
    usage example:

        python utils.py \
            --type dividing_resample_spliting_concat \
            --drop_last 1 \
            --s_path /home/yangruixiong/dataset/splicing-detection/music/music-8k-p11 \
            --s_path2 /home/yangruixiong/dataset/splicing-detection/music/music-8k-p22 \
            --t_path /home/yangruixiong/ASL2/ASL10_data \
            --s_sr 8000 \
            --t_sr 8000 \
            --cutting_time 3
"""
#=================================================================================
def dividing_resample_spliting_concat(data_path_1,data_path_2,save_path,s_sr,s_sr2,t_sr,cutting_time,spliting_ratio=0.8):
    target_path1,dataset_name1 = dividing_train_test_resample_spliting(data_path,save_path,s_sr,t_sr,cutting_time,feature_extraction=1)   
    target_path2,dataset_name2 = dividing_train_test_resample_spliting(data_path2,save_path,s_sr2,t_sr,cutting_time,feature_extraction=1)
    print('audio spliting complete.')

    concat_name = dataset_name1 + '-' + dataset_name2
    save_path = os.path.join(save_path,concat_name)
    train_dir1 = os.path.join(target_path1,'train')
    test_dir1 = os.path.join(target_path1,'val')
    train_dir2 = os.path.join(target_path2,'train')
    test_dir2 = os.path.join(target_path2,'val')
    train_save_path = os.path.join(save_path,'train')
    test_save_path = os.path.join(save_path,'val')

    concat(train_dir1,train_dir2,train_save_path,t_sr)
    concat(test_dir1,test_dir2,test_save_path,t_sr)
    os.makedirs('data',exist_ok=True)
    #记录生成的分割和拼接数据目录
    with open('data/data_path.txt','w+') as f:
        f.write(target_path1+'\n')
        f.write(target_path2+'\n')
        f.write(save_path+'\n')
    print('audio concating complete.')
    
"""
    usage example:

        python utils.py \
            --type dividing_resample_spliting_concat \
            --drop_last 1 \
            --s_path /home/yangruixiong/dataset/splicing-detection/music/music-8k-p11 \
            --s_path2 /home/yangruixiong/dataset/splicing-detection/music/music-8k-p22 \
            --t_path /home/yangruixiong/ASL2/ASL10_data \
            --s_sr 8000 \
            --s_sr2 8000 \
            --t_sr 8000 \
            --cutting_time 3
"""
#=================================================================================
def audio_preprocessing(data_path_1,data_path_2,save_path,s_sr,s_sr2,t_sr,cutting_time,spliting_ratio=0.8,):
    target_path1,dataset_name1 = dividing_train_test_resample_spliting(data_path,save_path,s_sr,t_sr,cutting_time,feature_extraction=1)   
    target_path2,dataset_name2 = dividing_train_test_resample_spliting(data_path2,save_path,s_sr2,t_sr,cutting_time,feature_extraction=1)
    print('audio spliting complete.')

    concat_name = dataset_name1 + '-' + dataset_name2
    save_path = os.path.join(save_path,concat_name)
    train_dir1 = os.path.join(target_path1,'train')
    test_dir1 = os.path.join(target_path1,'val')
    train_dir2 = os.path.join(target_path2,'train')
    test_dir2 = os.path.join(target_path2,'val')
    train_save_path = os.path.join(save_path,'train')
    test_save_path = os.path.join(save_path,'val')

    concat(train_dir1,train_dir2,train_save_path,t_sr)
    concat(test_dir1,test_dir2,test_save_path,t_sr)
    
    #记录生成的分割和拼接数据目录
    with open('data/data_path.txt','w+') as f:
        f.write(target_path1+'\n')
        f.write(target_path2+'\n')
        f.write(save_path+'\n')
    print('audio concating complete.')
    saving_feature()

    

"""
    usage example:

        python utils2.py \
            --type audio_preprocessing \
            --drop_last 1 \
            --s_path /home/yangruixiong/dataset/splicing-detection/music/music-8k-p11 \
            --s_path2 /home/yangruixiong/dataset/splicing-detection/music/music-8k-p22 \
            --t_path /home/yangruixiong/ASL2/ASL11_data \
            --s_sr 8000 \
            --s_sr2 8000 \
            --t_sr 8000 \
            --cutting_time 3
"""
#====================================================================================
def split_dataset(dataset_path1, data_path2, save_path1, save_path2, concat_save_path,split_ratio=0.8):
    # Create directories for training and validation sets
    train_dir = os.path.join(save_path, 'train')
    val_dir = os.path.join(save_path, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
     # Get list of all files in the dataset directory
    file_list = os.listdir(dataset_path)
    random.shuffle(file_list)
     # Calculate number of files for training and validation
    num_files = len(file_list)
    num_train = int(num_files * split_ratio)
    num_val = num_files - num_train
     # Move files to training set
    for file_name in tqdm(file_list[:num_train]):
        if '.wav' in file_name: 
            src_path = os.path.join(dataset_path, file_name)
            dest_path = os.path.join(train_dir, file_name)
            print(src_path)
            print(dest_path)
            shutil.move(src_path, dest_path)
     # Move files to validation set
    for file_name in tqdm(file_list[num_train:]):
        if '.wav' in file_name: 
            src_path = os.path.join(dataset_path, file_name)
            dest_path = os.path.join(val_dir, file_name)
            shutil.move(src_path, dest_path)
    print(f"Dataset split complete. {num_train} files moved to training set, {num_val} files moved to validation set.")

"""
    usage example:
        python data_splicing_utils.py \
            --type split_dataset \
            --s_path /home/yangruixiong/dataset/guangda-bank/original-data \
            --t_path /home/yangruixiong/dataset/new_splicing_detection_dataset/guangda-bank
"""
#=====================================================================================
if __name__ == '__main__':


    config_path = 'config.json'
    hps = get_hparams_from_file(config_path)
    data_paths = read_file(hps.data.raw_data)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", type=str, help="type of the function")
    parser.add_argument("--drop_last", type=int, default=1, help="drop pr keep the last segment ")
    parser.add_argument("--s_path",type=str, help="source file path")
    parser.add_argument("--s_path2",type=str, help="second source file path")
    parser.add_argument("--t_path",type=str, help="target file path")
    parser.add_argument("--sr",type=int, default=8000,help="sample rate")
    parser.add_argument("--s_sr",type=int,default=8000, help="source sample rate")
    parser.add_argument("--s_sr2",type=int,default=8000,help="second soure sample rate")
    parser.add_argument("--t_sr",type=int,default=8000, help="target sample rate")
    parser.add_argument("--cutting_time",type=float,default=3,help="data spliting interval")
    parser.add_argument("--split_ratop",type=float,default=0.8,help="training and validation dataset ratio")
    args = parser.parse_args()
    types= args.type
    data_path = args.s_path
    data_path2 = args.s_path2
    save_path = args.t_path
    drop_last = args.drop_last
    sr = args.sr
    s_sr = args.s_sr
    s_sr2 = args.s_sr2
    t_sr = args.t_sr
    cutting_time = args.cutting_time

    if args.type == 'resample':
        resample(data_path,save_path,s_sr,t_sr)
    elif args.type == 'splicing':
        data_splicing(data_path,save_path,s_sr,cutting_time,drop_last)
    elif args.type == 'resample_splicing':
        resample_and_splicing(data_path,save_path,s_sr,t_sr,cutting_time,drop_last)
    elif args.type == 'concat':
        concat(data_path,data_path2,save_path,sr)
    elif args.type == 'split_dataset':
        split_dataset(data_path,save_path)
    elif args.type =='dividing_resample_spliting_concat':
        dividing_resample_spliting_concat(data_path,data_path2,save_path,s_sr,s_sr2,t_sr,cutting_time,spliting_ratio=0.8)
    elif args.type =='audio_preprocessing':
        audio_preprocessing(data_path,data_path2,save_path,s_sr,s_sr2,t_sr,cutting_time,spliting_ratio=0.8)
        
        