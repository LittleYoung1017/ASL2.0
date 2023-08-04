import os
import glob
import sys
import argparse
import logging
import librosa
import json
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
import h5py
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

def modify_config(key, new_value, config_path='config.json'):
    with open(config_path,'r') as file:
        config_data = json.load(file)
    if len(key) == 1:
        config[key[0]] = new_value
    elif len(key) == 2:    
        config_data[key[0]][key[1]] = new_value
    
    with open(config_path,'w') as file:
        json.dump(config_data, file, indent=4)

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
    
#================================================================
#extract and save feature
def featureExtracting(wav,sr,feature_name,**params):
    if feature_name=='mfcc':
        mfccs = mfcc(wav,fs=sr,
                num_ceps=params['num_ceps'],
                window=params['window'],
                nfft=params['nfft'],
                low_freq=params['low_freq'],
                normalize=params['normalize'])
        return mfccs.T
    elif feature_name=='lfcc':
        lfccs= lfcc(wav,fs=sr,
            num_ceps=params['num_ceps'],
            window=params['window'],
            nfilts=params['nfilts'],
            nfft=params['nfft'],
            low_freq=params['low_freq'],
            normalize=params['normalize'])
        return lfccs.T
    elif feature_name=='cqcc':
        cqccs = cqcc(wav,
                    fs=sr,
                    window=params['window'],
                    nfft=params['nfft'],
                    low_freq=params['low_freq'],
                    high_freq=sr/2,
                    normalize="mvn")
        return cqccs.T
    elif feature_name=='mel_spectrogram':
        mel_spec,_ = mel_spectrogram(wav,
                                            fs=sr,
                                            window=params['window'],
                                            nfilts=params['nfilts'],
                                            nfft=params['nfft'],
                                            low_freq=params['low_freq'],
                                            high_freq=sr/2)
        return mel_spec
def extract_melspec(wav,sr):
    spectrogram = librosa.feature.melspectrogram(y=wav, sr=sr)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    return spectrogram_db
def extract_stft(wav,sr):
    return np.abs(librosa.stft(wav))
def saving_feature(audioset_dirs):
    config_path = 'config.json'
    hps = get_hparams_from_file(config_path)
    print('extracting features...')
    data_paths = audioset_dirs
    for data_path in tqdm(data_paths):
        type_list = os.listdir(data_path)

        # datasetName = data_path.split('/')[-2]
        datasetNamePath = data_path.split('split_audio')[0]
        datasetPath = os.path.join(datasetNamePath,'feature_data')
        if not os.path.exists(datasetPath):
            os.mkdir(datasetPath)
        count_train=0
        count_test=0
        for t in type_list:
            if 'train' in t:
                save_path = os.path.join(datasetPath,'train_data_mfcc_lfcc.npz')
            else:
                save_path = os.path.join(datasetPath,'test_data_mfcc_lfcc.npz')
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
                #提取mfcc特征和lfcc特征并合并在一起
                data_feature_mfcc = featureExtracting(wav,sr,hps.data.feature_type[0],
                            num_ceps= hps.data.n_mel_channels[0],
                            window=SlidingWindow(hps.data.win_length,hps.data.hop_length , hps.data.window_type),
                            nfilts=hps.data.nfilts,
                            nfft=hps.data.nfft,
                            low_freq=hps.data.mel_fmin,
                            normalize="mvn")
                data_feature_lfcc = featureExtracting(wav,sr,hps.data.feature_type[1],
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
                if len(X_data) >= 1000:#每1000个音频特征保存一个npz文件
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
            if len(X_data) >=1:
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
    


#=====================================================
# for large dataset training

def load_test_data(hps):   #load test data
    print('Loading data...')
    test_data_x = np.empty((1,32,300))
    test_data_y = np.empty((1,))
    setList = os.listdir(hps.data.downsample_data)
    temp_count=0
    for s in setList:
      setpath = os.path.join(hps.data.downsample_data,s,'feature_data')
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
          temp_count+=1
          if temp_count>=10:
            break
      if temp_count>=10:
        break
    test_data_x = test_data_x[1:]
    test_data_y = test_data_y[1:]
    test_data_x = test_data_x[:,np.newaxis,:,:]  #for LCNN
    test_data_y = test_data_y.astype('int64')
    
    return test_data_x, test_data_y


def preprocess_train_data(batch_data):
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
    train_data_x = train_data_x[:,np.newaxis,:,:]  #for LCNN
    train_data_y = train_data_y.astype('int64')
    return train_data_x, train_data_y
  
def get_train_data_list(hps):
  train_file_all = []
  setList = os.listdir(hps.data.downsample_data)
  for s in setList:
    setpath = os.path.join(hps.data.downsample_data,s,'feature_data')
    npyList = os.listdir(setpath)
    count = sum(1 for item in npyList if 'train' in item)
    npyName = os.path.join(setpath,'train_data_mfcc_lfcc.npz')
    for i in tqdm(range(count)):
        data_name = npyName.split('.')[0]+str(i)+'.npz'
        train_file_all.append(data_name)
  return train_file_all

def train_data_generator(train_file_all,batch_size):   #generate training data
    print('Loading training data.')
    num_data = len(train_file_all)
    indices = np.arange(num_data)
    np.random.shuffle(indices) #随机打乱数据
    
    start_idx = 0
    while start_idx < num_data:
        if start_idx + batch_size >= num_data:#保留最后不足batch部分
            excerpt = indices[ start_idx : num_data]
        else:
            excerpt = indices[ start_idx : start_idx + batch_size]
        batch_data = [train_file_all[i] for i in excerpt ]

        start_idx = start_idx + batch_size
        batch_x, batch_y = preprocess_train_data(batch_data)
        yield batch_x, batch_y
#====================================================
#load data
def load_data_new(hps):

    print('Loading data...')
    train_data_x = np.empty((1,32,300))
    train_data_y = np.empty((1,))
    test_data_x = np.empty((1,32,300))
    test_data_y = np.empty((1,))
    setList = os.listdir(hps.data.downsample_data)
    for s in setList:
        
        setpath = os.path.join(hps.data.downsample_data,s,'feature_data')
        print(setpath)
        npzList = os.listdir(setpath)
        npzList = [ i for i in npzList if '.npz' in i]
        for item in tqdm(npzList):
            npzName = os.path.join(setpath,item)
            data = np.load(npzName)
            matrix = data['matrix']
            labels = data['labels']
            if 'train' in npzName:
                train_data_x = np.concatenate((train_data_x,matrix),axis=0)
                train_data_y = np.concatenate((train_data_y,labels),axis=0)
            elif 'test' in npzName:
                test_data_x = np.concatenate((test_data_x,matrix),axis=0)
                test_data_y = np.concatenate((test_data_y,labels),axis=0)


    train_data_x = train_data_x[1:]
    train_data_x = train_data_x[:,np.newaxis,:,:]  #for LCNN
    train_data_y = train_data_y[1:]
    
    test_data_x = test_data_x[1:]
    test_data_x = test_data_x[:,np.newaxis,:,:]  #for LCNN
    test_data_y = test_data_y[1:]
    
    train_data_y = train_data_y.astype('int64')
    test_data_y = test_data_y.astype('int64')
    print(train_data_x.shape)
    print(test_data_x.shape)
    return train_data_x, train_data_y, test_data_x, test_data_y  

#resample
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
        python utils.py \
        --type resample \
        --drop_last True \
        --s_path /home/yangruixiong/dataset/ESC-50/ESC \
        --t_path /home/yangruixiong/dataset/ESC-50/ESC-8k \
        --s_sr 44100 \
        --t_sr 8000
"""
#=================================================================================
#data cutting 
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
        python utils.py \
        --type splicing \
        --drop_last 1 \
        --s_path /home/yangruixiong/dataset/new_splicing_detection_dataset/southern-power-grid/val \
        --t_path /home/yangruixiong/dataset/new_splicing_detection_dataset/southern-power-grid-3s-8k/val \
        --sr 8000 \
        --cutting_time 3
"""
#=================================================================================
#data concat
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
            cut_pos = random.uniform(0.2,0.8)
            #split
            start1 = int(cut_pos * len(data))

            start2 = int(cut_pos* len(data2))


            split_left, split_right = data[:start1],data[start1:]
            split_left2, split_right2 = data2[:start2],data2[start2:]

            #concat
            audio1 = np.hstack((split_left, split_right2))
            audio2 = np.hstack((split_left2, split_right))

            sf.write(os.path.join(save_path,'tampered_' + str(count) + '.wav'),audio1,sr)
            count+=1
            sf.write(os.path.join(save_path,'tampered_' + str(count) + '.wav'),audio2,sr)
            count+=1

        elif Type == 1:  #两个音频拼接
            splicing_data1 = np.hstack((data,data2))
            splicing_data2 = np.hstack((data2,data))
            
            sf.write(os.path.join(save_path, str(count) +  '_tampered_' + str(len(data)) + '.wav'),splicing_data1,sr)
            count+=1
            sf.write(os.path.join(save_path, str(count) +  '_tampered_' + str(len(data2)) + '.wav'),splicing_data2,sr)
            count+=1

"""
    usage example:
        python utils.py \
            --type concat \
            --drop_last 1 \
            --s_path /home/yangruixiong/dataset/new_splicing_detection_dataset/guangda-bank-3s-8k/val \
            --s_path2 /home/yangruixiong/dataset/new_splicing_detection_dataset/southern-power-grid-3s-8k/val \
            --t_path /home/yangruixiong/dataset/new_splicing_detection_dataset/concat-gdbank-spg-3s-8k-2/val \
            --s_sr 8000 \
            --t_sr 8000 \
            --cutting_time 3
"""

#================================================================================
#dividing train/test set and then resample and split data 
def dividing_train_test_resample_spliting(data_path,save_path,s_sr,t_sr,cutting_time,split_ratio=0.8,drop_last=1):
    #get dataset name
    dataset_Name = data_path.split('/')
    if dataset_Name[-1] == '':
        dataset_Name = dataset_Name[-2]
    else:
        dataset_Name = dataset_Name[-1]
        
    #划分train和test的音频保存位置    
    dataset_name_path = os.path.join(save_path,dataset_Name)
    dividing_dataset_path = os.path.join(dataset_name_path,'split_audio')
    os.makedirs(dividing_dataset_path,exist_ok=True)
    
    train_dir = os.path.join(dividing_dataset_path, 'train')
    val_dir = os.path.join(dividing_dataset_path, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

     # Get list of all files in the dataset directory
    file_list = os.listdir(data_path)
    file_list = [i for i in file_list if '.wav' in i]
    random.shuffle(file_list)
     # Calculate number of files for training and validation
    num_files = len(file_list)
    num_train = int(num_files * split_ratio)
    num_val = num_files - num_train
    
    dividing_len = int(cutting_time * t_sr)

    count=0
    train_dividing_count=0
    test_dividing_count=0
     # Move files to training set
    for file_name in tqdm(file_list):   
        # if '.wav' in file_name: 
        src_path = os.path.join(data_path, file_name)  
        try:
            data,rate = librosa.load(src_path,sr=s_sr)
            data = librosa.resample(y=data, orig_sr=s_sr, target_sr=t_sr)
        except:
            print('Wrong: cant read audio file.')
            continue

        while(len(data)>=dividing_len):
            cut = dividing_len
            split = data[0:cut]
            data = data[cut:]

            if count<num_train:
                path = os.path.join(train_dir,'original_' + str(train_dividing_count) + '.wav')
                train_dividing_count+=1
            else:
                path = os.path.join(val_dir,'original_' + str(test_dividing_count) + '.wav')
                test_dividing_count+=1
            sf.write(path,split,t_sr)
        if drop_last ==0:
            if len(data) >= dividing_len*0.7 :  #音频末尾不足设定长度的部分
                if count<num_train:
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
#audio preprocessing
def audio_preprocessing(data_path_1,data_path_2,save_path,s_sr,s_sr2,t_sr,
                        cutting_time,spliting_ratio=0.8,drop_last=1):
    os.makedirs(save_path,exist_ok = True)

    target_path1,dataset_name1 = dividing_train_test_resample_spliting(data_path,save_path,s_sr,t_sr,
                                                                       cutting_time,drop_last=drop_last)   
    target_path2,dataset_name2 = dividing_train_test_resample_spliting(data_path2,save_path,s_sr2,t_sr,
                                                                       cutting_time,drop_last=drop_last)
    print('audio spliting complete.')

    concat_name = dataset_name1 + '-' + dataset_name2
    concat_save_path = os.path.join(save_path,concat_name,'split_audio')
    train_dir1 = os.path.join(target_path1,'train')
    test_dir1 = os.path.join(target_path1,'val')
    train_dir2 = os.path.join(target_path2,'train')
    test_dir2 = os.path.join(target_path2,'val')
    train_save_path = os.path.join(concat_save_path,'train')
    test_save_path = os.path.join(concat_save_path,'val')

    concat(train_dir1,train_dir2,train_save_path,t_sr)
    concat(test_dir1,test_dir2,test_save_path,t_sr)
    
    #记录生成的分割和拼接数据目录

    print('audio concating complete.')
    
    audioset_dirs = [target_path1,target_path2,concat_save_path]
    saving_feature(audioset_dirs)

"""
    usage example:

        python utils.py \
            --type audio_preprocessing \
            --drop_last 1 \
            --s_path /home/yangruixiong/dataset/train \
            --s_path2 /home/yangruixiong/dataset/test \
            --t_path /home/yangruixiong/ASL2/data \
            --s_sr 44100 \
            --s_sr2 44100 \
            --t_sr 8000 \
            --cutting_time 3
"""
#====================================================================================
#calculate audio num
def audio_num(npz_data_path):
    set_list = os.listdir(npz_data_path)
    train_num_all=0
    test_num_all=0
    for set_name in set_list:
        set_path = os.path.join(npz_data_path, set_name,'feature_data')
        data_list = os.listdir(set_path)
        train_num_per_set=0
        test_num_per_set=0
        for data_file in data_list:
            data = np.load(os.path.join(set_path,data_file))
            data_x = data['matrix']
            if 'train' in data_file: 
                train_num_per_set += len(data_x)
                train_num_all +=len(data_x)
            elif 'test' in data_file:
                test_num_per_set += len(data_x)
                test_num_all +=len(data_x)
        print(set_name,', train:',train_num_per_set,' test:',test_num_per_set)

    print('all train file number:',train_num_all)
    print('all test file number:',test_num_all)
'''
    python utils.py --type audio_num
'''    

#=====================================================================================
def spliting_npz_file(npz_data_path):
    save_file_path = '/home/yangruixiong/ASL2/single_data/'
    os.makedirs(save_file_path,exist_ok=True)
    set_list = os.listdir(npz_data_path)
    for set_name in tqdm(set_list):
        set_path = os.path.join(npz_data_path, set_name,'feature_data')
        data_list = os.listdir(set_path)
        save_set_path = os.path.join(save_file_path, set_name,'feature_data')
        os.makedirs(save_set_path,exist_ok=True)
        
        for data_file in data_list:
            count = 0
            data = np.load(os.path.join(set_path,data_file))
            data_x = data['matrix']
            data_y = data['labels']
            for i in range(len(data_x)):
                x, y = data_x[i], data_y[i]
                save_path = os.path.join(save_set_path ,data_file + '_' + str(count)) 
                np.savez(save_path,matrix=x,labels=y)               
                count+=1
'''
    python utils.py  \
    --type spliting_npz_file  \
    --s_path /home/yangruixiong/ASL2/data
'''

#=============================================================
def npz_to_hdf5(npz_data_path):
    import h5py
    set_list = os.listdir(npz_data_path)
    save_file_path = '/home/yangruixiong/ASL2/data_hdf5'
    os.makedirs(save_file_path,exist_ok = True)
    for set_name in tqdm(set_list):
        set_path = os.path.join(npz_data_path, set_name,'feature_data')
        data_list = os.listdir(set_path)
        save_set_path = os.path.join(save_file_path, set_name,'feature_data')
        os.makedirs(save_set_path,exist_ok=True)
        train_x = np.empty((1,32,300))
        train_y = np.empty((1,))
        test_x = np.empty((1,32,300))
        test_y = np.empty((1,))
        for data_file in data_list:
            data = np.load(os.path.join(set_path,data_file))
            data_x = data['matrix']
            data_y = data['labels']
            if 'train' in data_file:
                train_x = np.concatenate((train_x,data_x),axis=0)
                train_y = np.concatenate((train_y,data_y),axis=0)
            elif 'test' in data_file:
                test_x = np.concatenate((test_x,data_x),axis=0)
                test_y = np.concatenate((test_y,data_y),axis=0)
        new_train_file = os.path.join(save_set_path ,'train_data.h5')
        new_test_file = os.path.join(save_set_path ,'test_data.h5')
        if len(train_x) > 1:
            f = h5py.File(new_train_file, 'w')
            f['data'] = train_x[1:]
            f['labels'] = train_y[1:]
            f.close()      
        if len(test_x) > 1:
            f = h5py.File(new_test_file, 'w')
            f['data'] = test_x[1:]
            f['labels'] = test_y[1:]
            f.close()
'''
    python utils.py \
        --type npz_to_hdf5  \
        --s_path /home/yangruixiong/ASL2/data
'''     
 
#===========
#load h5 to np
def load_h5_to_np(path):
    
    print('Loading data...')
    train_data_x = np.empty((1,32,300))
    train_data_y = np.empty((1,))
    test_data_x = np.empty((1,32,300))
    test_data_y = np.empty((1,))
    setList = os.listdir(path)
    for s in tqdm(setList):
        setpath = os.path.join(path,s,'feature_data')
        print(setpath)
        npzList = os.listdir(setpath)
        for item in npzList:
            h5file = os.path.join(setpath,item)
            data = h5py.FIle(h5file,'r')
            matrix = data['matrix']
            labels = data['labels']
            if 'train' in h5file:
                train_data_x = np.concatenate((train_data_x,matrix),axis=0)
                train_data_y = np.concatenate((train_data_y,labels),axis=0)
            elif 'test' in h5file:
                test_data_x = np.concatenate((test_data_x,matrix),axis=0)
                test_data_y = np.concatenate((test_data_y,labels),axis=0)


    train_data_x = train_data_x[1:]
    train_data_x = train_data_x[:,np.newaxis,:,:]  #for LCNN
    train_data_y = train_data_y[1:]
    
    test_data_x = test_data_x[1:]
    test_data_x = test_data_x[:,np.newaxis,:,:]  #for LCNN
    test_data_y = test_data_y[1:]
    
    train_data_y = train_data_y.astype('int64')
    test_data_y = test_data_y.astype('int64')
    print(train_data_x.shape)
    print(test_data_x.shape)
    return train_data_x, train_data_y, test_data_x, test_data_y  
'''
    python utils.py \
        --type load_h5_to_np    \
        --s_path /home/yangruixiong/ASL2/data_hdf5
'''
#=======================================================================================
#main
if __name__ == '__main__':
    config_path = 'config.json'
    hps = get_hparams_from_file(config_path)
    
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
        modify_config(['data','downsample_data'],save_path)
        audio_preprocessing(data_path,data_path2,save_path,s_sr,s_sr2,t_sr,cutting_time,spliting_ratio=0.8)
    elif args.type =='audio_num':
        audio_num('/home/yangruixiong/ASL2/data')
    elif args.type =='spliting_npz_file':
        spliting_npz_file(data_path)
    elif args.type == 'npz_to_hdf5':
        npz_to_hdf5(data_path)
    elif args.type == 'load_h5_to_np':
        load_h5_to_np(data_path)
# if __name__ == '__main__':
    
    # test_audio_path = '/home/yangruixiong/dataset/ESC-50/ESC/1-137-A-32.wav'
    # wav, sr = librosa.load(test_audio_path)
    # data = extract_stft(wav,sr)
    # data2,_ = mel_spectrogram(wav,sr)
    # mfcc = mfcc(wav,sr)
    # print(data.shape)
    # print(data2.shape)
    # print(mfcc.shape)
    # data_path = '/home/yangruixiong/ASL2/data'
    # audio_num(data_path)
    