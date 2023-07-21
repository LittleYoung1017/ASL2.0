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
from scipy.io.wavfile import read
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import logging
from tqdm import tqdm
import random
MATPLOTLIB_FLAG = False

logging.basicConfig(stream=sys.stdout, level=logging.WARNING)
logger = logging


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



def load_test_data(hps):
    test_data_x = np.empty((1,32,300))
    test_data_y = np.empty((1,))
    test_path = hps.test.test_data
    test_list = os.listdir(test_path)
    
    for i in range(len(test_list)):
      wav,sr = librosa.load( os.path.join(test_path,test_list[i]),sr=8000)
      if len(files[i].split('.')[0].split('_'))==2:
          cut_point = int(files[i].split('.')[0].split('_')[1])
      elif len(files[i].split('.')[0].split('_'))==3:
          cut_point = int(files[i].split('.')[0].split('_')[2])
      else:
          cut_point = 'None'
      cut_points.append((cut_point,len(wav)/sr))
    
    
    #===cutting audio into 3s 
"""
def load_data_npz(hps):
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
      count = sum(1 for item in npyList if 'train' in item)
      for i in tqdm(range(count)):
          npyName = os.path.join(setpath,'train_data_mfcc_lfcc.npz')
          data = np.load(npyName.split('.')[0]+str(i)+'.npz')
          
          matrix = data['matrix']
          labels = data['labels']
          train_data_x = np.concatenate((train_data_x,matrix),axis=0)
          train_data_y = np.concatenate((train_data_y,labels),axis=0)
          
      count = sum(1 for item in npyList if 'test' in item)
      for i in tqdm(range(count)):
          npyName = os.path.join(setpath,'test_data_mfcc_lfcc.npz')
          data = np.load(npyName.split('.')[0]+str(i)+'.npz')
          matrix = data['matrix']
          labels = data['labels']
          test_data_x = np.concatenate((test_data_x,matrix),axis=0)
          test_data_y = np.concatenate((test_data_y,labels),axis=0)

    train_data_x = train_data_x[1:]
    train_data_y = train_data_y[1:]
    test_data_x = test_data_x[1:]
    test_data_y = test_data_y[1:]
    train_data_y = train_data_y.astype('int64')
    test_data_y = test_data_y.astype('int64')
    perm_train = np.random.permutation(len(train_data_y))
    shuffled_train_data_x = train_data_x[perm_train]
    shuffled_train_data_y = train_data_y[perm_train]
    
    perm_test = np.random.permutation(len(test_data_x))
    shuffled_test_data_x = test_data_x[perm_test]
    shuffled_test_data_y = test_data_y[perm_test]

    return shuffled_train_data_x, shuffled_train_data_y, shuffled_test_data_x, shuffled_test_data_y
"""  

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
  
def get_train_data_list(hps):
  train_file_all = []
  setList = os.listdir(hps.data.downsample_data)
  for s in setList:
    setpath = os.path.join(hps.data.downsample_data,s)
    print(setpath)
    npyList = os.listdir(setpath)
    count = sum(1 for item in npyList if 'train' in item)
    npyName = os.path.join(setpath,'train_data_mfcc_lfcc.npz')
    for i in tqdm(range(count)):
        data_name = npyName.split('.')[0]+str(i)+'.npz'
        train_file_all.append(data_name)
  return train_file_all

def data_generator(train_file_all,batch_size):   #generate training data
    num_data = len(train_file_all)
    indices = np.arange(num_data)
    np.random.shuffle(indices) #随机打乱数据
    
    start_idx = 0
    while start_idx < num_data:
        if start_idx + batch_size >= num_data:#保留最后不足batch部分
            excerpt = indices[ start_idx : num_data]
        excerpt = indices[ start_idx : start_idx + batch_size]
        batch_data = [train_file_all[i] for i in excerpt ]

        start_idx = start_idx + batch_size
        batch_x, batch_y = preprocess_data(batch_data)
        yield batch_x, batch_y

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
      count = sum(1 for item in npyList if 'train' in item)
      for i in tqdm(range(int(count/2-1))):
          npyName = os.path.join(setpath,'train_data_mfcc_lfcc.npz')
          data = np.load(npyName.split('.')[0]+str(i)+'.npz')
          matrix = data['matrix']
          labels = data['labels']
          train_data_x = np.concatenate((train_data_x,matrix),axis=0)
          train_data_y = np.concatenate((train_data_y,labels),axis=0)

      count = sum(1 for item in npyList if 'test' in item)
      for i in tqdm(range(int(count/2-1))):
          npyName = os.path.join(setpath,'test_data_mfcc_lfcc.npz')
          data = np.load(npyName.split('.')[0]+str(i)+'.npz')
          matrix = data['matrix']
          labels = data['labels']
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
    
def load_data(hps):

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
      count = sum(1 for item in npyList if 'train' in item)
      for i in tqdm(range(int(count/2-1))):
          x_npyName = os.path.join(setpath,hps.data.prepared_train_data_x)
          single_data_x = np.load(x_npyName.split('.')[0]+str(i)+'.npy')
          train_data_x = np.concatenate((train_data_x,single_data_x),axis=0)
          y_npyName = os.path.join(setpath,hps.data.prepared_train_data_y)
          single_data_y = np.load(y_npyName.split('.')[0]+str(i)+'.npy')
          train_data_y = np.concatenate((train_data_y,single_data_y),axis=0)
          
      count = sum(1 for item in npyList if 'test' in item)
      for i in tqdm(range(int(count/2-1))):
          x_npyName = os.path.join(setpath,hps.data.prepared_test_data_x)
          single_data_x = np.load(x_npyName.split('.')[0]+str(i)+'.npy')

          test_data_x = np.concatenate((test_data_x,single_data_x),axis=0)
          y_npyName = os.path.join(setpath,hps.data.prepared_test_data_y)
          single_data_y = np.load(y_npyName.split('.')[0]+str(i)+'.npy')
          test_data_y = np.concatenate((test_data_y,single_data_y),axis=0)


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
   
  
if __name__=="__main__":
    config_path = 'config.json'
    hps = get_hparams_from_file(config_path)
    train_x,train_y,test_x,test_y = load_data_npz(hps)
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)

