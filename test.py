import os 
import numpy as np
import torch
from torch import nn
from torch.utils import data
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision.transforms import ToTensor
from torchvision import datasets
from model import CNN
from model.LCNN2 import LCNN
from model.CNNLSTM2 import CNNLSTM2
import utils
import librosa
from utils import featureExtracting
from spafe.utils.preprocessing import SlidingWindow
import math

def test(X_data, y_data, model,files ):
    # size = len(dataloader.dataset)
    # print("size:",size)
    # num_batches = len(dataloader)
    # print("num_batch:",num_batches)
    model.eval()
    test_num = len(X_data)
    correct = 0
    with torch.no_grad():

        for i in range(test_num):
            X = np.array(X_data[i])
            X = X[np.newaxis,np.newaxis,:,:]
            X = torch.tensor(X).to(device)
            # y = y.reshape(-1,1)
            X = X.type(torch.cuda.FloatTensor)
            pred = model(X)
            result = pred.argmax(1)
            print("files:",files[i],"predict:",pred.argmax(1))
            print("--------------------------")
            if 'original' in files[i] and result == 0:
                correct+=1
            elif 'tampered' in files[i] and result == 1:
                correct+=1

    accuracy = correct / test_num
    print(f"Test: \n Accuracy: {(100*accuracy):>0.001f}% \n")

if __name__ == '__main__':
    


    config_path = 'config.json'
    hps = utils.get_hparams_from_file(config_path)
    test_dirs = hps.test.test_data

    print('data loading ...')

    for test_dir in test_dirs:
        files = os.listdir(test_dir)


        X_data = []
        y_data = []
        batch_size = hps.train.batch_size

        for i in range(len(files)):
            wav,sr = librosa.load(os.path.join(test_dir,files[i]),sr=8000)
            
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
            if 'original' in files[i]:
                y_data.append(0)
            elif 'tampered' in files[i]:
                y_data.append(1)
                
    device = (
        hps.train.device
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print('data loading complete...')
    print(f"Using {device} device")

    model_name = hps.model.model_name
    if model_name == 'LCNN':
        model = LCNN().to(device)
    elif model_name == 'CNN':
        model = CNN().to(device)
    elif model_name == 'CNNLSTM':
        model = CNNLSTM(32,32,2,3).to(device)
    else:
        raise Exception(f"Error: {model_name}")
    print(model)

    model.load_state_dict(torch.load(hps.test.checkpoint_name),strict=False)

    test(X_data,y_data,model,files)
        
    print("Test done!")
        
