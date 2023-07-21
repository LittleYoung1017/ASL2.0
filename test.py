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
from model.LCNN import LCNN
from model.CNNLSTM import CNNLSTM
import utils
import librosa
from MyTool.featureExtraction import featureExtracting
from spafe.utils.preprocessing import SlidingWindow
import math
def feature_block(H,w):
    feature_blocks = []
    y = []
    Ns = H.shape[1] #特征长度
    D = H.shape[0]  #窗的高度
    Nr = math.floor(Ns / w) #相关系数向量长度
    if Ns % w != 0:
        Nr+=1
    r = np.zeros((1, Nr))
    

    for i in range(Nr): #沿着特征滑动
        w_start = i *  w
        w_end = w_start + w
        # print("start point and end point:",w_start,w_end)
        if (w_start + w ) <= Ns :
            H1 = H[:,w_start:w_start + w]
        else:
            arr_0 = np.zeros((D,w-(Ns-w_start)))
            H1 = np.concatenate((H[:,w_start:],arr_0),axis=1)
        print("H1.shape:",H1.shape)
        
        feature_blocks.append(H1)
    # print("y:",y)
    return np.array(feature_blocks)

def test(X_data, model,files,cut_points ):
    # size = len(dataloader.dataset)
    # print("size:",size)
    # num_batches = len(dataloader)
    # print("num_batch:",num_batches)
    model.eval()
    count=0
    with torch.no_grad():
        for X in X_data:
            X = torch.tensor(X).to(device)
            # y = y.reshape(-1,1)
            X = X.type(torch.cuda.FloatTensor)
            pred = model(X)
            print("files:",files[count])
            if cut_points[count][0]!='None':
                print(f"splicing_points:{cut_points[count][0]/8000}s, audio_length:{cut_points[count][1]}s.")
            else:
                print(f"splicing_points:{cut_points[count][0]}, audio_length:{cut_points[count][1]}s.")

            print(pred.argmax(1))
            print("--------------------------")
            count+=1

            # correct += (pred.argmax(1) == y).type(torch.float).sum().item() #tensor.argmax(dim=-1)每一行最大值下标
    # correct /= size
    # print(f"Test: \n Accuracy: {(100*correct):>0.001f}% \n")

if __name__ == '__main__':
    


    config_path = 'config.json'
    hps = utils.get_hparams_from_file(config_path)
    test_dir = hps.test.test_data
    
    cut_points = []
    data = []
    batch_size = hps.train.batch_size
    # test_data = utils.MyDataset(X_data)
    # train_dataloader = DataLoader(data, batch_size=200, shuffle=False, drop_last=True, num_workers=0)
    files = os.listdir(test_dir)
    for i in range(len(files)):
        wav,sr = librosa.load(os.path.join(test_dir,files[i]),sr=8000)
        if len(files[i].split('.')[0].split('_'))==2:
            cut_point = int(files[i].split('.')[0].split('_')[1])
        elif len(files[i].split('.')[0].split('_'))==3:
            cut_point = int(files[i].split('.')[0].split('_')[2])
        else:
            cut_point = 'None'
        cut_points.append((cut_point,len(wav)/sr))

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
        print(data_feature.shape)
        w = 300
        X = feature_block(data_feature,w)
        
        # X_data = np.array(X)
        # X_data = X_data[:,np.newaxis,:,:]
        data.append(X)
    # print("拼接时间点:",int(cut_point)/sr)
    device = (
        hps.train.device
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using {device} device")

    model_name = hps.model.model_name
    if model_name == 'LCNN':
        model = LCNN().to(device)
    elif model_name == 'CNN':
        model = CNN().to(device)
    elif model_name == 'CNNLSTM':
        model = CNNLSTM(32,128,2,3).to(device)
    else:
        raise Exception(f"Error: {model_name}")
    print(model)

    model.load_state_dict(torch.load(hps.test.checkpoint_name))



    # test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=False)


    test(data, model,files,cut_points)
        
    print("Test done!")
        
