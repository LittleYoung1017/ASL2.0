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
def feature_block(H,w):  #最后测试时用与长音频分段
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

def test(X_data, y_data, model,files ):
    # size = len(dataloader.dataset)
    # print("size:",size)
    # num_batches = len(dataloader)
    # print("num_batch:",num_batches)
    model.eval()
    count=0
    correct = 0
    with torch.no_grad():
        for X in X_data:
            X = np.array(X)
            X = X[np.newaxis,:,:]
            X = torch.tensor(X).to(device)
            # y = y.reshape(-1,1)
            X = X.type(torch.cuda.FloatTensor)
            pred = model(X)
            result = pred.argmax(1)
            print("files:",files[count],"predict:",pred.argmax(1))
            print("--------------------------")
            if 'original' in files[count] and result == 0:
                correct+=1
            elif 'tampered' in files[count] and result == 1:
                correct+=1
            count+=1
    accuracy = correct / count
    #         correct += (pred.argmax(1) == y).type(torch.float).sum().item() #tensor.argmax(dim=-1)每一行最大值下标
    # correct /= size
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
        # test_data = utils.MyDataset(X_data)
        # train_dataloader = DataLoader(data, batch_size=200, shuffle=False, drop_last=True, num_workers=0)

        for i in range(len(files)):
            wav,sr = librosa.load(os.path.join(test_dir,files[i]),sr=8000)


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
            # print(data_feature.shape)
            # w = 300
            # X = feature_block(data_feature,w)

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
        model = CNNLSTM(32,64,2,3).to(device)
    else:
        raise Exception(f"Error: {model_name}")
    print(model)

    model.load_state_dict(torch.load(hps.test.checkpoint_name))
    
    
    #后续添加dataloader方便多文件测试
    # test_dataloader = DataLoader(test_data,batch_size=batch_size,shuffle=False)


    test(X_data,y_data,model,files)
        
    print("Test done!")
        
