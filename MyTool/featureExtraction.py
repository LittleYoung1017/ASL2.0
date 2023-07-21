import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from librosa import display
import librosa
from spafe.features.mfcc import mfcc
from spafe.features.lfcc import lfcc
from spafe.features.cqcc import cqcc
from spafe.features.mfcc import mel_spectrogram
from spafe.utils.preprocessing import SlidingWindow
from spafe.utils.vis import show_features
# import plotAllFeatures as myplt
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
    
if __name__ == "__main__":
    file_path = 'd:/YRX/tamper_detection/My_Splicing_detection/concat-data-8k/'
    file = os.listdir(file_path)
    for f in file:
        f_path = os.path.join(file_path,f)
        wav,sr = librosa.load(f_path,sr=8000)
        # wav_rec = Mywavelet.wavelet(data=wav,sr=8000)

        # mfccs = featureExtracting(wav,sr,'mfcc')
        # mfccs_rec = featureExtracting(wav_rec,sr,'mfcc')
        # base_noise_mfcc = np.absolute(mfccs - mfccs_rec)


        # lfccs = featureExtracting(wav,sr,'lfcc')
        # lfccs_rec = featureExtracting(wav_rec,sr,'lfcc')
        # base_noise_lfcc = np.absolute(lfccs - lfccs_rec)

        # cqccs = featureExtracting(wav,sr,'cqcc')
        # cqccs_rec = featureExtracting(wav_rec,sr,'cqcc')
        # base_noise_cqcc = np.absolute(cqccs - cqccs_rec)
        # myplt.AllFeatures(wav,sr)
