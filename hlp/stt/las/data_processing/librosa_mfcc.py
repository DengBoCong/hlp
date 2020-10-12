# -*- coding: utf-8 -*-

#import wav
import librosa
'''
path='.\\wav\\BAC009S0002W0123.wav'
y,sr = librosa.load(path=path)
mfccs = librosa.feature.mfcc(y=y,n_mfcc=20)
print(mfccs.shape)
#mfccs 
#(20, 259) 259 = 时间步（wav的份数） n_mfcc=20 每个时间步的特征数

path='.\\wav\\BAC009S0002W0122.wav'
y,sr = librosa.load(path=path)
mfccs = librosa.feature.mfcc(y=y,n_mfcc=20)
print(mfccs.shape)
'''
def mfcc_extract(path):
    y,sr = librosa.load(path=path)
    mfccs = librosa.feature.mfcc(y=y,n_mfcc=20)
    return mfccs
    