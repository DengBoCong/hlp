import tensorflow as tf
import os
import librosa
ch=[]

def Dataset_wave(path):
    x = tf.constant(0, shape=(1, 500, 80))
    i = 0
    dirs = os.listdir(path)
    for file in dirs:
        #file=file
        y, sr = librosa.load(path+file, sr=None)
        melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=80)
        logmelspec = librosa.power_to_db(melspec)
        logmelspec = tf.keras.preprocessing.sequence.pad_sequences(logmelspec, maxlen=500, padding='post')
        logmelspec = tf.transpose(logmelspec, [1, 0])
        logmelspec = tf.expand_dims(logmelspec, 0)
        if i == 0:
            x = logmelspec
            i = i + 1
            continue
        x = tf.concat([x, logmelspec], 0)
    return x

