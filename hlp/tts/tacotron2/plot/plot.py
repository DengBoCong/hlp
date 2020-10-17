import matplotlib.pyplot as plt
import librosa
import librosa.display
def plot_mel(mel_gts):
    for i in range((mel_gts.shape)[0]):
        x = mel_gts[i]
        x = x.numpy()
        plt.figure()
        librosa.display.specshow(x, sr=22050, x_axis='time', y_axis='mel')
        plt.show()
