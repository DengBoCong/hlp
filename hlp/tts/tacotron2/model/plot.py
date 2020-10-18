import matplotlib.pyplot as plt
import numpy as np


def plot_spectrogram_to_numpy(spectrogram):
    fig, ax = plt.subplots(figsize=(12, 3))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower",
                   interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel("Frames")
    plt.ylabel("Channels")
    plt.tight_layout()

    fig.canvas.draw()
    data = save_figure_to_numpy(fig)
    plt.close()
    return data


def save_figure_to_numpy(fig):
    # save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def plot_mel(mel_gts):
    # x, y = get_spectrograms('1.wav')
    # print("x:",x.shape)
    # print("mel", x)
    # wav = melspectrogram2wav(x)
    # wave.write('4.wav', rate=sr, data=wav)
    # for i in range((mel_gts.shape)[0]):
    for i in range((mel_gts.shape)[0]):
        plt.figure()
        plt.imshow(plot_spectrogram_to_numpy(mel_gts[i]))
        plt.show()
