from prepocesses import get_spectrograms
import numpy as np

path_ref1 = "./真实1.wav"
path_ref2 = "./预测1.wav"

def evluate(path1,path2):
    mel1,mag1 = get_spectrograms(path1)

    mel2,mag2 = get_spectrograms(path2)

    score = np.sqrt(np.sum((mel1 - mel2)**2))
    print(score)
    return score
evluate(path_ref1,path_ref2)