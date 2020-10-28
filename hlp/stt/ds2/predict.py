import tensorflow as tf
from audio_process import record, wav_to_mfcc
from model import decode_output, get_ds2_model
from utils import get_config, get_index_word
import numpy as np

if __name__ == "__main__":
    configs = get_config()
    # 录音
    record()
    index_word = get_index_word()
    # 加载模型检查点
    model=get_ds2_model()
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    audio_path = "./number/dev/1_nicolas_0.wav"
    # audio_path = configs["record"]["record_path"]
    n_mfcc = configs["other"]["n_mfcc"]
    x_test = wav_to_mfcc(audio_path, n_mfcc)
    x_test_input = tf.keras.preprocessing.sequence.pad_sequences(
            [x_test],
            maxlen=configs["preprocess"]["max_inputs_len"],
            padding='post',
            dtype='float32',
            truncating='post'
            )
    y_test_pred = model(x_test_input)
    output = tf.keras.backend.ctc_decode(
        y_pred=y_test_pred,
        input_length=tf.constant([y_test_pred.shape[1]]),
        greedy=True
    )
    str = decode_output(output[0][0].numpy()[0], index_word)
    print("Output:" + str)
