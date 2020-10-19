import config
import tensorflow as tf
from model import DS2
from utils import wav_to_mfcc,decode_output,record,get_index_word
import json


if __name__ == "__main__":
    # 录音
    record(file_path=config.configs_record()["record_path"])
    index_word = get_index_word()
    # 加载模型检查点
    model=DS2(len(index_word)+2)
    # 加载检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=config.configs_checkpoint()['directory'],
        max_to_keep=config.configs_checkpoint()['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    audio_path = config.configs_record()["record_path"]
    x_test = wav_to_mfcc(audio_path)
    x_test_input = tf.expand_dims(x_test, axis=0)
    print(x_test_input)
    y_test_pred = model(x_test_input)
    print(y_test_pred)
    output = tf.keras.backend.ctc_decode(
        y_pred=y_test_pred,
        input_length=tf.constant([y_test_pred.shape[1]]),
        greedy=True
    )
    str = decode_output(output[0][0].numpy()[0], index_word)
    print(output[0][0].numpy()[0])
    print(str)
