import tensorflow as tf
import config
from model import DS2
from utils import int_to_text_sequence, record, wav_to_mfcc, get_index_and_char_map

if __name__=="__main__":
    #录音
    record(file_path = config.configs_record()["record_path"])

    #加载模型检查点
    model=DS2()
    #加载检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
       checkpoint,
       directory=config.configs_checkpoint()['directory'],
       max_to_keep=config.configs_checkpoint()['max_to_keep']
       )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    audio_path = config.configs_record()["record_path"]
    x_test = wav_to_mfcc(config.configs_other()["n_mfcc"],audio_path)
    x_test_input=tf.expand_dims(x_test,axis=0)
    y_test_pred=model(x_test_input)
    output=tf.keras.backend.ctc_decode(
        y_pred=y_test_pred,
        input_length=tf.constant([y_test_pred.shape[1]]),
        greedy=True
        )
    index_map = get_index_and_char_map()[0]
    str="".join(int_to_text_sequence(output[0][0].numpy()[0],index_map))
    print(str)
