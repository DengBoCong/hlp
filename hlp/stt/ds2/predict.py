import tensorflow as tf
from model import get_ds2_model
from utils import get_index_word, get_config
from model import decode_output
from audio_process import record, wav_to_mfcc


if __name__ == "__main__":
    configs = get_config()
    # 录音
    record(file_path = configs["record"]["record_path"])
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
    #audio_path = configs["record"]["record_path"]
    x_test = wav_to_mfcc(audio_path)
    x_test_input = tf.keras.preprocessing.sequence.pad_sequences(
            [x_test],
            maxlen=configs["preprocess"]["max_inputs_len"],
            padding='post',
            dtype='float32'
            )
    y_test_pred = model(x_test_input)
    output = tf.keras.backend.ctc_decode(
        y_pred=y_test_pred,
        input_length=tf.constant([y_test_pred.shape[1]]),
        greedy=True
    )
    str = decode_output(output[0][0].numpy()[0], index_word)
    print(output[0][0].numpy()[0])
    print(str)
