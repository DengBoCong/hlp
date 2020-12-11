
import tensorflow as tf
from hlp.stt.ds2.model import DS2, decode_output
from hlp.stt.ds2.util import get_config, get_dataset_information, compute_ctc_input_length
from hlp.stt.utils.features import wav_to_feature
from hlp.stt.utils.record import record


if __name__ == "__main__":
    configs = get_config()
    dataset_information = get_dataset_information(configs["preprocess"]["dataset_information_path"])

    # 获取模型配置，加载模型
    conv_layers = configs["model"]["conv_layers"]
    filters = configs["model"]["conv_filters"]
    kernel_size = configs["model"]["conv_kernel_size"]
    strides = configs["model"]["conv_strides"]
    bi_gru_layers = configs["model"]["bi_gru_layers"]
    gru_units = configs["model"]["gru_units"]
    fc_units = configs["model"]["fc_units"]
    dense_units = dataset_information["vocab_size"] + 2

    model = DS2(conv_layers, filters, kernel_size, strides, bi_gru_layers, gru_units, fc_units, dense_units)
    optimizer = tf.keras.optimizers.Adam()

    # 加载模型检查点
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    # 加载预测、解码所需的参数
    record_path = "./record.wav"
    audio_feature_type = configs["other"]["audio_feature_type"]
    index_word = dataset_information["index_word"]
    mode = configs["preprocess"]["text_process_mode"]
    max_input_length = dataset_information["max_input_length"]

    while True:
        try:
            record_duration = int(input("请设定录音时长(秒, <=0则结束):"))
        except BaseException:
            print("录音时长只能为int数值")
        else:
            if record_duration <= 0:
                break
            # 录音
            record(record_path, record_duration)
            # record_path = "./1088-134315-0000.flac"

            # 加载录音数据并预测
            x_test = wav_to_feature(record_path, audio_feature_type)
            x_test_input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
                [x_test],
                padding='post',
                maxlen=max_input_length,
                dtype='float32'
            )
            y_test_pred = model(x_test_input_tensor)
            ctc_input_length = compute_ctc_input_length(x_test_input_tensor.shape[1], y_test_pred.shape[1],
                                                        tf.convert_to_tensor([[len(x_test)]]))

            output = tf.keras.backend.ctc_decode(
                y_pred=y_test_pred,
                input_length=tf.reshape(ctc_input_length, [ctc_input_length.shape[0]]),
                greedy=True
            )

            # 解码
            str = decode_output(output[0][0].numpy()[0], index_word, mode)
            print("Output:" + str)
