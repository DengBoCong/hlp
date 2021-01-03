import tensorflow as tf

from hlp.stt.las.config import config
from hlp.stt.las.model import plas, las
from hlp.stt.utils.features import wav_to_feature
from hlp.stt.utils.record import record


def recognize(wav_path):
    model_type = config.model_type
    embedding_dim = config.embedding_dim
    units = config.units
    cnn1_filters = config.cnn1_filters
    cnn1_kernel_size = config.cnn1_kernel_size
    cnn2_filters = config.cnn2_filters
    cnn2_kernel_size = config.cnn2_kernel_size
    max_pool_strides = config.max_pool_strides
    max_pool_size = config.max_pool_size
    d = config.d
    w = config.w
    emb_dim = config.emb_dim
    dec_units = config.dec_units
    BATCH_SIZE = 1  # 只支持BATCH_SIZE为1的评估
    dataset_information = config.get_dataset_info()
    vocab_tar_size = dataset_information["vocab_tar_size"]
    word_index = dataset_information["word_index"]
    max_label_length = dataset_information["max_label_length"]
    index_word = dataset_information["index_word"]
    optimizer = tf.keras.optimizers.Adam()

    # 选择模型类型
    if model_type == "las":
        model = plas.PLAS(vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    elif model_type == "las_d_w":
        model = las.LAS(vocab_tar_size,
                        cnn1_filters,
                        cnn1_kernel_size,
                        cnn2_filters,
                        cnn2_kernel_size,
                        max_pool_strides,
                        max_pool_size,
                        d,
                        w,
                        emb_dim,
                        dec_units,
                        BATCH_SIZE)
    # 检查点
    checkpoint_dir = config.checkpoint_dir
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=checkpoint_dir,
        max_to_keep=config.max_to_keep
    )
    print("恢复检查点目录 （checkpoint_dir） 中最新的检查点......")
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    enc_hidden = model.initialize_hidden_state()
    wav_tensor = wav_to_feature(wav_path, "mfcc")
    wav_tensor = tf.expand_dims(wav_tensor * BATCH_SIZE, 0)
    dec_input = tf.expand_dims([word_index['<start>']] * BATCH_SIZE, 1)
    result = ''  # 识别结果字符串

    for t in range(max_label_length):  # 逐步解码或预测
        predictions, _ = model(wav_tensor, enc_hidden, dec_input)
        predicted_ids = tf.argmax(predictions, 1).numpy()  # 贪婪解码，取最大
        idx = str(predicted_ids[0])
        if index_word[idx] == '<end>':
            break
        else:
            result += index_word[idx]  # 目标句子
        # 预测的 ID 被输送回模型
        dec_input = tf.expand_dims(predicted_ids, 1)
    print('Speech recognition results: {}'.format(result))


if __name__ == "__main__":
    record("record.wav", 2)
    file_path = ".\\record.wav"
    recognize(file_path)
