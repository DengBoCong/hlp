import tensorflow as tf
from load_dataset import load_data
from generator import data_generator
from metric import lers, wers
from model import decode_output, get_ds2_model
from utils import get_config, get_index_word
from math import ceil

if __name__ == "__main__":
    configs = get_config()
    
    # 加载模型检查点
    model = get_ds2_model()
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    test_data_path = configs["test"]["data_path"]
    num_examples = configs["test"]["num_examples"]
    dataset_name = configs["preprocess"]["dataset_name"]
    batch_size = configs["test"]["batch_size"]

    # 加载测试集数据生成器
    test_data = load_data(dataset_name, test_data_path, "test", num_examples)
    batchs = ceil(len(test_data[0]) / batch_size)
    test_data_generator = data_generator(test_data, "test", batchs, batch_size)

    aver_wers = 0
    aver_lers = 0
    aver_norm_lers = 0
    
    # 构建字符集对象
    index_word = get_index_word()

    for batch, (input_tensor, labels_list) in zip(range(1, batchs+1), test_data_generator):
        originals = labels_list
        results = []
        y_pred = model(input_tensor)
        output = tf.keras.backend.ctc_decode(
            y_pred=y_pred,
            input_length=tf.fill([y_pred.shape[0]], y_pred.shape[1]),
            greedy=True
        )
        results_int_list = output[0][0].numpy().tolist()

        # 解码
        for i in range(len(results_int_list)):
            str = decode_output(results_int_list[i], index_word).strip()
            results.append(str)
        
        # 通过wer、ler指标评价模型
        rates_wer, aver_wer = wers(originals, results)
        rates_ler, aver_ler, norm_rates_ler, norm_aver_ler = lers(originals, results)
        aver_wers += aver_wer
        aver_lers += aver_ler
        aver_norm_lers += norm_aver_ler
    
    print("WER:")
    print("aver:", aver_wers/batchs)
    print("LER:")
    print("aver:", aver_lers/batchs)
    print("aver_norm:", aver_norm_lers/batchs)
