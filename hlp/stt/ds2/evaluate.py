import config
import tensorflow as tf
from model import DS2
from utils import wers,lers,get_index_word,decode_output
from data_preprocess import load_dataset_test

if __name__ == "__main__":
    index_word = get_index_word()
    # 加载模型检查点
    model = DS2(len(index_word)+2)
    # 加载检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=config.configs_checkpoint()['directory'],
        max_to_keep=config.configs_checkpoint()['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    # 评价
    test_data_path = config.configs_test()["data_path"]
    num_examples = config.configs_test()["num_examples"]

    inputs, labels_list = load_dataset_test(test_data_path,num_examples)
    originals = labels_list
    results = []
    y_pred = model(inputs)
    output = tf.keras.backend.ctc_decode(
        y_pred=y_pred,
        input_length=tf.fill([y_pred.shape[0]], y_pred.shape[1]),
        greedy=True
    )
    results_int_list = output[0][0].numpy().tolist()

    # 构建字符集对象
    index_word = get_index_word()
    for i in range(len(results_int_list)):
        str = "".join(decode_output(results_int_list[i], index_word)).strip()
        results.append(str)
    rates_wers, aver_wers = wers(originals, results)
    rates_lers, aver_lers, norm_rates_lers, norm_aver_lers = lers(originals, results)
    print("WER:")
    print("rates", rates_wers)
    print("aver:", aver_wers)
    print("LER:")
    print("rates:", rates_lers)
    print("aver:", aver_lers)
    print("norm_rates:", norm_rates_lers)
    print("norm_aver:", norm_aver_lers)
