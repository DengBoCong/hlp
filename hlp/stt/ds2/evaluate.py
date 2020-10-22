import tensorflow as tf
from model import init_ds2
from utils import wers,lers,get_index_word,decode_output,get_config,get_config_model
from data_preprocess import load_dataset_test

if __name__ == "__main__":
    configs = get_config()
    # 加载模型检查点
    model = init_ds2()
    # 加载检查点
    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        directory=configs["checkpoint"]['directory'],
        max_to_keep=configs["checkpoint"]['max_to_keep']
    )
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)

    # 评价
    test_data_path = configs["test"]["data_path"]
    num_examples = configs["test"]["num_examples"]

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
        str = decode_output(results_int_list[i], index_word).strip()
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
