import tensorflow as tf
from model import get_ds2_model, decode_output
from utils import get_config, get_index_word
from data_preprocess import load_data
from metric import wers, lers

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
    #加载测试集数据并通过模型计算出结果序列
    inputs, labels_list = load_data(test_data_path, "test", num_examples)
    originals = labels_list
    results = []
    y_pred = model(inputs)
    output = tf.keras.backend.ctc_decode(
        y_pred=y_pred,
        input_length=tf.fill([y_pred.shape[0]], y_pred.shape[1]),
        greedy=True
    )
    results_int_list = output[0][0].numpy().tolist()

    # 构建字符集对象,并解码出预测的结果list
    index_word = get_index_word()
    for i in range(len(results_int_list)):
        str = decode_output(results_int_list[i], index_word).strip()
        results.append(str)
    
    # 通过wer、ler指标评价模型
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
