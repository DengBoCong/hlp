# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:37:32 2020

@author: 九童
使用训练集进行模型评估
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import tensorflow as tf
from hlp.stt.las import train
from hlp.stt.las.model import las
from hlp.stt.utils.metric import lers
from hlp.stt.las.config import config
from hlp.stt.las.data_processing import load_dataset
from hlp.stt.las.data_processing.generator import data_generator


if __name__ == "__main__":

    # 用测试集wav文件语音识别出中文 
    # 测试集wav文件
    wav_path = config.test_wav_path

    # 测试集文本标签
    label_path = config.test_label_path
    
    # 尝试实验不同大小的数据集
    test_num = config.test_num
    
    # 每一步mfcc所取得特征数
    n_mfcc = config.n_mfcc
    
    embedding_dim = config.embedding_dim
    units = config.units
    batch_size = config.test_batch_size
    dataset_name = config.dataset_name
    audio_feature_type = config.audio_feature_type
    num_examples = config.test_num
    
    print("获取训练语料信息......")
    dataset_information = config.get_dataset_information()
    '''
    steps_per_epoch,test_targ_tokenizer,test_max_length_targ,test_max_length_inp,dataset = train.create_dataset(test_path,
                                                    test_path_to_file,test_num, n_mfcc,BATCH_SIZE)    
    '''
    test_vocab_tar_size = dataset_information["vocab_tar_size"]
    optimizer = tf.keras.optimizers.Adam()
    
    model = las.las_model(test_vocab_tar_size, embedding_dim, units, batch_size)
    
    # 检查点
    checkpoint_dir = config.checkpoint_dir
    checkpoint_prefix = os.path.join(checkpoint_dir, config.checkpoint_prefix)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    results = []
    labels_list = []
    
    # 加载测试集数据生成器
    test_data = load_dataset.load_data(dataset_name, wav_path, label_path, "test", num_examples) 
    batchs = len(test_data[0]) // batch_size
    print("构建数据生成器......")
    test_data_generator = data_generator(
        test_data,
        "test",
        batchs,
        batch_size,
        audio_feature_type,
        dataset_information["max_input_length"],
        dataset_information["max_label_length"]
    )

    
    word_index = dataset_information["word_index"]
    index_word = dataset_information["index_word"]
    max_label_length = dataset_information["max_label_length"]

    for batch, (inp, targ) in zip(range(1, batchs + 1), test_data_generator):
        hidden = tf.zeros((batch_size, units))
        dec_input = tf.expand_dims([word_index['<start>']] * batch_size, 1)
        result = ''  # 识别结果字符串
        
        for t in range(max_label_length):  # 逐步解码或预测
            predictions, dec_hidden = model(inp, hidden, dec_input)
            predicted_ids = tf.argmax(predictions, 1).numpy()  # 贪婪解码，取最大 
            idx = str(predicted_ids[0])            
            if index_word[idx] == '<end>':
                break
            else:
                result += index_word[idx]  # 目标句子
            # 预测的 ID 被输送回模型            
            dec_input = tf.expand_dims(predicted_ids, 1)

                
        results.append(result)
        labels_list.append(targ[0])
    print('results ==== {}'.format(results))
    print('labels_list ==== {}'.format(labels_list))
    rates_lers, aver_lers, norm_rates_lers, norm_aver_lers = lers(labels_list, results)


    print("字母错误率:")
    print("每条语音字母错误数:", rates_lers)
    print("所有语音平均字母错误数:", aver_lers)
    print("每条语音字母错误率，错误字母数/标签字母数:", norm_rates_lers)
    print("所有语音平均字母错误率:", norm_aver_lers)