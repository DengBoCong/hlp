# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 15:37:32 2020

@author: 九童
使用训练集进行模型评估
"""
import os
import tensorflow as tf
from hlp.stt.las import train
from hlp.stt.las.metric import lers
from hlp.stt.las.model import las


if __name__ == "__main__":

    # 用测试集wav文件语音识别出中文 
    # 测试集wav文件
    test_path = ".\\data\\wav_test"
    
    # 测试集文本标签
    test_path_to_file = ".\\data\\data_test.txt"
    
    # 尝试实验不同大小的数据集
    test_num = 80
    
    # 每一步mfcc所取得特征数
    n_mfcc = 20
    
    embedding_dim = 256
    units = 512
    BATCH_SIZE = 64
    steps_per_epoch,test_targ_tokenizer,test_max_length_targ,test_max_length_inp,dataset = train.create_dataset(test_path,
                                                    test_path_to_file,test_num, n_mfcc,1)    
    test_vocab_tar_size = len(test_targ_tokenizer.word_index) + 1  # 含填充的0
    optimizer = tf.keras.optimizers.Adam()
    
    #vocab_tar_size应从配置文件中得到，此处暂用test_vocab_tar_size代替试试
    #decoder = decoder.Decoder(test_vocab_tar_size, embedding_dim, units, BATCH_SIZE)
    model = las.las_model(test_vocab_tar_size, n_mfcc, embedding_dim, units, BATCH_SIZE)
    
    # 检查点
    checkpoint_dir = './checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    
    # 恢复检查点目录 （checkpoint_dir） 中最新的检查点
    checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
    
    results = []
    labels_list = []
    
    for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
        hidden = tf.zeros((1, units))
        enc_hidden = model.initialize_hidden_state()
        dec_input = tf.expand_dims([test_targ_tokenizer.word_index['<start>']] * BATCH_SIZE, 1)

        result = ''  # 识别结果字符串
        
        for t in range(test_max_length_targ):  # 逐步解码或预测
            predictions, dec_hidden = model(inp, enc_hidden, dec_input)
            predicted_ids = []
            for prediction in predictions:
                predicted_id = tf.argmax(prediction).numpy()  # 贪婪解码，取最大
                predicted_ids.append(predicted_id)
            result += test_targ_tokenizer.index_word[predicted_id]  # 目标句子
            if test_targ_tokenizer.index_word[predicted_id] == '<end>':
                break
            # 预测的 ID 被输送回模型            
            dec_input = tf.expand_dims(predicted_ids, 1)
        tar_result = ''
        
        for target_id in targ[0]:
            target_id = target_id.numpy()
            if target_id == 0 or target_id == 1:
                tar_result += ''  # 目标句子
            else:
                tar_result += test_targ_tokenizer.index_word[target_id]  # 目标句子
                
        results.append(result)
        labels_list.append(tar_result)

    rates_lers, aver_lers, norm_rates_lers, norm_aver_lers = lers(labels_list, results)


    print("LER:")
    print("rates:", rates_lers)
    print("aver:", aver_lers)
    print("norm_rates:", norm_rates_lers)
    print("norm_aver:", norm_aver_lers)