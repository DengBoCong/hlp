import numpy as np
import tensorflow as tf

from prepocesses import map_to_text, dataset_seq, dataset_mel

# train数据生成器
def generator(wav_name_list, batch_size, csv_dir, tokenizer, wave_path, config, mode):
    # generator只能进行一次生成，故需要while True来进行多个epoch的数据生成
    if mode == 'train':
        while True:
            # 每epoch将所有数据进行一次shuffle
            order = np.random.choice(len(wav_name_list), len(wav_name_list), replace=False)
            audio_data_path_list = [wav_name_list[i] for i in order]
            batchs = len(wav_name_list)//batch_size
            for idx in range(batchs):
                #逐步取音频名
                wav_name_list2 = audio_data_path_list[idx * batch_size: (idx + 1) * batch_size]
                #根据文本取数据
                sentence_list = map_to_text(csv_dir, wav_name_list2)
                input_ids = dataset_seq(sentence_list, tokenizer, config)
                input_ids = tf.convert_to_tensor(input_ids)

                #取音频数据
                wav_tensor, input_length = dataset_mel(
                    wave_path, config.max_input_length, wav_name_list2, config
                )

                yield input_ids, wav_tensor, input_length
    else:
        while True:
            batchs = len(wav_name_list) // batch_size
            for idx in range(batchs):
                # 逐步取音频名
                wav_name_list2 = wav_name_list[idx * batch_size: (idx + 1) * batch_size]
                # 根据文本取数据
                sentence_list = map_to_text(csv_dir, wav_name_list2)
                input_ids = dataset_seq(sentence_list, tokenizer, config)
                input_ids = tf.convert_to_tensor(input_ids)

                # 取音频数据
                wav_tensor, input_length = dataset_mel(
                    wave_path, config.max_input_length, wav_name_list2, config
                )

                yield input_ids, wav_tensor, input_length

if __name__ == '__main__':
    pass