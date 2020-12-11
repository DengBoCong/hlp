import numpy as np

from hlp.stt.utils.audio_process import get_input_and_length
from stt.utils.text_process import get_label_and_length


# train数据生成器
def train_generator(data, batches, batch_size, audio_feature_type,
                    max_input_length, max_label_length):
    """

    :param data: 语音文件列表，向量化转写列表
    :param batches:
    :param batch_size:
    :param audio_feature_type:
    :param max_input_length:
    :param max_label_length:
    :return:
    """
    audio_path_list, text_int_sequences_list = data

    # generator只能进行一次生成，故需要while True来进行多个epoch的数据生成
    while True:
        # 每epoch将所有数据进行一次shuffle
        order = np.random.choice(len(audio_path_list), len(audio_path_list), replace=False)
        audio_path_list = [audio_path_list[i] for i in order]
        text_int_sequences_list = [text_int_sequences_list[i] for i in order]

        for idx in range(batches):
            batch_input_tensor, batch_input_length = get_input_and_length(
                audio_path_list[idx * batch_size: (idx + 1) * batch_size],
                audio_feature_type,
                max_input_length
            )
            batch_label_tensor, batch_label_length = get_label_and_length(
                text_int_sequences_list[idx * batch_size: (idx + 1) * batch_size],
                max_label_length
            )

            yield batch_input_tensor, batch_label_tensor, batch_input_length, batch_label_length


# 测试数据生成器
def test_generator(data, batches, batch_size, audio_feature_type, max_input_length):
    audio_data_path_list, text_list = data

    while True:
        for idx in range(batches):
            batch_input_tensor, batch_input_length = get_input_and_length(
                audio_data_path_list[idx * batch_size: (idx + 1) * batch_size],
                audio_feature_type,
                max_input_length
            )
            batch_text_list = text_list[idx * batch_size: (idx + 1) * batch_size]

            # 测试集只需要文本串list
            yield batch_input_tensor, batch_input_length, batch_text_list


if __name__ == '__main__':
    pass
