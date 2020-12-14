import tensorflow as tf
from hlp.stt.utils.features import wav_to_feature


def get_input_and_length(audio_path_list, audio_feature_type, maxlen):
    """ 获得语音文件的特征和长度

    :param audio_path_list: 语音文件列表
    :param audio_feature_type: 语音特征类型
    :param maxlen: 最大补齐长度
    :return: 补齐后的语音特征数组，每个语音文件的帧数
    """
    audio_feature_list = []
    input_length_list = []
    for audio_path in audio_path_list:
        audio_feature = wav_to_feature(audio_path, audio_feature_type)
        audio_feature_list.append(audio_feature)
        input_length_list.append([audio_feature.shape[0]])

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(audio_feature_list,
                                                                 maxlen=maxlen,
                                                                 dtype='float32',
                                                                 padding='post'
                                                                 )
    input_length = tf.convert_to_tensor(input_length_list)

    return input_tensor, input_length


def max_audio_length(audio_path_list, audio_feature_type):
    """ 获得语音特征帧最大长度

    注意：这个方法会读取所有语音文件，并提取特征.

    :param audio_path_list: 语音文件列表
    :param audio_feature_type: 语音特征类型
    :return: 最大帧数
    """
    return max(wav_to_feature(audio_path, audio_feature_type).shape[0] for audio_path in audio_path_list)


if __name__ == "__main__":
    pass
