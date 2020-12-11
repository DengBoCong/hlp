import tensorflow as tf
from hlp.stt.utils.features import wav_to_feature


def get_input_and_length(audio_path_list, audio_feature_type, maxlen):
    audio_feature_list = []
    input_length_list = []
    for audio_path in audio_path_list:
        audio_feature = wav_to_feature(audio_path, audio_feature_type)
        audio_feature_list.append(audio_feature)
        input_length_list.append([audio_feature.shape[0]])

    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        audio_feature_list,
        maxlen=maxlen,
        dtype='float32',
        padding='post'
    )
    input_length = tf.convert_to_tensor(input_length_list)

    return input_tensor, input_length


# 获取最长的音频length(timesteps)
def max_audio_length(audio_path_list, audio_feature_type):
    return max(wav_to_feature(audio_path, audio_feature_type).shape[0] for audio_path in audio_path_list)


if __name__ == "__main__":
    pass
