import os


def preprocess_lj_speech_raw_data(transcripts_file: str, audio_dir: str, save_file: str):
    """
    用于处理LJSpeech数据集的方法，将数据整理为<音频地址, 句子>的
    形式，这样方便后续进行分批读取。
    Args:
        transcripts_file: 元数据CSV文件路径
        audio_dir: 音频目录路径
        save_file: 保存处理之后的数据路径
    Returns:
    """
    audios_list = os.listdir(audio_dir)

    count = 0
    with open(transcripts_file, 'r', encoding='utf-8') as raw_file, \
            open(save_file, 'w', encoding='utf-8') as save_file:
        for line in raw_file:
            line = line.strip('\n').replace('/', '')
            pair = line.split('|')
            audio_file = audio_dir + pair[0] + '.wav'
            if audios_list.find(audio_file) == -1:
                print('音频数据不完整，请检查后重试!')
            save_file.write(audio_dir + pair[0] + '.wav' + '\t' + pair[1])
            count += 1

    print("数据处理完毕，共计{}条语音数据".format(count))
