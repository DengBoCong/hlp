# hlp
基于深度学习的对话系统、语音识别、机器翻译和语音合成等。
# 目录结构
- hlp: 顶层包目录
   - mt: 机器翻译包目录
   - stt: 语音识别包目录
   - tts: 语音合成包目录
   - chat: 对话系统包目录
   - utils: 公共功能包目录

每个部分的不同方法、模型和实现应该在mt、stt、tts、chat其中一个目录下建立单独的子包目录。<br>
例如，Tacotron实现语音合成，应在tts下建立tacotron包。
# In Progress
- 基于Seq2Seq的闲聊系统
- 基于DeepSpeech2的语音识别
- 基于Tacotron2的语音合成
- 基于Transformer的闲聊系统
- 基于Transformer的机器翻译
- 基于Transformer的语音识别
- 基于Transformer的语音合成
- 基于Listen-Attend-Spell的语音识别
- 基于检索的多轮闲聊系统
- RNN-T流式语音识别
- WaveRNN声码器
