# 运行说明
+ execute.py主入口文件：指令需要附带运行参数
+ cmd：python execute.py [模型类别] [执行模式]
+ 模型类别：seq2seq/gpt2
+ 执行类别：chat/train/pre_treat

+ pre_treat模式为文本预处理模式，如果在没有分词结果集的情况下，需要先运行pre_treat模式
+ train模式为训练模式
+ chat模式为对话模式。chat模式下运行时，输入exit即退出对话。

**正常执行顺序为pre_treat->train->chat**

**相关参数后继补充**

# 版本日志：
+ 2020.9.9 
   + 目前分词处理的文本读取还有一点逻辑问题，待处理，所以在没有生成足够的train_tokenized.txt
的分词数据进行训练和预测时，需要执行pre_treat模式
   + 还有模型评估和保存下一个issue完善
+ 2020.9.10
   + 更新了text.txt和text_tokenized.txt用于测试，只包含一百左右的问答对


# 数据集
data目录中的语料为缩减版，是原版语料的六分之一，需要原版语料可以自取

链接：https://pan.baidu.com/s/1qAIrdX-mv-Bzq1Y7MDjmLg 

提取码：r6da

# Seq2Seq

# GPT-2

# ALBert