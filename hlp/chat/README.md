# 运行说明
+ 运行入口：
   + seq2seq_chatter.py为seq2seq的执行入口文件：指令需要附带运行参数
   + transformer_chatter.py为transformer的执行入口文件：指令需要附带运行参数
+ 执行的指令格式：
   + seq2seq：python seq2seq_chatter.py -t/--type [执行模式]
   + transformer：python transformer_chatter.py -t/--type [执行模式]
+ 执行类别：pre_treat(默认)/train/chat
+ 执行指令示例：
   + python seq2seq_chatter.py
   + python seq2seq_chatter.py -t pre_treat
   + python transformer_chatter.py
   + python transformer_chatter.py -t pre_treat
+ pre_treat模式为文本预处理模式，如果在没有分词结果集的情况下，需要先运行pre_treat模式
+ train模式为训练模式
+ chat模式为对话模式。chat模式下运行时，输入exit即退出对话。

+ 正常执行顺序为pre_treat->train->chat

# 模型效果
#### seq2seq模型说明及效果
配置文件中的max_train_data_size用来设置训练用的最大数据量，目前设置的是100个问答对，在运行了100epoch之后的测试结果如下：

![](https://img-blog.csdnimg.cn/20200911224136498.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

#### Transformer架构
目前设置的是100个问答对，在运行了300epoch之后的测试结果如下：

![](https://img-blog.csdnimg.cn/20200916135748737.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

# 版本日志：
+ 2020.09.21：完善Beam Search代码
   + 完善Chatter聊天器
   + 完善Beam Search在各模型应用的逻辑体系


# 数据集
data目录中的语料为缩减版，是原版语料的六分之一，需要原版语料可以自取

链接：https://pan.baidu.com/s/1qAIrdX-mv-Bzq1Y7MDjmLg 

提取码：r6da

# Seq2Seq

# GPT-2

# ALBert