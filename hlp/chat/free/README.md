# 目录
+ [运行说明](#运行说明)
+ [模型效果](#模型效果)
+ [BeamSearch说明及效果](#BeamSearch)


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

# BeamSearch
Beam Search功能已经基本完成且完善，抽象完成并能够应用在不同模型之间(目前已经非常方便的应用在了Seq2Seq和Transformer模型中)。注：在调用本Beam Search实现时，需要满足以下几点要求：
+ 首先需要将问句编码成token向量并对齐，然后调用init_input方法进行初始化
+ 对模型要求能够进行批量输入
+ BeamSearch使用实例已经集成到Chatter中，如果不进行自定义调用，可以将聊天器继承Chatter，在满足上述两点的基础之上设计create_predictions方法，并调用BeamSearch

+ **beam_size = 1**

![](https://img-blog.csdnimg.cn/2020092221154427.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

+ **beam_size = 2**

![](https://img-blog.csdnimg.cn/20200922211209570.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)

+ **beam_size = 3**

![](https://img-blog.csdnimg.cn/20200922211722639.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0RCQ18xMjE=,size_16,color_FFFFFF,t_70#pic_center)
