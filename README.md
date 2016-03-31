# pattern_predict
1.简介
  pattern_predict目标是通过以往收集的芯片测试pattern的相关数据和在测试机上的表现，预测新项目由设计工程师传递给测试工程师的pattern的表现。目的是节约测试时间，减少测试工程师和设计工程师的交互次数，达到节约成本的目标。

2.数据特点介绍
  1.分布不均衡，pass的pattern比fail的pattern远远要多（比例在10-20左右 ）
  2.有影响的feature相对明确，根据以前debug的经验，pattern的表现主要与时钟频率，pattern的覆盖逻辑数，pattern的运行功耗，pattern的运行时序表现相关，所以主要选择这几个特征做训练。

3.中心思想和贡献点
  基于LR模型和SVM模型作为基础学习器，选出表现较好的基础模型，然后进行集成。

  贡献点：
  （1）基于LR和非核SVM进行ensemble;
  （2）数据均衡化处理，基于聚类算法和re-sample
  （3）模型权重因子计算

 4.结果
   在训练集上表现良好，平均AUC为0.88。
   在测试集上，经过权重处理后，模型预测的fail pattern能够覆盖70%的实际fail pattern，强于预期。
   在新项目中的表现有待观察。

 5.运行(由于保密性，原始数据集未能提供)
   python pattern_predict.py
