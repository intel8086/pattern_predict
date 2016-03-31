# -*- coding: utf-8 -*-

"""copyright at Freescale YinHao
   2016.03
   use LR and SVM Essemble to do fail pattern prediction
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

df = pd.read_csv("imx_chip_1.csv")
#删除id列
df = df.drop('ID',1)

#feature 选择
df = df[['clock_frequency',
 'cells_num',
 'static_power',
 'dynamic_power',
 'test_timing_slack',
 'low_power_timing_slack',
 'function_power_timing_slack',
 'burnin_timing_slack',
 'TARGET']]
#数据分类
df_pass = df[df['TARGET'] == 0]
df_fail = df[df['TARGET'] == 1]
#填充无效数据，选取的是平均值
df_pass = df_pass.fillna(df.mean())
df_fail = df_fail.fillna(df.mean())

#计算df_fail和pass的数目比，实际是pass远大于fail，所以对pass的pattern进行聚类
ratio = int(df_pass.shape[0]/df_fail.shape[0])

#数据归一化，对于k-means算法，选择最大最小归一化
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = np.shape(dataSet)[0]
    normDataSet = dataSet - np.tile(minVals, (m,1))
    normDataSet = normDataSet/np.tile(ranges, (m,1))
    return normDataSet

pass_pattern = autoNorm(np.mat(df_pass.drop('TARGET',1)))
fail_pattern = autoNorm(np.mat(df_fail.drop('TARGET',1)))

clf = KMeans(n_clusters = int(ratio/3))
s = clf.fit(pass_pattern)
#获取聚类的结果
labels_cluster = clf.labels_

#放弃较小的子数据集,选择较大的子数据集
selected_list = []
for i in range(ratio):
    if sum(labels_cluster == i) > int(df_fail.shape[0]/3):
        selected_list.append(i)

models_linear  = {}
prediction = {}
dataSet_pass = []
j = 0
roc_auc = 0

#主训练函数
#划分数据集,从pass pattern中根据聚类和筛选的结果选择数据集与fail pattern数据集merge，再传入训练集切割函数
def data_partition(pass_pattern,fail_pattern,i):
    dataSet_pass = pass_pattern[labels_cluster == i].tolist()
#权重，以聚类的结果为主
    dataSet_pass_length = len(dataSet_pass)
    if int(len(dataSet_pass)/len(fail_pattern)) > 3:
        dataSet_pass = re_sample(dataSet_pass,len(fail_pattern) * 3)
    #print(len(dataSet_pass))
    fail_pattern = fail_pattern.tolist()
    labels_pass = [0 for i in range(len(dataSet_pass))]
    labels_fail = [1 for i in range(len(fail_pattern))]
    labels_pass.extend(labels_fail)
    dataSet_pass.extend(fail_pattern)
    print(len(labels_pass))
    return dataSet_pass,labels_pass,dataSet_pass_length

#重采样，如果聚类后簇的分布依然不均匀，那么进行重采样，总的策略就是去掉较小的簇，重采样较大的簇
def re_sample(dataSet,sample_num):
    sample_list = np.random.permutation(range(len(dataSet))).tolist()[0:sample_num]
    return np.mat(dataSet)[sample_list].tolist()

#训练和切割函数
def train_and_predict(dataSet,labels,model = 'LR'):
    dataSet = np.array(dataSet)
    labels = np.array(labels)
    x_train,x_test,y_train,y_test = train_test_split(dataSet,labels,test_size = 0.2, random_state = 20)
    if model == 'LR':
        model_train = linear_model.LogisticRegression(C = 1e5)
        print(x_train.shape)
        print(y_train.shape)
        model_train.fit(x_train,y_train)
    elif model == 'SVM':
        model_train = svm.SVC(kernel = 'linear')
        model_train.fit(x_train,y_train)
    else:
        raise NameError('Not support model type! You can use LR or SVM!')
    prediction = model_train.predict(x_test)
    return model_train,prediction,y_test

#选择ROC和AUC作为训练模型评估指标，原因在于ROC和AUC并不受数据分布的影响
def ROC_AUC(y_test,prediction):
    false_positive_rate,true_positive_rate,thresholds = roc_curve(y_test,prediction)
    roc_auc = auc(false_positive_rate,true_positive_rate)
    return false_positive_rate,true_positive_rate,roc_auc
    #plt.plot(false_positive_rate, true_positive_rate, colors[i], label='%s: AUC %0.2f'% j,roc_auc)

#画出每一次训练中AUC较高的那组模型的fp,tp和AUC
cmp = 0
colors = ['b','g','y','m','k']

def AUC_ROC_plot(false_positive_rate,true_positive_rate,model,colors,i,roc_auc):
    j = i % len(colors)
    plt.plot(false_positive_rate,true_positive_rate,colors[j], \
        label = '%s_%d: AUC %0.2f'%(model,i,roc_auc))

#总过程
clusters_weights = []
models = []
auc_total = 0
for i in range(len(selected_list)):
    #数据划分，这里选择的是20%的测试集
    dataSet,labels,weight = data_partition(pass_pattern,fail_pattern,selected_list[i])
#集合权重由merge后的数据集数目决定
    clusters_weights.append(weight)
#logistic regression模型和预测结果
    print(len(dataSet))
    print(len(labels))
    model_LR,prediction_LR,y_test_lr = train_and_predict(dataSet,labels,'LR')
#SVM模型核预测结果
    model_SVM,prediction_SVM,y_test_svm = train_and_predict(dataSet,labels,'SVM')
#LR结果的roc指标
    false_positive_rate_lr,true_positive_rate_lr,roc_auc_lr = \
            ROC_AUC(y_test_lr,prediction_LR)
#SVM结果的roc指标
    false_positive_rate_SVM,true_positive_rate_SVM,roc_auc_svm = \
            ROC_AUC(y_test_svm,prediction_SVM)
#每一次二者选择其中表现更优异的模型
    if roc_auc_lr >= roc_auc_svm:
        models.append(model_LR)
        print("%d th model is LR" % i)
        print(roc_auc_lr)
        AUC_ROC_plot(false_positive_rate_lr,true_positive_rate_lr,'LR',colors,i,roc_auc_lr)
        auc_total += roc_auc_lr * weight
    else:
        models.append(model_SVM)
        print("%d th model is SVM" % i)
        print(roc_auc_svm)
        AUC_ROC_plot(false_positive_rate_SVM,true_positive_rate_SVM,'SVM',colors,i,roc_auc_svm)
        auc_total += roc_auc_svm * weight

#隐含的权重因子归一化
total_weights = float(sum(clusters_weights))
clusters_weights = list(map(lambda x:x/total_weights,clusters_weights))
#最终整体的训练模型AUC值，代表着经验风险最小化的结果
print("our prediction mode total auc is %0.2f" % float(auc_total/total_weights))
print("train process are done")

plt.title("classifiers comparaison with ROC by YinHao")
plt.legend(loc = 'lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Postitive Rate')
plt.xlabel('False Postitive Rate')
plt.show()

#新数据的数据预处理过程
df_new = pd.read_csv("imx_chip_2.csv")
#删除id列
df_new = df_new.drop('ID',1)
df_new = df_new.fillna(df.mean())

#feature 选择
df = df[['clock_frequency',
 'cells_num',
 'static_power',
 'dynamic_power',
 'test_timing_slack',
 'low_power_timing_slack',
 'function_power_timing_slack',
 'burnin_timing_slack',
 'TARGET']]

#新数据归一化
dataSet_new = autoNorm(np.mat(df_new))
#预测函数
def patternPredict(dataSet_input,models,clusters_weights):
    models_num = len(models)
    print("models_num",models_num)
    predictions = []
    for i in range(models_num):
        predictions.append(models[i].predict(dataSet_input))
    print(np.mat(predictions).T[0,:])
    print(clusters_weights)
    results = (np.mat(predictions).T * np.mat(clusters_weights).T).tolist()
    return results

#转换为DataFrame然后输出至csv文件
results = pd.DataFrame(patternPredict(dataSet_new,models,clusters_weights))
results = results[0].map(lambda x:'fail' if x > 0.5 else 'pass')
results.to_csv('new_patterns_results.csv')
print("all prediction work is done,please get new_patterns_results.csv for your job")
