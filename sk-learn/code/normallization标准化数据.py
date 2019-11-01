from sklearn import preprocessing
import numpy as np

# 将测试数据和训练数据分开的库
from sklearn.model_selection import train_test_split
from sklearn.datasets.samples_generator import make_classification

# 处理model
from sklearn.svm import SVC
import matplotlib.pyplot as plt

# #  数据类型范围
# a = np.array([[10, 2.7, 3.6],
#               [-100, 5, -2],
#               [120, 20, 40]], dtype=np.float64)

# print(a)
# print(preprocessing.scale(a))
# # 更适合Machine Learning 处理的数据结构
X, y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=22,
    n_clusters_per_class=1,
    scale=100,
)
# n_features :特征个数= n_informative（） + n_redundant + n_repeated
# n_informative：多信息特征的个数
# n_redundant：冗余信息，informative特征的随机线性组合
# n_repeated ：重复信息，随机提取n_informative和n_redundant 特征
# n_classes：分类类别
# n_clusters_per_class ：某一个类别是由几个cluster构成的
# plt.scatter(X[:, 0], X[:, 1], c=y)
# x[m,n]是通过numpy库引用数组或矩阵中的某一段数据集的一种写法，
# m代表第m维，n代表m维中取第几段特征数据。
# x[:,n]表示在全部数组（维）中取第n个数据，直观来说，x[:,n]就是取所有集合的第n个数据,
# x[n,:]表示在n个数组（维）中取全部数据，直观来说，x[n,:]就是取第n集合的所有数据,
# x[:,m:n]，即取所有数据集的第m到n-1列数据
# plt.show()
# feature_range 的默认范围是0,1
# X = preprocessing.scale(X)
# 输出结果 0.9555555555555556
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
clf = SVC()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
# 这个流程大致的意思(机器学习的步骤)
# 将x，y的数据分成了测试数据和训练数据
# 用train data去学习，test data去预测
# 如果不适用preprocessing normallization 的输出结果
# 0.45555555555555555
