# 可视化学习的整个过程，如何降低loss和误差
from sklearn.model_selection import learning_curve
# 数字data 12345
from sklearn.datasets import load_digits
# 使用的model
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

# learning-curve 输出的是是前三个
train_sizes, train_loss, test_loss = learning_curve(
    # 更改一个model的参数为0.001
    SVC(gamma=0.001), X, y, cv=10,
    # 方差值对比
    scoring="neg_mean_squared_error",
    # 记录的点的百分比10%，25%，50%，75%，100%
    train_sizes=[0.1, 0.25, 0.5, 0.75, 1]
)

# 平均值输出的是负值
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

# 输出结果可视化
# 学习长度，学习误差平均值，颜色，label
plt.plot(train_sizes, train_loss_mean, 'o-',
         color='r', label='Training')
plt.plot(train_sizes, test_loss_mean, 'o-',
         color="g", label="Cross_validation")

plt.xlabel('Training examples')
plt.ylabel('Loss')
# plt.legend  是显示图例的方法
# loc(设置图例显示的位置)
plt.legend(loc="best")
plt.show()
