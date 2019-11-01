# 可视化学习的整个过程，如何降低loss和误差
from sklearn.model_selection import validation_curve
# 数字data 12345
from sklearn.datasets import load_digits
# 使用的model
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target
# 可以自己设定，变动的值还是gamma的范围,单位是log
# 数字的意思的在log-6在log-2.3 中 取 5个点进行参数的拟合
param_range = np.logspace(-6, -2.3, 5)
# learning-curve 输出的是是前三个
train_loss, test_loss = validation_curve(
    # 更改一个model的参数为0.001
    # param_name 定义哪个要更改的值,param_range是改变值的范围
    SVC(), X, y, param_name="gamma", param_range=param_range,
    cv=10,
    # 方差值对比
    scoring="neg_mean_squared_error"
)

# 平均值输出的是负值
train_loss_mean = -np.mean(train_loss, axis=1)
test_loss_mean = -np.mean(test_loss, axis=1)

# 输出结果可视化
# 学习长度，学习误差平均值，颜色，label
plt.plot(param_range, train_loss_mean, 'o-',
         color='r', label='Training')
plt.plot(param_range, test_loss_mean, 'o-',
         color="g", label="Cross_validation")

plt.xlabel('Gamma')
plt.ylabel('Loss')
# plt.legend  是显示图例的方法
# loc(设置图例显示的位置)
plt.legend(loc="best")
plt.show()
