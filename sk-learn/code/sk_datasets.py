from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data  # 没有括号只是属性
data_y = loaded_data.target

model = LinearRegression()  # 如果没有能力，就取默认参数，定义这个model
model.fit(data_X, data_y)  # 这是代表默认值，默认值已经很好了

print(model.predict(data_X[:4, :]))
print(data_y[:4])
# 以上是用自己导入的模型，训练
X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1, noise=1)
plt.scatter(X, y)
plt.show()
