from sklearn import datasets
from sklearn.linear_model import LinearRegression

# import matplotlib.pyplot as plt

loaded_data = datasets.load_boston()
data_X = loaded_data.data  # 没有括号只是属性
data_y = loaded_data.target

model = LinearRegression()  # 如果没有能力，就取默认参数，定义这个model，是这个model
model.fit(data_X, data_y)  # 这是代表默认值，默认值已经很好了

# print(model.predict(data_X[:4, :]))
# print(data_y[:4])
# 以上是用自己导入的模型，训练
# X, y = datasets.make_regression(n_samples=100, n_features=1, n_targets=1,
#                                 noise=1)
# plt.scatter(X, y)
# plt.show()
# fit 完之后才会有一个稳定的值
# 形如  y=0.1x+0.3  model.coef_ = 0.1x , model.intercept_ = 0.3
# print(model.coef_)
# print(model.intercept_)
# [-1.08011358e-01  4.64204584e-02  2.05586264e-02  2.68673382e+00
#  -1.77666112e+01  3.80986521e+00  6.92224640e-04 -1.47556685e+00
#   3.06049479e-01 -1.23345939e-02 -9.52747232e-01  9.31168327e-03
#  -5.24758378e-01] 楼层*第一个= ，面积*第二个= 等等，这样的一个x值
# 36.459488385089855 与y轴的交点
# print(model.get_params())
# {'copy_X': True, 'fit_intercept': True, 'n_jobs': None, 'normalize': False}
# 查看之前model定义的参数
print(model.score(data_X, data_y))
# 对model 学到的东西进行打分,前者为预测，后者为对比，
# 在Regression 中用R^2 coefficient of determination
# 主要对比真实数据和预测数据到底有多吻合， 精确度的输出结果
