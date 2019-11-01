from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# print(iris_X[:2, :])
# print(iris_y)

# 将所有的数据分成用于学习的数据和用于训练的数据，
# 好处，不会互相影响，不会出现人为误差
# test_size 是百分比，测试比总数据
X_train, X_test, y_train, y_test = train_test_split(iris_X,
                                                    iris_y, test_size=0.3)
# print(y_train)
# 打乱数据，乱的数据比不乱的数据更好
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
# 放进去，所有train数据
print(knn.predict(X_test))
print(y_test)
# 只能大概模拟，机器学习的本质
