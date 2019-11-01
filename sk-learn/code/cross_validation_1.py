# 来判断之前的model 到底好不好，取那个值好
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 调用cross_val_score 得分
# from sklearn.model_selection import cross_val_score
# 调用cross_val_score 得分
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

# 这个顺序是不能变的，xxyytraintrain test test
# X_train,  X_test, y_train, y_test = train_test_split(X, y, random_state=4)
# 附近5个点的值的中和得出y_pred
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred = knn.predict(X_test, y_test)
# print(knn.score(X_test, y_test))

# knn = KNeighborsClassifier(n_neighbors=5)
# # 验证交叉分数，分为5组
# scores = cross_val_score(knn, X, y, cv=5, scoring="accuracy")
# 平均之前的输出 [0.96666667 1.         0.93333333 0.96666667 1.        ]
# 将数据平均一下 .mean()
# print(scores.mean())
# 平均之后的输出  0.9733333333333334


k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 分成了10组，for classification
    # scores = cross_val_score(knn, X, y, cv=10, scoring="accuracy")
    # mean_squared_error 是判断误差，判断预测值和真实值的误差是多少
    # for regression
    loss = -cross_val_score(knn, X, y, cv=10, scoring="neg_mean_squared_error")
    k_scores.append(loss.mean())

plt.plot(k_range, k_scores)
plt.xlabel("Value of K for KNN")  # x轴的name
plt.ylabel("Cross-Validation Accuracy")
plt.savefig("d:/TOFU/Github/github_learn/" +
            "VSCode_git_github/TensorFlow/Machine_learning/sk-learn/loss.png")
plt.show()
