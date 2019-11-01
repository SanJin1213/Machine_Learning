from sklearn import svm
from sklearn import datasets

clf = svm.SVC()
iris = datasets.load_iris()
X, y = iris.data, iris.target
clf.fit(X, y)


# save model 参数的方式
# method 1: pickle(python 自带的保存方式)
# import pickle
# # save
# # # with open('D:\TOFU\Github\github_learn\VSCode_git_github'
# # #             + '\TensorFlow\Machine_learning\sk-learn\'
# # #              'save\clf.pickle', 'wb') as f:
# # #                      pickle.dump(clf, f)
# # restore
# with open(
#     "d:/TOFU/Github/github_learn/VSCode_git_github/" +
#     "TensorFlow/Machine_learning/sk-learn/save/clf.pickle",
#     "rb",
# ) as f:
#     clf2 = pickle.load(f)
#     print(clf2.predict(X[0:1]))

# method 2:joblib
from sklearn.externals import joblib

# save
joblib.dump(
    clf,
    "d:/TOFU/Github/github_learn/VSCode_git_github/"
    + "TensorFlow/Machine_learning/sk-learn/save/clf.pkl"
)
# restore
clf3 = joblib.load(
    "d:/TOFU/Github/github_learn/VSCode_git_github/"
    + "TensorFlow/Machine_learning/sk-learn/save/clf.pkl"
)
print(clf3.predict(X[0:1]))
