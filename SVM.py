from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. data
iris_dataset = load_iris()

print(iris_dataset.DESCR)
print(iris_dataset.feature_names)

x = iris_dataset.data
y = iris_dataset.target

print(x.shape, y.shape)
print(x[:10])
print(y)


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=2022
)

# 2. model
svm_model = SVC(kernel="rbf", C=8, gamma=0.1)


# 3. train
svm_model.fit(x_train, y_train)


# 4. predcit, evaluate
y_prdecit = svm_model.predict(x_test)


y_prdecit = svm_model.predict(x_test)

print("예측값:", y_prdecit)
print("정답:", y_test)

acc = accuracy_score(y_test, y_prdecit)

print("정확도:", acc)
