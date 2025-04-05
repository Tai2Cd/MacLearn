import numpy as np
import pandas as pd

# 读取数据集
df = pd.read_csv('./iris/iris.data', header=None)


# 提取特征和标签
X = df.iloc[:100, :4].values  # 选取前100行和前4列作为特征
y = df.iloc[:100, 4].values   # 选取前100行的标签

# 将标签转换为数字
y = np.where(y == 'Iris-setosa', 0, np.where(y == 'Iris-versicolor', 1, 2))
indices = np.random.permutation(len(X))  # 随机生成一个排列索引
X = X[indices]  # 打乱特征数据
y = y[indices]  # 打乱标签数据
# 标准化数据
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 将数据集分为训练集和测试集
X_train, X_test = X[:75], X[75:]
y_train, y_test = y[:75], y[75:]

#逻辑回归
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, weights, bias):
    m = len(y)
    predictions = sigmoid(np.dot(X, weights) + bias)
    cost = -(1/m) * np.sum(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
    return cost

def gradient_descent(X, y, weights, bias, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = sigmoid(np.dot(X, weights) + bias)
        dw = (1/m) * np.dot(X.T, (predictions - y))
        db = (1/m) * np.sum(predictions - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
        cost = compute_cost(X, y, weights, bias)
    return weights, bias

# 初始化权重和偏置
weights = np.zeros(X_train.shape[1])
bias = 0
learning_rate = 0.01
iterations = 1000

# 训练模型
weights, bias = gradient_descent(X_train, y_train, weights, bias, learning_rate, iterations)

print(f"Trained weights: {weights}")
print(f"Trained bias: {bias}")

#朴素贝叶斯
def naive_bayes_train(X, y):
    class_stats = {}
    classes = np.unique(y)
    for cls in classes:
        class_data = X[y == cls]
        mean = np.mean(class_data, axis=0)
        std = np.std(class_data, axis=0)
        prior = len(class_data) / len(y)
        class_stats[cls] = {'mean': mean, 'std': std, 'prior': prior}
    return class_stats

def naive_bayes_predict(X, class_stats):
    predictions = []
    for x in X:
        class_probs = []
        for cls, stats in class_stats.items():
            mean, std, prior = stats['mean'], stats['std'], stats['prior']
            likelihood = np.prod(np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi)))
            prob = likelihood * prior
            class_probs.append(prob)
        predictions.append(np.argmax(class_probs))
    return np.array(predictions)

# 训练朴素贝叶斯模型
class_stats = naive_bayes_train(X_train, y_train)
print("naive bayes params: \nmean: ")
for cls, stats in class_stats.items():
    mean = stats['mean']
    print(" ",mean,", ")
print("std: ")
for cls, stats in class_stats.items():
    std = stats['std']
    print(" ", mean, ", ")
# 预测
y_pred_nb = naive_bayes_predict(X_test, class_stats)
def predict_naive_bayes(X, class_stats):
    posteriors = []
    for cls, stats in class_stats.items():
        mean, std, prior = stats['mean'], stats['std'], stats['prior']
        likelihood = np.prod((1 / (np.sqrt(2 * np.pi) * std)) *
                             np.exp(-0.5 * ((X - mean) / std) ** 2), axis=1)
        posteriors.append(likelihood * prior)
    return np.argmax(posteriors, axis=0)

#评估
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def f1_score(y_true, y_pred):
    precision = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_pred == 1)
    recall = np.sum((y_true == 1) & (y_pred == 1)) / np.sum(y_true == 1)
    return 2 * (precision * recall) / (precision + recall)

# 逻辑回归预测
y_pred_lr = sigmoid(np.dot(X_test, weights) + bias)
y_pred_lr = (y_pred_lr >= 0.5).astype(int)

# 计算准确率和 F1-score
acc_lr = accuracy(y_test, y_pred_lr)
f1_lr = f1_score(y_test, y_pred_lr)

acc_nb = accuracy(y_test, y_pred_nb)
f1_nb = f1_score(y_test, y_pred_nb)

print(f"Logistic Regression Accuracy: {acc_lr:.8f}, F1-score: {f1_lr:.8f}")
print(f"Naive Bayes Accuracy: {acc_nb:.8f}, F1-score: {f1_nb:.8f}")

import matplotlib.pyplot as plt


def plot_decision_boundary(X, y, model, weights=None, bias=None, class_stats=None):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    if model == 'logistic':
        Z = sigmoid(np.dot(grid, weights[:2]) + bias)
        Z = (Z >= 0.5).astype(int)
    elif model == 'naive_bayes':
        reduced_stats = {cls: {
            'mean': stats['mean'][:2],
            'std': stats['std'][:2],
            'prior': stats['prior']
        } for cls, stats in class_stats.items()}
        Z = predict_naive_bayes(grid, reduced_stats)

    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(f'{model} Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid()
    plt.show()


# 绘制逻辑回归和朴素贝叶斯决策边界
plot_decision_boundary(X_train[:, :2], y_train, 'logistic', weights, bias)
plot_decision_boundary(X_train[:, :2], y_train, 'naive_bayes', class_stats=class_stats)