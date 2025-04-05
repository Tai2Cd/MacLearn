import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv("./iris/iris.data", header=None)

# 提取前两个类别的样本
data = data[data.iloc[:, -1].isin(['Iris-setosa', 'Iris-versicolor'])].reset_index(drop=True)

# 编码目标列 (class1 -> 0, class2 -> 1)
data.iloc[:, -1] = data.iloc[:, -1].map({'Iris-setosa': 0, 'Iris-versicolor': 1})

# 提取特征和目标
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 打乱数据集并划分训练集和测试集
np.random.seed(42)
indices = np.arange(len(X))
np.random.shuffle(indices)
X, y = X[indices], y[indices]

# 划分 70% 训练集和 30% 测试集
split_ratio = 0.75
split_index = int(len(X) * split_ratio)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression_train(X, y, lr=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0

    for _ in range(epochs):
        # 计算预测值
        linear_model = np.dot(X, weights) + bias
        y_pred = sigmoid(linear_model)

        # 计算梯度
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)

        # 更新参数
        weights -= lr * dw
        bias -= lr * db

    return weights, bias

weights, bias = logistic_regression_train(X_train, y_train)
print("weights: ",weights)
print("bias: ",bias)

def naive_bayes_train(X, y):
    class_stats = {}
    classes = np.unique(y)

    for cls in classes:
        X_cls = X[y == cls]
        mean = X_cls.mean(axis=0)
        std = X_cls.std(axis=0)
        prior = len(X_cls) / len(X)
        class_stats[cls] = {'mean': mean, 'std': std, 'prior': prior}

    return class_stats


class_stats = naive_bayes_train(X_train, y_train)
print(class_stats)


def predict_logistic_regression(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    y_pred = sigmoid(linear_model)
    return (y_pred >= 0.5).astype(int)


def predict_naive_bayes(X, class_stats):
    posteriors = []
    for cls, stats in class_stats.items():
        mean, std, prior = stats['mean'], stats['std'], stats['prior']
        likelihood = np.prod((1 / (np.sqrt(2 * np.pi) * std)) *
                             np.exp(-0.5 * ((X - mean) / std) ** 2), axis=1)
        posteriors.append(likelihood * prior)
    return np.argmax(posteriors, axis=0)


def calculate_metrics(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return accuracy, precision, recall, f1_score


# 逻辑回归性能
y_pred_lr = predict_logistic_regression(X_test, weights, bias)
accuracy_lr, precision_lr, recall_lr, f1_score_lr = calculate_metrics(y_test, y_pred_lr)
print(y_test)
# 朴素贝叶斯性能
y_pred_nb = predict_naive_bayes(X_test, class_stats)
accuracy_nb, precision_nb, recall_nb, f1_score_nb = calculate_metrics(y_test, y_pred_nb)


print("Logistic Regression - Accuracy: {:.4f}, F1-Score: {:.4f}".format(accuracy_lr, f1_score_lr))
print("Naive Bayes - Accuracy: {:.4f}, F1-Score: {:.4f}".format(accuracy_nb, f1_score_nb))

def predict_logistic_regression_two_features(X, weights, bias):
    # 只使用前两个特征
    X_reduced = X[:, :2]
    linear_model = np.dot(X_reduced, weights) + bias
    y_pred = sigmoid(linear_model)
    return (y_pred >= 0.5).astype(int)
def plot_decision_boundary(X, y, model, title):
    # 定义绘图范围
    x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
    y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 根据模型计算预测结果
    if model == 'logistic':
        Z = predict_logistic_regression(grid, weights[:2], bias)
    elif model == 'naive_bayes':
        # 只使用前两列均值和标准差
        reduced_stats = {cls: {
            'mean': stats['mean'][:2],
            'std': stats['std'][:2],
            'prior': stats['prior']
        } for cls, stats in class_stats.items()}
        Z = predict_naive_bayes(grid, reduced_stats)
    else:
        raise ValueError("Unknown model type")

    # 将预测结果映射到网格
    Z = Z.reshape(xx.shape)

    # 绘制决策边界和数据点
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, levels=np.unique(y).size, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.grid()
    plt.show()


# 逻辑回归决策边界
plot_decision_boundary(X_train[:,:2], y_train, 'logistic', 'Logistic Regression Decision Boundary')

# 朴素贝叶斯决策边界
plot_decision_boundary(X_train[:,:2], y_train, 'naive_bayes', 'Naive Bayes Decision Boundary')