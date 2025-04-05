import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

# 1. 数据加载
# 假设数据保存在 'data.csv' 中，数据没有列名
df = pd.read_csv('wdbc.data', header=None)  # header=None 表示没有列名

# 2. 删除第一列（ID 列），并设置诊断列作为标签
df = df.drop(columns=[0])  # 删除第一列（ID）

# 3. 将诊断列映射为二分类标签：M -> 1 (恶性), B -> 0 (良性)
df[1] = df[1].map({'M': 1, 'B': 0})

# 4. 提取特征和标签
X = df.iloc[:, 1:].values  # 特征列，从第二列开始
y = df.iloc[:, 0].values   # 标签列，第一列为诊断标签

# 5. 手动划分数据集（70% 训练集，30% 测试集）
np.random.seed(42)  # 固定随机种子，以确保可复现
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))  # 70% 训练集
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# 划分数据集
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]

# 6. 构建决策树
tree = DecisionTreeClassifier(criterion='entropy', max_depth=None, random_state=42)
tree.fit(X_train, y_train)

# 7. 绘制决策树
plt.figure(figsize=(20, 10))
plot_tree(tree,
          filled=True,
          feature_names=list(df.columns[1:]),  # 转换为 list 类型
          class_names=['Benign', 'Malignant'],
          rounded=True)
plt.title("Decision Tree")
plt.show()

# 8. 评估模型性能
accuracy = tree.score(X_test, y_test)
print(f"模型准确率：{accuracy * 100:.2f}%")

# 9. 计算AUC（Area Under the Curve）
# 获取模型对测试集的预测概率
y_prob = tree.predict_proba(X_test)[:, 1]  # 获取属于类别1（恶性）的概率
auc = roc_auc_score(y_test, y_prob)
print(f"AUC值：{auc:.4f}")