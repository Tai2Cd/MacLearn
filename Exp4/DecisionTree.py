import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
# 1. 数据加载
df = pd.read_csv('wdbc.data',header=None)

# 2. 数据预处理
df = df.drop(columns=[0])
# 将诊断列映射为二分类标签：M -> 1 (恶性), B -> 0 (良性)
df[1] = df[1].map({'M': 1, 'B': 0})

# 3. 提取特征和标签
X = df.iloc[:, 1:].values  # 特征列，从第三列开始
y = df.iloc[:, 0].values  # 标签列，第二列为诊断

# 4. 手动划分数据集（70% 训练集，30% 测试集）
# 计算划分点
np.random.seed(42)  # 固定随机种子，以确保可复现
indices = np.random.permutation(len(X))
train_size = int(0.7 * len(X))  # 70% 训练集
train_indices = indices[:train_size]
test_indices = indices[train_size:]

# 划分数据集
X_train, y_train = X[train_indices], y[train_indices]
X_test, y_test = X[test_indices], y[test_indices]


# 5. 计算信息熵
def calculate_entropy(y):
    # 计算熵
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return -np.sum(probs * np.log2(probs))


# 6. 计算信息增益（ID3 算法）
def calculate_info_gain(X_col, y):
    # 计算特征列的熵
    entropy_before = calculate_entropy(y)

    # 根据特征 X_col 划分数据
    values, counts = np.unique(X_col, return_counts=True)
    weighted_entropy = 0
    for value, count in zip(values, counts):
        y_sub = y[X_col == value]
        weighted_entropy += (count / len(y)) * calculate_entropy(y_sub)

    # 返回信息增益
    return entropy_before - weighted_entropy


# 7. 选择最优特征（信息增益最大）
def choose_best_feature(X, y):
    best_info_gain = -1
    best_feature = -1
    for i in range(X.shape[1]):
        info_gain = calculate_info_gain(X[:, i], y)
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = i
    return best_feature


# 8. 构建决策树（递归构建）
def build_tree(X, y, depth=0, max_depth=15):
    # 停止条件：
    # 1) 所有样本属于同一类
    # 2) 达到最大深度
    # 3) 没有特征可用
    if len(np.unique(y)) == 1:
        return np.unique(y)[0]
    if X.shape[1] == 0 or depth >= max_depth:
        return np.bincount(y).argmax()

    best_feature = choose_best_feature(X, y)

    # 根据最佳特征进行数据分裂
    tree = {best_feature: {}}
    values = np.unique(X[:, best_feature])
    for value in values:
        # 分割数据集
        # 根据最佳特征进行数据分割
        mask = X[:, best_feature] == value
        X_sub = X[mask]
        y_sub = y[mask]
        tree[best_feature][value] = build_tree(X_sub, y_sub, depth + 1, max_depth)

    return tree

#8.5 剪枝
def prune_tree(tree, X, y):
    """
    后剪枝：剪除不必要的子树
    - 如果子树的误差大于其叶节点的误差，则剪除子树
    """
    if isinstance(tree, dict):
        feature = list(tree.keys())[0]  # 当前节点的特征
        subtrees = tree[feature]
        # 遍历所有子树进行剪枝
        for value in subtrees:
            subtree = subtrees[value]
            # 递归剪枝子树
            prune_tree(subtree, X, y)

        # 计算当前树和当前树的叶节点误差
        tree_predictions = [predict(tree, x) for x in X]
        tree_error = np.mean(tree_predictions != y)

        # 计算所有叶节点的误差
        leaf_predictions = [np.bincount(y[X[:, feature] == value]).argmax() for value in subtrees]
        leaf_error = np.mean(leaf_predictions != y)

        # 如果叶节点误差更小，则替换子树为叶节点
        if leaf_error < tree_error:
            most_common = np.bincount(y).argmax()
            return most_common  # 剪枝成叶节点，返回最常见的类
    return tree

# 9. 预测函数
def predict(tree, X):
    if isinstance(tree, dict):
        feature = list(tree.keys())[0]  # 获取当前节点的特征
        feature_value = X[feature]  # 获取当前样本的特征值
        if feature_value in tree[feature]:
            return predict(tree[feature][feature_value], X)
        else:
            return -1  # 如果找不到特征值，则返回 -1（表示预测失败）
    return tree


# 11. 可视化决策树
def plot_tree(tree, parent_name, graph, node_pos, x=0.5, y=1, width=0.2, vert_gap=0.2, node_id=0):
    """
    递归绘制决策树，返回所有节点的位置和编号
    """
    if parent_name not in node_pos:
        node_pos[parent_name] = (x, y)  # 给根节点分配位置

    if isinstance(tree, dict):
        feature = list(tree.keys())[0]  # 获取当前节点的特征
        values = list(tree[feature].keys())
        for i, value in enumerate(values):
            node_id += 1
            current_node_id = node_id
            # 计算子节点的位置
            child_x = x - width * (len(values) - 1) / 2 + i * width
            graph.append((parent_name, current_node_id, f"Feature {feature} = {value}"))
            node_pos[current_node_id] = (child_x, y - vert_gap)
            node_id = plot_tree(tree[feature][value], current_node_id, graph, node_pos, child_x, y - vert_gap,
                                width / 2, vert_gap, node_id)
    else:
        leaf_name = f"Class {tree}"
        graph.append((parent_name, node_id, leaf_name))
        node_pos[node_id] = (x, y)

    return node_id


def draw_tree(tree):
    """
    绘制决策树图形
    """
    graph = []
    node_pos = {}
    plot_tree(tree, 'Root', graph, node_pos)  # 确保根节点被添加到节点位置字典中

    # 准备画图
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 绘制树的连接线和节点标签
    for parent, child, label in graph:
        parent_pos = node_pos[parent]
        child_pos = node_pos[child]
        ax.plot([parent_pos[0], child_pos[0]], [parent_pos[1], child_pos[1]], 'k-')
        ax.text(child_pos[0], child_pos[1], label, fontsize=10, ha='center', va='center')

    plt.axis('off')  # 关闭坐标轴
    plt.show()

# 10. 评估模型性能
def evaluate_tree(tree, X_test, y_test):
    predictions = [predict(tree, x) for x in X_test]
    accuracy = np.mean(predictions == y_test)
    return accuracy


# 11. 训练模型
tree = build_tree(X_train, y_train)

tree = prune_tree(tree, X_train, y_train)

# 12. 评估模型
draw_tree(tree)
accuracy = evaluate_tree(tree, X_test, y_test)
print(f"模型准确率：{accuracy}")
