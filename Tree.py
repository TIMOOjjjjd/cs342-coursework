from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.metrics import mean_squared_error as MSE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression


class Node():
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        ''' constructor '''

        # for decision node
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red

        # for leaf node
        self.value = value


class DecisionTreeRegressor():
    def __init__(self, min_samples_split=2, max_depth=2):
        ''' constructor '''

        # initialize the root of the tree
        self.root = None

        # stopping conditions
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, dataset, curr_depth=0):
        ''' recursive function to build the tree '''

        X, Y = dataset[:, :-1], dataset[:, -1]
        num_samples, num_features = np.shape(X)
        best_split = {}
        # split until stopping conditions are met
        if num_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            # find the best split
            best_split = self.get_best_split(dataset, num_samples, num_features)
            # check if information gain is positive
            if best_split["var_red"] > 0:
                # recur left
                left_subtree = self.build_tree(best_split["dataset_left"], curr_depth + 1)
                # recur right
                right_subtree = self.build_tree(best_split["dataset_right"], curr_depth + 1)
                # return decision node
                return Node(best_split["feature_index"], best_split["threshold"],
                            left_subtree, right_subtree, best_split["var_red"])

        # compute leaf node
        leaf_value = self.calculate_leaf_value(Y)
        # return leaf node
        return Node(value=leaf_value)

    def get_best_split(self, dataset, num_samples, num_features):
        ''' function to find the best split '''

        # dictionary to store the best split
        best_split = {}
        max_var_red = -float("inf")
        # loop over all the features
        for feature_index in range(num_features):
            feature_values = dataset[:, feature_index]
            possible_thresholds = np.unique(feature_values)
            # loop over all the feature values present in the data
            for threshold in possible_thresholds:
                # get current split
                dataset_left, dataset_right = self.split(dataset, feature_index, threshold)
                # check if childs are not null
                if len(dataset_left) > 0 and len(dataset_right) > 0:
                    y, left_y, right_y = dataset[:, -1], dataset_left[:, -1], dataset_right[:, -1]
                    # compute information gain
                    curr_var_red = self.variance_reduction(y, left_y, right_y)
                    # update the best split if needed
                    if curr_var_red > max_var_red:
                        best_split["feature_index"] = feature_index
                        best_split["threshold"] = threshold
                        best_split["dataset_left"] = dataset_left
                        best_split["dataset_right"] = dataset_right
                        best_split["var_red"] = curr_var_red
                        max_var_red = curr_var_red

        # return best split
        return best_split

    def split(self, dataset, feature_index, threshold):
        ''' function to split the data '''

        dataset_left = np.array([row for row in dataset if row[feature_index] <= threshold])
        dataset_right = np.array([row for row in dataset if row[feature_index] > threshold])
        return dataset_left, dataset_right

    def variance_reduction(self, parent, l_child, r_child):
        ''' function to compute variance reduction '''

        weight_l = len(l_child) / len(parent)
        weight_r = len(r_child) / len(parent)
        reduction = np.var(parent) - (weight_l * np.var(l_child) + weight_r * np.var(r_child))
        return reduction

    def calculate_leaf_value(self, Y):
        ''' function to compute leaf node '''

        val = np.mean(Y)
        return val

    def print_tree(self, tree=None, indent=" "):
        ''' function to print the tree '''

        if not tree:
            tree = self.root

        if tree.value is not None:
            print(tree.value)

        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?", tree.var_red)
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

    def fit(self, X, Y):
        ''' function to train the tree '''

        dataset = np.concatenate((X, Y), axis=1)
        self.root = self.build_tree(dataset)

    def make_prediction(self, x, tree):
        ''' function to predict new dataset '''

        if tree.value != None: return tree.value
        feature_val = x[tree.feature_index]
        if feature_val <= tree.threshold:
            return self.make_prediction(x, tree.left)
        else:
            return self.make_prediction(x, tree.right)

    def predict(self, X):
        ''' function to predict a single data point '''

        preditions = [self.make_prediction(x, self.root) for x in X]
        return preditions

class EnsembleRegressionTree:
    def __init__(self, n_trees=5, min_samples_split=10, learning_rate=0.1, max_depth=2):
        self.n_trees = n_trees
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate
        self.trees = []
        self.initial_prediction = None
        self.max_depth = max_depth

    def fit(self, X, y):
        self.initial_prediction = np.mean(y)  # 初始化预测值
        predictions = np.full(y.shape, self.initial_prediction)  # 初始化预测值，与 y 形状一致

        for i in range(self.n_trees):
            residuals = y.ravel() - predictions.ravel()  # 确保 residuals 是一维
            print(f"Tree {i + 1}: residuals shape = {residuals.shape}, predictions shape = {predictions.shape}")
            tree = DecisionTreeRegressor(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X, residuals.reshape(-1, 1))  # 确保 residuals 是二维
            self.trees.append(tree)
            predictions += self.learning_rate * np.array(tree.predict(X)).reshape(-1, 1)  # 保持 predictions 是二维

    def predict(self, X):
        predictions = np.full((X.shape[0], 1), self.initial_prediction)  # 初始化预测值为二维数组
        for tree in self.trees:
            predictions += self.learning_rate * np.array(tree.predict(X)).reshape(-1, 1)  # 更新预测值
        return predictions.ravel()  # 返回一维数组作为最终输出



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


train = pd.read_csv('train_x.csv')
trainGT = pd.read_csv('train_y.csv')
test = pd.read_csv('test_x.csv')
testGT = pd.read_csv('test_y.csv')


X_train = train.iloc[:, :-1].values
X_test = test.iloc[:, :-1].values

# Target (last column)
y_train = trainGT.iloc[:, -1].values
y_test = testGT.iloc[:, -1].values

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")

# 如果 Y 是一维数组，转换为二维数组
if len(y_train.shape) == 1:
    y_train = y_train.reshape(-1, 1)

print(f"y_train reshaped: {y_train.shape}")



sklearn_tree = DecisionTreeRegressor(min_samples_split=10)
sklearn_tree.fit(X_train, y_train)

# 使用自定义回归树
my_tree = DecisionTreeRegressor(min_samples_split=10)
my_tree.fit(X_train, y_train)
my_tree.print_tree()




# 比较预测结果
y_pred_sklearn = sklearn_tree.predict(X_test)
y_pred_custom = my_tree.predict(X_test)

ensemble = EnsembleRegressionTree(n_trees=5, min_samples_split=10, learning_rate=0.1, max_depth=5)
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)


print("y_pred_custom:", y_pred_custom)
print("Type of y_pred_custom:", type(y_pred_custom))


print(f"y_pred_custom contains NaN: {np.isnan(y_pred_custom).any()}")
print(f"y_pred_custom: {y_pred_custom}")

print("MSE (sklearn):", mean_squared_error(y_test, y_pred_sklearn))
print("MSE (custom):", mean_squared_error(y_test, y_pred_custom))
print("MSE (ensemble):", mean_squared_error(y_test, y_pred_ensemble))


