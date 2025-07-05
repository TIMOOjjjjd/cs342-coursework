# second commit


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, var_red=None, value=None):
        """Node constructor for tree."""
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.var_red = var_red
        self.value = value


class DecisionTreeRegressorCustom():
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
        self.initial_prediction = np.mean(y)
        predictions = np.full_like(y, self.initial_prediction)
        for _ in range(self.n_trees):
            residuals = y - predictions
            tree = DecisionTreeRegressorCustom(min_samples_split=self.min_samples_split, max_depth=self.max_depth)
            tree.fit(X, residuals.reshape(-1, 1))
            self.trees.append(tree)
            predictions += self.learning_rate * np.array(tree.predict(X))

    def predict(self, X):
        predictions = np.full((X.shape[0],), self.initial_prediction)
        for tree in self.trees:
            predictions += self.learning_rate * np.array(tree.predict(X))
        return predictions


# Load and preprocess data
train = pd.read_csv('train_x.csv')
trainGT = pd.read_csv('train_y.csv')
test = pd.read_csv('test_x.csv')
testGT = pd.read_csv('test_y.csv')

X_train = train.values
X_test = test.values
y_train = trainGT.values.flatten()
y_test = testGT.values.flatten()

# Train sklearn decision tree
sklearn_tree = DecisionTreeRegressor(max_depth=5)
sklearn_tree.fit(X_train, y_train)
y_pred_sklearn = sklearn_tree.predict(X_test)

# Train single custom decision tree
my_tree = DecisionTreeRegressorCustom(min_samples_split=10, max_depth=5)
my_tree.fit(X_train, y_train.reshape(-1, 1))

# Print tree structure
print("Custom Decision Tree Structure:")
my_tree.print_tree()

# Train ensemble model
ensemble = EnsembleRegressionTree(n_trees=5, min_samples_split=10, learning_rate=0.1, max_depth=5)
ensemble.fit(X_train, y_train)
y_pred_ensemble = ensemble.predict(X_test)

# Evaluate models
y_pred_custom = my_tree.predict(X_test)
print("MSE (sklearn):", mean_squared_error(y_test, y_pred_sklearn))
print("MSE (custom single tree):", mean_squared_error(y_test, y_pred_custom))
print("MSE (ensemble):", mean_squared_error(y_test, y_pred_ensemble))
