# Dependencies
import numpy as np
from collections import Counter


# Class representing a single node in the decision tree
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature  # The feature index used for splitting at this node
        self.threshold = threshold  # The threshold value for the feature to split the data
        self.left = left  # Left child node (subtree)
        self.right = right  # Right child node (subtree)
        self.value = value  # If this is a leaf node, value holds the prediction

    # Check if the current node is a leaf node
    def is_leaf_node(self):
        return self.value is not None


# Class for implementing the decision tree algorithm
class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=100, n_features=None):
        self.min_samples_split = min_samples_split  # Min number of samples required to split a node
        self.max_depth = max_depth  # Max depth of the tree
        self.n_features = n_features  # Number of features to consider when look for the best split
        self.root = None  # The root node (first node) of the tree

    # Fit the decision tree model on the training data
    def fit(self, X, y):
        # If n_features is not specified, use all features
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)  # Begin growing the tree starting from the root

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape  # Number of samples and features
        n_labels = len(np.unique(y))  # Number of unique labels

        # Check the stopping criteria for recursion
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            # If stopping criteria met, create a leaf node with the most common label
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Select a random subset of features to consider for the best split
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find the best feature and threshold to split the data
        best_feature, best_threshold = self._find_best_split(X, y, feat_idxs)

        # Split the data into left and right branches based on the best split
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        # Recursively grow the left and right subtrees
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        # Return the current node with the best feature & threshold to split on
        return Node(best_feature, best_threshold, left, right)

    # Find the best feature and threshold to split the data
    def _find_best_split(self, X, y, feat_idxs):
        best_gain = -1  # Initialize best information gain as the worst possible
        split_idx, split_threshold = None, None  # Initialize best split feature and threshold

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]  # Get all values for the current feature
            thresholds = np.unique(X_column)  # Get all unique values in the feature column

            for threshold in thresholds:
                # Calculate information gain for this threshold
                gain = self._information_gain(y, X_column, threshold)

                # If this split provides better info gain than current best, update best gain & feature & threshold
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold

        return split_idx, split_threshold

    # Calculate information gain of a split
    def _information_gain(self, y, X_column, threshold):
        # Calculate entry of parent node
        parent_entropy = self._entropy(y)

        # Split the data into left and right branches (L & R children)
        left_idxs, right_idxs = self._split(X_column, threshold)

        # If the split doesn't divide the data, return zero gain
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate weighted avg. entropy of the children nodes
        num = len(y)
        num_left, num_right = len(left_idxs), len(right_idxs)
        entropy_left, entropy_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = ((num_left / num) * entropy_left) + ((num_right / num) * entropy_right)

        # Calculate inforamtion gain as the reduction in entropy from parent to children
        information_gain = parent_entropy - child_entropy

        return information_gain

    # Calculate entropy of a label distribution
    def _entropy(self, y):
        hist = np.bincount(y)  # Count occurrences of each label
        probabilities = hist / len(y)  # Calculate probabilities
        return -np.sum([prob * np.log(prob) for prob in probabilities if prob > 0])  # Entropy formula

    # Split the data based on the given threshold
    def _split(self, X_column, split_threshold):
        left_idxs = np.argwhere(X_column <= split_threshold).flatten()  # Indices for the left branch
        right_idxs = np.argwhere(X_column >= split_threshold).flatten()  # Indices for the right branch
        return left_idxs, right_idxs

    # Get the most common label in the data
    def _most_common_label(self, y):
        counter = Counter(y)  # Count occurrences of each label
        value = counter.most_common(1)[0][0]  # Get the label with the highest count
        return value

    # Preict the label for each sample in the dataset
    def predict(self, X, ):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    # Traverse the tree to predict the label for a single sample
    def _traverse_tree(self, x, node):
        # If the current node is a leaf, return its value (predicted label)
        if node.is_leaf_node():
            return node.value

        # Recursively traverse the left or right subtree depending on the feature value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

