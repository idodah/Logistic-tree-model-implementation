from collections import Counter
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd


class Node:
    root = None

    def __init__(self, x, y, min_leaf=5, class_weight=None, max_depth=5, depth=0):
        self.num_of_total_samples = len(x.index)
        self.col_names = list(x.columns)
        self.criterion = None
        self.left, self.right = None, None
        self.parent = None
        self.num_of_class_samples = Counter(y)
        self.x, self.y = x, y
        self.max_depth = max_depth
        self.depth = depth
        if (len(np.unique(list(y))) > 1):
            self.lr = LogisticRegression().fit(x, list(y))
        self.min_leaf = min_leaf

    # a recursive method that builds the tree.
    # this method splits each node by the maximum gini impurity the node can get.
    def grow_tree(self):
        if (self.parent == None):
            Node.root = self
        class_cnt = list(self.num_of_class_samples.values())
        if (len(class_cnt) <= 1):
            return

        max_gini = self.gini_impurity(class_cnt[0], class_cnt[1])
        best_l, best_r = [], []
        for i in self.col_names:
            val, l, r = self.find_best_split(i)
            gini = self.gini_impurity(len(l), len(r))
            if (gini > max_gini):
                max_gini = gini
                best_r, best_l = r, l
                self.criterion = (i, val)

        xl = pd.DataFrame(self.x, index=best_l)
        yl = pd.Series([int(self.y[j]) for j in best_l], index=[int(j) for j in best_l])
        xr = pd.DataFrame(self.x, index=best_r)
        yr = pd.Series([int(self.y[j]) for j in best_r], index=[int(j) for j in best_r])

        if (len(xl) >= self.min_leaf and len(xr) >= self.min_leaf):
            self.depth += 1
            if (self.depth >= self.max_depth):
                return
            else:
                self.left = Node(xl, yl, depth=self.depth, max_depth=self.max_depth, min_leaf=self.min_leaf)
                self.left.parent = self
                self.right = Node(xr, yr, depth=self.depth, max_depth=self.max_depth, min_leaf=self.min_leaf)
                self.right.parent = self
                self.left.grow_tree()
                self.right.grow_tree()
        else:
            return

    # a method that gets a name of a feature and returns the value of the feature that
    # gives the largest gini value and 2 lists of left child and right child indexes.
    def find_best_split(self, var_idx):
        col = self.x[var_idx]
        unique_vals = col.unique()
        max_gini = -1
        best_lhs, best_rhs = [], []
        for i in range(len(unique_vals)):
            lhs, rhs = [], []
            for j in zip(col.index, col):
                if (j[1] <= unique_vals[i]):
                    lhs.append(j[0])
                else:
                    rhs.append(j[0])
            new_gini = self.get_gini_gain(lhs, rhs)
            if new_gini > max_gini:
                max_gini = new_gini
                best_lhs, best_rhs = lhs, rhs
                max_val = unique_vals[i]
        return (max_val, best_lhs, best_rhs)

    # a method that gets 2 lists of the indexes of the samples in each child node
    # and returns the gini gain of the parent node after the split
    def get_gini_gain(self, lhs, rhs):
        gini_before = self.gini_impurity(len(lhs), len(rhs))
        p_left = len(lhs) / (len(lhs) + len(rhs))
        p_right = len(rhs) / (len(lhs) + len(rhs))

        left_cnt = list(Counter([self.y[i] for i in lhs]).values())
        right_cnt = list(Counter([self.y[i] for i in rhs]).values())

        if (len(left_cnt) == 1 and len(right_cnt) == 1):
            return gini_before
        elif (len(left_cnt) == 1 or len(left_cnt) == 0):
            return gini_before - (p_right * self.gini_impurity(right_cnt[0], right_cnt[1]))
        elif (len(right_cnt) == 0 or len(right_cnt) == 1):
            return gini_before - (p_left * self.gini_impurity(left_cnt[0], left_cnt[1]))
        else:
            return gini_before - (p_left * self.gini_impurity(left_cnt[0], left_cnt[1]) + p_right * self.gini_impurity(
                right_cnt[0], right_cnt[1]))

    # a method that checks if a node is a leaf
    def is_leaf(self):
        return True if self.left == None and self.right == None else False

    # a method that gets a dataframe that contains all the samples in a node
    # and returns a list of the predictions of each row in the dataframe using the predict_row method.
    def predict(self, x):
        return [self.predict_row(row) for index, row in x.iterrows()]

    # a method that gets a sample and returns its prediction by traversing the tree.
    # if the leaf node has only samples from one class return this class
    # else we use the logistic regression model we defined in init to predict the sample class.
    def predict_row(self, xi):
        node = Node.root
        while (node != None):
            if (node.is_leaf()):
                num_of_classes_samples = Counter(node.y)
                if (len(num_of_classes_samples.keys()) == 1):
                    return list(num_of_classes_samples.keys())[0]
                return node.lr.predict([xi])[0]
            else:
                if (xi[node.criterion[0]] <= node.criterion[1]):
                    node = node.left
                else:
                    node = node.right

    # a static method that calculates the gini impurity of a node given the number of samples from each class
    @staticmethod
    def gini_impurity(y1_count, y2_count):
        return (1 - (math.pow(y1_count / (y1_count + y2_count), 2) + math.pow(y2_count / (y1_count + y2_count), 2)))
