#!/usr/bin/env python3
'''Module that creates decision trees'''

import numpy as np


class Node:
    '''Class of nodes in a decision tree'''
    def __init__(self, feature=None, threshold=None,
                 left_child=None, right_child=None, is_root=False, depth=0):
        '''Node attributes'''
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        '''Recursively verifies if it is a node or a leaf
            to add both

        Returns:
            int: the maximun depth of the tree
        '''
        # check if it is a leaf
        if self.is_leaf is True:
            return self.depth
        # check if it is a node on the left
        if self.left_child:
            L_depth = self.left_child.max_depth_below()
        # # check if it is a node on the right
        if self.right_child:
            R_depth = self.right_child.max_depth_below()
        return max(L_depth, R_depth)

    def count_nodes_below(self, only_leaves=False):
        '''Count all the nodes internal node root and leaves

        Args:
            only_leaves (bool, optional): check if it is just
            a leaf. Defaults to False.

        Returns:
            int: total of nodes
        '''
        if only_leaves:
            count = self.left_child.count_nodes_below(only_leaves=True)
            count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        else:
            count = 1 + self.left_child.count_nodes_below()
            count += self.right_child.count_nodes_below()
            return count


class Leaf(Node):
    '''Class of leafs in a decision tree'''
    def __init__(self, value, depth=None):
        '''Initializes the leaf objects'''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        '''returns the value of depth'''
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''returns 1 when counting leaf'''
        return 1


class Decision_Tree():
    '''Class of a decision tree'''
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''Initialize the attributes of a decision tree'''
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        '''Finds the maximum depth of a tree'''
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''calls method of node to count nodes and leaves'''
        return self.root.count_nodes_below(only_leaves=only_leaves)
