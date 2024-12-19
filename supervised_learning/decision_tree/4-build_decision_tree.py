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
        # check if it is a node on the right
        if self.right_child:
            R_depth = self.right_child.max_depth_below()
        # adds the both variables values with max and returns the sum
        return max(L_depth, R_depth)

    def count_nodes_below(self, only_leaves=False):
        '''Count all the nodes internal node root and leaves

        Args:
            only_leaves (bool, optional): check if it is just
            a leaf. Defaults to False.

        Returns:
            int: total of nodes
        '''
        # verifies if its only leaves
        if only_leaves:
            # recursively checks the left and right childs
            # stores data to count variable
            count = self.left_child.count_nodes_below(only_leaves=True)
            count += self.right_child.count_nodes_below(only_leaves=True)
            return count
        # if it is not only leaves
        else:
            # counts it as one node and recursively looks at the childs
            # adds the findings to count also
            count = 1 + self.left_child.count_nodes_below()
            count += self.right_child.count_nodes_below()
            return count

    def left_child_add_prefix(self, text):
        '''Adds prefix to left child

        Args:
            text (str):prefex

        Returns:
            str: left child prefex
        '''
        # split the input text into lines
        lines = text.split("\n")
        # add a prefix to the first line(line[0])
        new_text = "    +--" + lines[0] + "\n"
        # iterate over the lines after the first
        for i, x in enumerate(lines[1:]):
            if i == len(lines[1:]) - 1:
                continue
            # add indentation and | to give continuation
            new_text += ("    |  ") + x + "\n"
        # return modified text
        return new_text

    def right_child_add_prefix(self, text):
        '''Adds prefix to left child

        Args:
            text (str):prefex

        Returns:
            str: left child prefex
        '''
        # this splits the inputs text into lines
        lines = text.split("\n")
        # add the prefix to the first line(line[0])
        new_text = "    +--" + lines[0] + "\n"
        # iterate over the next lines
        for i, x in enumerate(lines[1:]):
            if i == len(lines[1:]) - 1:
                new_text += "   " + x
            # add indentation to them
            else:
                new_text += "       " + x + "\n"
        # finally return the modified text
        return new_text

    def __str__(self):
        '''creates a visual representation of the nodes

        Returns:
            str: scheme for nodes
        '''
        # verifies if it is a root node
        if self.is_root:
            # adds this formatted string representation of the root node
            result = (
                f'root [feature={self.feature}, threshold={self.threshold}]\n'
            )
        else:
            # adds this formatted string for other nodes
            result = (
                f'-> node [feature={self.feature}, '
                f'threshold={self.threshold}]\n'
            )
        # Checks if its right or left child
        # Adds the findings to two strings left and right
        if self.left_child:
            left = self.left_child.__str__()
            # the left variable is used to give a visual
            # representation of left childs
            result += self.left_child_add_prefix(left)
        if self.right_child:
            # the right variable is used to give a visual
            # representation of right childs
            right = self.right_child.__str__()
            result += self.right_child_add_prefix(right)
        # Returns the whole tree formatted into a scheme
        return result

    def get_leaves_below(self):
        """ Method that returns the leaves below the current node """
        leaves = []
        if self.is_leaf:
            leaves.append(self)
        else:
            leaves += self.left_child.get_leaves_below()
            leaves += self.right_child.get_leaves_below()
        return leaves

    def update_bounds_below(self):
        """Update the bounds of the leaves below the current node."""
        if self.is_root:
            self.upper = {0: np.inf}
            self.lower = {0: -np.inf}

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.lower = self.lower.copy()
                child.upper = self.upper.copy()
                if child == self.left_child:
                    child.lower[self.feature] = max(
                        child.lower.get(self.feature, -np.inf), self.threshold)
                else:  # right child
                    child.upper[self.feature] = min(
                        child.upper.get(self.feature, np.inf), self.threshold)

        for child in [self.left_child, self.right_child]:
            if child is not None:
                child.update_bounds_below()


class Leaf(Node):
    '''Class of leafs in a decision tree'''
    # Initializes the object
    def __init__(self, value, depth=None):
        '''Initializes the leaf objects'''
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        '''returns the value of depth'''
        # Returns the depth of leaf
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        '''returns 1 when counting leaf'''
        # Returns 1 because its a leaf it doesnt have childs
        return 1

    def __str__(self):
        '''Returns formatted string of value attribute'''
        # returns a visual rep of the leaf
        return (f"-> leaf [value={self.value}]")

    def get_leaves_below(self):
        '''creates a list of leaves

        Returns:
            List: leaves and their value
        '''
        # gets current node in list form
        return [self]

    def update_bounds_below(self) :
        pass


class Decision_Tree():
    '''Class of a decision tree'''
    # initializes tree
    def __init__(self, max_depth=10, min_pop=1, seed=0,
                 split_criterion="random", root=None):
        '''Initialize the attributes of a decision tree'''
        # stores the bits generated of default rng
        # uses seeds specified
        self.rng = np.random.default_rng(seed)
        # if it has roots it adds the number of roots it has
        if root:
            self.root = root
        # if it does not have roots creates the first node
        # as the root
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
        # uses method in node to find the depth
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        '''calls method of node to count nodes and leaves'''
        # uses method in node to find the total of nodes and leaves
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        '''returns string of decision tree object'''
        # uses method in node to get a visual rep of tree
        return self.root.__str__()

    def get_leaves(self):
        '''calls get leaves below method
        in node class

        Returns:
            list: list of values of leaves in tree
        '''
        # Returns the list of values in leaves
        return self.root.get_leaves_below()

    def update_bounds(self) :
        self.root.update_bounds_below()
