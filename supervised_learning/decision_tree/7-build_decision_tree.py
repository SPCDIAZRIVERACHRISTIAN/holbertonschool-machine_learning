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

    def update_indicator(self):
        '''This function computes the indicator function
            from the node.lower and upper dictionaries and
            stores it in a attribute
        '''

        def is_large_enough(x):
            '''computes if the lower bounds of a leaf are
                large enough.

            Args:
                x (dict): values of bounds
            '''
            # creates a numpy array with the values of x
            # uses list comprehension to get every element in x
            # if i isn't found on the dictionary it assigns an -inf value
            low_bounds = np.array([self.lower.get(
                i, -np.inf) for i in range(x.shape[1])])
            # verifies if x greater or equal than the bounds in the columns
            return np.all(x >= low_bounds, axis=1)

        def is_small_enough(x):
            '''computes if the lower bounds of a leaf are
                small enough.

            Args:
                x (dict): values of bounds

            Returns:
                bool: true if bounds are small enough
            '''
            # creates a numpy array with the values of x
            # uses list comprehension to get every element in x
            # if i isn't found on the dictionary it assigns an -inf value
            high_bounds = np.array([self.upper.get(
                i, np.inf) for i in range(x.shape[1])])
            # verifies if x less or equal than the bounds in the columns
            return np.all(x <= high_bounds, axis=1)
        # assigns in list form the booleans found for each bound
        self.indicator = lambda x: np.all(np.array(
            [is_large_enough(x), is_small_enough(x)]), axis=0)

    def pred(self, x):
        '''makes a prediciton for a simple data sample'''
        if x[self.feature] > self.threshold:
            return self.left_child.pred(x)
        else:
            return self.right_child.pred(x)


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

    def update_bounds_below(self):
        '''Does nothing! :)'''
        pass

    def pred(self, x):
        '''returns value of leaf'''
        return self.value


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

    def update_bounds(self):
        '''Calls node method to update bounds'''
        self.root.update_bounds_below()

    def update_predict(self):
        '''returns a list of predictions of leaves'''
        # update the bounds of all nodes in tree
        self.update_bounds()
        # get a list of leaves
        leaves = self.get_leaves()
        # iterate over the list
        for leaf in leaves:
            # verify if the bounds are good enough
            leaf.update_indicator()
        # assign a np array to prepare the decision tree to predict
        self.predict = lambda A: np.array([self.pred(x) for x in A])

    def pred(self, x):
        '''Predict a target for a simple x'''
        return self.root.pred(x)

    def fit(self, explanatory, target, verbose=0):
        '''This function makes our tree trainable'''
        # if the criteria is set to random
        if self.split_criterion == "random":
            # assign the split criteria to random function
            self.split_criterion = self.random_split_criterion
        # otherwise use a gini impurity criterion
        else:
            self.split_criterion = self.Gini_split_criterion
        # initialize the explanatory & target attributes
        self.explanatory = explanatory
        self.target = target
        # make an array with the same shape as target of 1's
        self.root.sub_population = np.ones_like(self.target, dtype='bool')

        # to be defined
        self.fit_node(self.root)
        # returns a prediction of the tree
        self.update_predict()
        # if the verbose = 1 print that its done training
        if verbose == 1:
            print(f"  Training finished.")
            print(f"    - Depth                     : {self.depth()}")
            print(f"    - Number of nodes           : {self.count_nodes()}")
            print(f"    - Number of leaves          : "
                  f"{self.count_nodes(only_leaves=True)}")
            print(f"    - Accuracy on training data : "
                  f"{self.accuracy(self.explanatory,self.target)}")

    def np_extrema(self, arr):
        '''Returns the minimum and maximum of an array'''
        return np.min(arr), np.max(arr)

    def random_split_criterion(self, node):
        '''criteria to split randomly a node

        Args:
            node (list): node from the decision tree

        Returns:
            list: return feature and threshold
        '''
        # set to ensure the feature
        # selected has a non-zero range
        diff = 0
        while diff == 0:
            # select a rando feature
            feature = self.rng.integers(0, self.explanatory.shape[1])
            # calculate the min and max values of feature
            feature_min, feature_max = self.np_extrema(
                self.explanatory[:, feature][node.sub_population])
            # calculate the range of the feature
            diff = feature_max - feature_min
        # generate random threshold for the feature
        x = self.rng.uniform()
        threshold = (1 - x) * feature_min + x * feature_max
        return feature, threshold

    def fit_node(self, node):
        '''

        Args:
            node (_type_): _description_
        '''
        # Extract the sub populaiton and depth
        sub_pop = node.sub_population
        n_sub = sub_pop.sum()
        depth = node.depth
        # check if it should be a leaf node
        if (
            n_sub < self.min_pop
            or depth >= self.max_depth
            or np.unique(self.target[sub_pop]).size == 1
        ):
            # convert node to leaf
            self.make_leaf(node)
            return
        # determine the split criterion
        node.feature, node.threshold = self.split_criterion(node)
        # indicates which samples fall in left child
        left_population = node.sub_population & (
            self.explanatory[:, node.feature] > node.threshold)
        # indicates which nodes fall into right child
        right_population = node.sub_population & (~left_population)
        # both are based on the split criterion

        # this blocks check if left or right childs are empty
        # if they are the node is converted to a leaf with the value
        # of the most common target value in sub population
        if (not np.any(left_population)) or (not np.any(right_population)):
            leaf_child = self.get_leaf_child(node, sub_pop)
            node.is_leaf = True
            node.feature = None
            node.threshold = None
            node.value = leaf_child.value
            return

        # check if left child node should be a leaf
        is_left_leaf = (
            left_population.sum() < self.min_pop
            or depth + 1 >= self.max_depth
            or np.unique(self.target[left_population]).size == 1
        )

        if is_left_leaf:
            # if its a leaf creates a leaf child node
            node.left_child = self.get_leaf_child(node, left_population)
        else:
            # if not it creates a new node and recursively calls method
            node.left_child = self.get_node_child(node, left_population)
            self.fit_node(node.left_child)

        # check if right child node should be a leaf
        is_right_leaf = (
            right_population.sum() < self.min_pop
            or depth + 1 >= self.max_depth
            or np.unique(self.target[right_population]).size == 1
        )

        if is_right_leaf:
            # if its a leaf creates a leaf child node
            node.right_child = self.get_leaf_child(node, right_population)
        else:
            # if not it creates a new node and recursively calls method
            node.right_child = self.get_node_child(node, right_population)
            self.fit_node(node.right_child)

    def get_leaf_child(self, node, sub_population):
        sub_targets = self.target[sub_population]
        value = np.bincount(sub_targets).argmax()
        leaf_child = Leaf(value)
        leaf_child.depth = node.depth + 1
        leaf_child.sub_population = sub_population
        return leaf_child

    def get_node_child(self, node, sub_population):
        n = Node()
        n.depth = node.depth + 1
        n.sub_population = sub_population
        return n

    def accuracy(self, test_explanatory, test_target):
        return np.sum(np.equal(
            self.predict(test_explanatory), test_target)) / test_target.size
