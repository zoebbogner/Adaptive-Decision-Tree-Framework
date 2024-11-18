import numpy as np
import matplotlib.pyplot as plt

# ID's : 
# Zoe Bogner : 322374570
# Oz Dekel : 318930708

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 0.0
    
    # Separate the labels from the data
    labels = data[:, -1]
    
    _,counts = np.unique(labels,return_counts=True)
    prob = counts/len(data)
    gini = 1-np.sum(prob**2)
    
    return gini                          


def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0

    labels = data[:, -1]
    _,counts = np.unique(labels,return_counts=True)
    prob = counts/len(data)
    entropy = -1*np.sum(prob*np.log2(prob))
    
    return entropy   
    

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0

    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        
        unique,counts = np.unique(self.data.T[-1],return_counts =True)
        dict_key = dict(zip(unique,counts))
        pred = max(dict_key,key = dict_key.get)
        
        return pred
        
        
    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        
        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        
        prob = len(self.data)/n_total_sample
        goodness,_ = self.goodness_of_split(feature=self.feature)
        self.feature_importance = prob * goodness
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
       
        split_Information = 0
        children_impurity=0
        
        # Calculate the current impurity function
        if self.gain_ratio :
            self.impurity_func= calc_entropy
            
        current_impurity = self.impurity_func(self.data)
        unique = np.unique(self.data.T[feature])
        
        # Calculate the impurity of the children
        for instances in unique:
            
            # Split the data according to the feature values
            groups[instances] = self.data[self.data[:,feature]==instances]
            prob = len(groups[instances])/len(self.data)
            
            # Calculate the impurity of the children
            children_impurity += prob*self.impurity_func(groups[instances])
            # In case the gain_ratio is true
            split_Information+= -prob*np.log2(prob)
            
        goodness = current_impurity - children_impurity
        
        if self.gain_ratio :
            
            if split_Information == 0:
                return 0,groups
            
            goodness = goodness/split_Information
            
        return goodness,groups
        
    
    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        
        if self.depth == self.max_depth:
            self.terminal = True
            return
        
        
        num_features = len(self.data[0])-1
        importance = {}
        
        # Find the best feature to split according to
        for current_feature in range(num_features):
            
            self.feature = current_feature
            self.calc_feature_importance(n_total_sample=len(self.data))
            importance[current_feature] = self.feature_importance
            
        self.feature= max(importance,key = importance.get)
        goodness,groups = self.goodness_of_split(feature=self.feature)
        
        # Prune the node if it won't improve the tree
        if len(groups) == 1 or goodness == 0.0:
            self.terminal = True
            return
        
        # Prune the node according to the chi square test
        if self.prune_by_chi_square(groups=groups) == False:
            
            # Create the children of the current node
            for val, subset_data in groups.items():
                child_node = DecisionNode(data=subset_data, impurity_func=self.impurity_func,
                                    depth=self.depth + 1,
                                    chi=self.chi,
                                    max_depth=self.max_depth,
                                    gain_ratio=self.gain_ratio)
                self.add_child(child_node, val)
                
        else:
            self.terminal = True
            return


    def prune_by_chi_square(self , groups):
        """
            Determine if the node should be pruned according to the chi square test.
            
            Input:
            - groups: the data after splitting the current node.
            
            Returns:
            - True if the node should be pruned, False otherwise.
            
        """

        # If the chi value is 1, don't prune
        if self.chi == 1:
            return False
        
        # Extract the labels and their counts
        labels,counts = np.unique(self.data.T[-1] , return_counts = True)

        # Calculate the probabilities of each label
        probabilities = counts/len(self.data)

        chi_square = 0
        
        # Calculate the degrees of freedom
        Df = (len(groups) - 1)

        # Calculate the chi square statistic , by iterating over the groups
        for group_data in groups.values():
            
            # Calculate the number of instances in the group
            observed_counts = {}
            
            # Calculate the observed counts of each label in the group
            for label in labels:
                observed_counts[label] = np.sum(group_data[:, -1] == label)
            
            # Calculate the expected counts of each label in the group
            for label, observed_count in observed_counts.items():
                expected_count = len(group_data) * probabilities[np.where(labels == label)[0][0]]
                chi_square += ((observed_count - expected_count) ** 2) / expected_count

        chi_critical_value = chi_table[Df][self.chi]
        
        if chi_square > chi_critical_value:
            return False  # Don't prune
        
        else:
            return True  # Prune
        
        
class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = None
        
        self.root = DecisionNode(data=self.data, impurity_func=self.impurity_func,
                                depth=0, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        nodes_to_split = [self.root]
        num_features = len(self.data[0])-1
        
        while len(nodes_to_split) > 0:
            current_node = nodes_to_split.pop(0)
            importance = {}
            
            # Find the best feature to split according to
            for current_feature in range(num_features):
                current_node.feature = current_feature
                current_node.calc_feature_importance(n_total_sample=len(self.data))
                importance[current_feature] = current_node.feature_importance
                
            current_node.feature = max(importance, key=importance.get)
            
            # Split the current node
            current_node.split()
            
            # Add the children of the current node to the list of nodes to split
            if current_node.terminal == False:
                nodes_to_split += current_node.children
        
        

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element 
                    of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        pred = None
        
        node = self.root
        found_child = True
        while node.terminal == False and found_child == True:
            found_child = False
            
            # Find the child that corresponds to the instance
            for i,child in enumerate(node.children):
                
                if instance[node.feature] == node.children_values[i]:
                    
                    node = child
                    found_child = True
                    break
                
        pred = node.pred
        return pred
        
        

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        correct = 0
        total = len(dataset)
        if total == 0:
            return 0
    
        for instance in dataset:
            
            if self.predict(instance) == instance[-1]:
                correct += 1
                
        accuracy = correct/total
        return accuracy 
        
        
        
    def depth(self):
        return self.root.depth()
    
    def calc_depth(self):
        """
        Calculate the depth of the tree
     
        Output: the depth of the tree.
        """
        
        tree_depth = 0
        current_node = self.root
        
        if current_node is None:
            return tree_depth
        
        nodes = [current_node]
        while len(nodes) > 0:
            current_node = nodes.pop(0)
            if current_node.depth > tree_depth:
                tree_depth = current_node.depth
                
            if current_node.terminal == False:
                for child in current_node.children:
                    nodes.append(child)
        
        return tree_depth
       
        

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    best_impurity = calc_entropy
    gain_ratio = True
    
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        
        tree = DecisionTree(data=X_train, impurity_func=best_impurity, chi=1, max_depth=max_depth, gain_ratio=gain_ratio)
        tree.build_tree()
        training.append(tree.calc_accuracy(X_train))
        validation.append(tree.calc_accuracy(X_validation))
    
    return training, validation
    


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []
    best_impurity = calc_entropy
    gain_ratio = True
    
    for chi in [1, 0.5, 0.25, 0.1, 0.05, 0.0001]:
        
        tree = DecisionTree(data=X_train, impurity_func=best_impurity, chi=chi, max_depth=1000, gain_ratio=gain_ratio)
        tree.build_tree()
        chi_training_acc.append(tree.calc_accuracy(X_train))
        chi_validation_acc.append(tree.calc_accuracy(X_test))
        depth.append(tree.calc_depth())
        
    return chi_training_acc, chi_validation_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    
    n_nodes = 1
    
    if node.terminal:
        return n_nodes
    
    for child in node.children:
        n_nodes += count_nodes(child)
    return n_nodes
    






