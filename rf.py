import pandas as pd
import numpy as np
# Import the DecisionTreeClassifier class from seperate file
from dt_exer import DecisionTreeClassifier


class RandomForest:
    """
    Hyperparameters:
        n_trees (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the trees.
        max_features (int): The maximum number of features to consider when looking for the best split.
        min_samples_split (int): The minimum number of samples required to split an internal node.
        min_samples_leaf (int): The minimum number of samples required to be at a leaf node.

    Methods:
        fit(X, y): Build the forest of trees from the training set (X, y).
        predict(X): Predict the class of each sample in X.
    Notes:
        Only supports binary or multi-class classification, not regression
    """

    def __init__(self, n_trees=10, max_depth=5, max_features=None, min_samples_split=2, min_samples_leaf=1):
        """
        Attributes:
        n_trees (int): The number of trees in the forest.
        max_depth (int): The maximum depth of the trees.
        max_features (int): The maximum number of features to consider when looking for the best split.
        trees (list): A list of DecisionTreeClassifier objects.
        features (list): A list of lists of the feature indices used for each tree.
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features  # must be in between 1 and n_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

        self.trees = []
        self.features = []

    def fit(self, X, y):
        """
        Build the forest of trees from the training set (X, y).

        Parameters:
            X (2D list): Features matrix where each row represents a sample and each column represents a feature.
            y (list): Target labels corresponding to each sample in X.

            Note: X cannot be a Pandas dataframe. It must be a 2D list.

        Returns:
            None
        """
        # Loop n_trees times to build each tree
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)

            # Call the DecisionTreeClassifier class to build a single tree
            tree = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                          min_samples_leaf=self.min_samples_leaf)

            # Fit the tree on a bootstrapped sample
            tree.fit(X_sample, y_sample)

            # Save the tree to self.trees
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        """
        Create a single bootstrap sample of the data.
        In other words, creates a sample of the same size as the original data 
            by sampling with replacement and with n_features sampled without replacement

        Parameters:
            X (2D list): Features matrix where each row represents a sample and each column represents a feature.
            y (list): Target labels corresponding to each sample in X.

        Returns:
            X_sample (2D list): The bootstrap sample of the features matrix.
            y_sample (list): The bootstrap sample of the target labels.
        """
        # Find the sizes of the matrix
        n_samples = len(X)
        n_features = len(X[0])

        # Determine the number of features to use
        if self.max_features is None:
            max_features = n_features
        else:
            max_features = min(self.max_features, n_features)

        # Randomly sample the row and feature indicies
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        feature_idxs = np.random.choice(
            np.arange(n_features), size=max_features, replace=False)

        # Create the bootstrap sample
        X_sample = X[idxs][:, feature_idxs]
        y_sample = np.array(y)[idxs]

        # Save the features used for the tree
        self.features.append(feature_idxs)

        return X_sample, y_sample

    def predict(self, X):
        """
        Returns a prediction for an array of samples X using the majority vote from all the trees.

        Parameters:
            X (2D list): Features matrix where each row represents a sample and each column represents a feature.

        Returns:
            predictions (list): The predicted classes of each sample.
        """
        # Make predictions from all trees using the features used for each tree
        all_predictions = [tree.predict(X[:, self.features[i]])
                           for i, tree in enumerate(self.trees)]

        # Convert to numpy array for easier usage
        all_predictions = np.array(all_predictions)

        # Calculate the most frequent class of predictions for each sample
        predictions = np.apply_along_axis(
            lambda x: np.bincount(x).argmax(), axis=0, arr=all_predictions)

        return predictions

    def visualize_forest(self):
        """
        Visualize all the trees in the forest.

        Parameters:
            None

        Returns:
            None
        """
        for i, tree in enumerate(self.trees):
            print(f"Tree {i + 1}")
            print(tree.visualize_tree(tree.tree))


# # Initialize a DecisionTreeClassifier
# clf = RandomForest(max_features=2)

# # Example dataset (features matrix and labels)
# X_train = pd.DataFrame([[2.0, 3.0],
#                         [5.0, 4.0],
#                         [9.0, 6.0],
#                         [4.0, 7.0],
#                         [8.0, 1.0]])

# y_train = [0, 1, 0, 1, 0]  # Example binary labels

# # Train the classifier
# clf.fit(X_train, y_train)

# # Example predictions
# X_test = pd.DataFrame([[5.0, 4.0], [3.0, 2.0], [2.0, 3.0]])

# predictions = clf.predict(X_test)

# print("Predictions:", predictions)
