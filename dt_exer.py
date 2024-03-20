class DecisionTreeClassifier:
    def __init__(self, max_depth=None):
        self.tree = None
        self.max_depth = max_depth

    def fit(self, X, y):
        """
        Build the decision tree classifier.

        Parameters:
        X (list of lists): Features matrix where each row represents a sample and each column represents a feature.
        y (list): Target labels corresponding to each sample in X.

        Returns:
        None
        """
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """
        Recursively build the decision tree.

        Parameters:
        X (list of lists): Features matrix where each row represents a sample and each column represents a feature.
        y (list): Target labels corresponding to each sample in X.

        Returns:
        dict: The decision tree represented as a nested dictionary.
        """
        # Base case: if all labels are the same or max_depth is reached, return a leaf node with that label
        if len(set(y)) == 1 or (self.max_depth is not None and depth == self.max_depth):
            majority_label = max(set(y), key=y.count)
            return {'label': majority_label}

        # Find the best split point
        best_split_index, best_split_value = self._find_best_split(X, y)

        # Split the dataset based on the best split point
        left_X, left_y, right_X, right_y = self._split_dataset(
            X, y, best_split_index, best_split_value)

        # Recursive calls to build subtrees
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        right_subtree = self._build_tree(right_X, right_y, depth + 1)

        # Create a node representing the best split
        return {'split_index': best_split_index,
                'split_value': best_split_value,
                'left': left_subtree,
                'right': right_subtree}

    def _find_best_split(self, X, y):
        """
        Find the best feature and value to split on.

        Parameters:
        X (list of lists): Features matrix where each row represents a sample and each column represents a feature.
        y (list): Target labels corresponding to each sample in X.

        Returns:
        int, float: Index of the feature to split on and the value of the split.
        """
        best_split_index = None
        best_split_value = None
        best_gini = float('inf')

        # Loop through each feature
        for feature_index in range(len(X[0])):
            # Loop through each unique value of the feature
            for value in set([sample[feature_index] for sample in X]):
                # Split the dataset based on this feature and value
                left_X, left_y, right_X, right_y = self._split_dataset(
                    X, y, feature_index, value)

                # Calculate Gini impurity for this split
                gini = self._calculate_gini(left_y, right_y)

                # Update the best split if this split is better
                if gini < best_gini:
                    best_gini = gini
                    best_split_index = feature_index
                    best_split_value = value

        return best_split_index, best_split_value

    def _split_dataset(self, X, y, split_index, split_value):
        """
        Split the dataset based on a given feature and value.

        Parameters:
        X (list of lists): Features matrix where each row represents a sample and each column represents a feature.
        y (list): Target labels corresponding to each sample in X.
        split_index (int): Index of the feature to split on.
        split_value (float): Value of the split.

        Returns:
        list of lists, list, list of lists, list: Left and right split datasets for features and labels.
        """
        left_X = []
        left_y = []
        right_X = []
        right_y = []

        # Loop through each sample
        for i in range(len(X)):
            # If the feature value is less than or equal to the split value, assign to left split
            if X[i][split_index] <= split_value:
                left_X.append(X[i])
                left_y.append(y[i])
            # Otherwise, assign to right split
            else:
                right_X.append(X[i])
                right_y.append(y[i])

        return left_X, left_y, right_X, right_y

    def _calculate_gini(self, left_y, right_y):
        """
        Calculate the Gini impurity for a split.

        Parameters:
        left_y (list): Labels for the left split.
        right_y (list): Labels for the right split.

        Returns:
        float: Gini impurity.
        """
        # Total number of samples
        total_samples = len(left_y) + len(right_y)

        # Calculate Gini impurity for the left split
        gini_left = 1 - \
            sum([(left_y.count(label) / len(left_y)) ** 2 for label in set(left_y)])

        # Calculate Gini impurity for the right split
        gini_right = 1 - \
            sum([(right_y.count(label) / len(right_y))
                ** 2 for label in set(right_y)])

        # Weighted average of the Gini impurities
        weighted_gini = (len(left_y) / total_samples) * \
            gini_left + (len(right_y) / total_samples) * gini_right

        return weighted_gini

    def predict(self, X):
        """
        Make predictions using the trained decision tree.

        Parameters:
        X (list of lists): Features matrix where each row represents a sample and each column represents a feature.

        Returns:
        list: Predicted labels for each sample.
        """
        predictions = []
        # Loop through all test samples
        # Loop through all test samples
        for sample in X:
            predictions.append(self._predict_sample(sample, self.tree))

        return predictions

    def _predict_sample(self, sample, tree):
        """
        Recursively predict the label for a single sample.

        Parameters:
        sample (list): Feature vector for a single sample.
        tree (dict): Decision tree.

        Returns:
        int: Predicted label for the sample.
        """
        if 'label' in tree:
            return tree['label']

        if sample[tree['split_index']] <= tree['split_value']:
            return self._predict_sample(sample, tree['left'])
        else:
            return self._predict_sample(sample, tree['right'])


# Initialize a DecisionTreeClassifier
clf = DecisionTreeClassifier()

# Example dataset (features matrix and labels)
X_train = [[2.0, 3.0],
           [5.0, 4.0],
           [9.0, 6.0],
           [4.0, 7.0],
           [8.0, 1.0]]

y_train = [0, 1, 0, 1, 0]  # Example binary labels

# Train the classifier
clf.fit(X_train, y_train)

# Example predictions
X_test = [[5.0, 4.0], [3.0, 2.0], [2.0, 3.0]]

predictions = clf.predict(X_test)

print("Predictions:", predictions)