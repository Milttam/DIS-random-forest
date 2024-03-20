from sklearn.tree import DecisionTreeClassifier


class RandomForest:
    def __init__(self, n_trees=10, max_depth=5, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.max_features = max_features  # must be in between 1 and n_features
        self.trees = []

    def fit(self, X, y):
        # Train n_trees different trees
        for _ in range(self.n_trees):
            X_sample, y_sample = self._bootstrap_sample(X, y)
            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def _bootstrap_sample(self, X, y):
        # Returns a random sample of the same size as the original dataset
        # Must be of size n_samples and use max_features for the number of features to consider
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[idxs], y[idxs]

    def predict(self, X):
        return np.mean([tree.predict(X) for tree in self.trees], axis=0)


# Initialize a DecisionTreeClassifier
clf = RandomForest()

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
