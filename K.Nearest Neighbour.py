import numpy as np
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k  # Correct indentation and assignment

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
            R_indices = np.argsort(distances)[:self.k]  # Correctly use 'self.k'
            K_nearest_labels = [self.y_train[i] for i in R_indices]
            most_common = Counter(K_nearest_labels)
            prediction = most_common.most_common(1)[0][0]  # Efficiently get the most common label
            predictions.append(prediction)
        return np.array(predictions)


if __name__ == "__main__":  # Correct syntax for conditional execution

    X_train = np.array([[1, 2], [1.5, 2.5], [5, 8], [8, 8], [1, 8], [9, 11]])
    y_train = np.array([0, 0, 1, 1, 0, 1])
    knn = KNN(k=3)
    knn.fit(X_train, y_train)

    X_test = np.array([[5, 7], [3, 4], [8, 9], [0, 0]])

    predictions = knn.predict(X_test)
    print("Predictions:", predictions)
