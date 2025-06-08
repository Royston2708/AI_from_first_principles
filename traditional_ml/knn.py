import numpy as np
import math
from collections import Counter


class KNN:
    def __init__(self, k=3) -> None:
        self.k = k

    def fit(self, X, y): 
        self.X_train = X 
        self.y_train = y 


    def predict(self, X):
        predictions = [self._predict_label(x) for x in X]

        return predictions
    

    def _predict_label(self, x):
        """
        Given a particular input value we want to return the label using K nearest neighbors
        """
        # Use Euclidean Distance
        distances = [math.dist(x, i) for i in self.X_train]

        # Get closest K 
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [int(self.y_train[i]) for i in k_indices]

        # Final Label
        final_label = Counter(k_nearest_labels).most_common()

        return final_label[0][0]



if __name__ == "__main__":
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report

    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=27)

    clf = KNN(k=5)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    print(predictions)
    print(classification_report(predictions, y_test))

