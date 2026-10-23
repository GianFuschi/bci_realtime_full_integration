from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle

class OnlineLDA:
    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.trained = False

    def train(self, X, y):
        self.model.fit(X, y)
        self.trained = True

    def predict(self, x):
        if not self.trained:
            return None
        return int(self.model.predict([x])[0])

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)

    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.trained = True

