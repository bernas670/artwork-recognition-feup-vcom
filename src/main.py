import pandas as pd
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
import pickle

if __name__ == '__main__':
    dataset = pickle.load(open("output/processed.p", "rb"))
    X = dataset['X']
    y = dataset['y']
    print(X[:2])
    print(y[:2])

    clf = SGDClassifier()

    clf.fit(X, y)
    scores = cross_val_score(clf, X, y, cv=3, scoring='accuracy')

