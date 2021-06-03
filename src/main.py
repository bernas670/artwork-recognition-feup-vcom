import pandas as pd
from sklearn.utils import shuffle
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn import svm
import pickle
import matplotlib.pyplot as plt

from utils import plot_confusion_matrix

if __name__ == '__main__':
    dataset = pickle.load(open("output/processed.p", "rb"))
    X = dataset['X']
    y = dataset['y']

    used_classes = [51, 13, 70, 69]
    X = [xi for xi, yi in zip(X, y) if yi in used_classes]
    y = [yi for yi in y if yi in used_classes]

    # https://scikit-learn.org/stable/modules/svm.html
    clf = svm.LinearSVC()
    #scores = cross_val_score(clf, X, y, cv=7, scoring='accuracy')

    y_pred = cross_val_predict(clf, X, y)

    conf_mx = confusion_matrix(y, y_pred)
    plot_confusion_matrix(conf_mx, len(used_classes))

