import pickle
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn import svm

from utils import plot_confusion_matrix


def load_dataset(path):
    dataset = pickle.load(open(path, "rb"))
    X, y = dataset['X'], dataset['y']

    return X, y


def train_classifier(classifier, X, y):
    cv = cross_validate(
        classifier, X, y, cv=5, return_train_score=True,
        scoring=['accuracy', 'f1_macro', 'f1_micro'], n_jobs=-2,
    )

    for metric, scores in sorted(cv.items()):
        if (metric.endswith('time')):
            print('{}: {:.3} +- {:.2}'.format(metric,
                                              np.mean(scores), np.std(scores)))
        else:
            print('{}: {:.3%} +- {:.2%}'.format(metric,
                  np.mean(scores), np.std(scores)))
        print('->', scores, end='\n\n')


# def grid_search():
#     param_grid = {
#     }
#     grid = GridSearchCV(
#         estimator=estimator, param_grid=param_grid,
#         cv=5, return_train_score=True,
#         scoring=['accuracy', 'f1_macro', 'f1_micro'],
#         n_jobs=-2,
#     results = grid.fit(X, y)


if __name__ == '__main__':
    X, y = load_dataset("output/processed_1000.p")
    num_classes = len(set(y))

    clf = svm.SVC()

    train_classifier(clf, X, y)

    y_pred = cross_val_predict(clf, X, y)
    conf_mx = confusion_matrix(y, y_pred)
    plot_confusion_matrix(conf_mx, num_classes)
