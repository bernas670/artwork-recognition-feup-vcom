from sklearn.model_selection import train_test_split
import os
import cv2 as cv
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
from vocabulary import Vocabulary
from feature_extractor import FeatureExtractor
from utils import openImage

imgDir = "data/images"
outputDir = "output"

if __name__ == '__main__':
    data = pd.read_csv("data/multiclass.csv")

    imgIds = []
    labels = []
    for _, row in data.iterrows():
        imgIds.append(row['id'])
        labels.append(row['attribute_ids'])

    imagePaths = [os.path.join(imgDir, '{}.png'.format(id)) for id in imgIds]

    # X_train, X_test, y_train, y_test = train_test_split(imagePaths, labels,
    #                                                     stratify=labels,
    #                                                     test_size=0.10)

    # labels = y_test
    # imagePaths = X_test

    # plt.hist(labels, bins=70)
    # plt.savefig('plots/hist.png')

    nWords = 100

    vocab = Vocabulary(nWords)
    vocab.train(imagePaths)
    vocab.save(os.path.join(outputDir, 'vocab.npy'))

    featurizer = FeatureExtractor(vocab)
    features = []
    for path in tqdm(imagePaths, desc="Extracting BoW feature vector"):
        img = openImage(path)
        bow = featurizer.featurize(img)
        features.append(bow)

    dataset = {
        'ids': imgIds,
        'X': features,
        'y': labels,
        'nWords': nWords,
    }

    pickle.dump(dataset, open(os.path.join(outputDir, 'processed.p'), 'wb'))
