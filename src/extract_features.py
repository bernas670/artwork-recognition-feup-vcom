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

    nWords = 1000
    existingDescriptors = None

    # Check if descriptor files exists
    if os.path.exists(os.path.join(outputDir, f'descriptors.p')):
        with open(os.path.join(outputDir, f'descriptors.p'), 'rb') as inFile:
            existingDescriptors = pickle.load(inFile)
    
    # Check if vocab files exists
    if os.path.exists(os.path.join(outputDir, f'vocab_{nWords}.npy')):

        with open(os.path.join(outputDir, f'vocab_{nWords}.npy'), 'rb') as inFile:
            vocab = Vocabulary.load(inFile)
    else:
        # Create vocabulary
        vocab = Vocabulary(nWords)
        descriptors = vocab.train(imagePaths, existingDescriptors)
        # Save vocabulary
        vocab.save(os.path.join(outputDir, f'vocab_{nWords}.npy'))
    
    # Save descriptors 
    if not existingDescriptors:
        with open(os.path.join(outputDir, f'descriptors.p'), 'wb') as outFile:
            pickle.dump(descriptors, outFile)
    del existingDescriptors
    # Featurize images using BOW
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

    with open(os.path.join(outputDir, f'processed_{nWords}.p'), 'wb') as outFile:
        pickle.dump(dataset,outFile )
