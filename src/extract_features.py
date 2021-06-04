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
from utils import open_image

img_dir = "data/images"
out_dir = "output"

if __name__ == '__main__':
    data = pd.read_csv("data/multiclass.csv")

    img_ids = []
    labels = []
    for _, row in data.iterrows():
        img_ids.append(row['id'])
        labels.append(row['attribute_ids'])

    image_paths = [os.path.join(img_dir, '{}.png'.format(id))
                   for id in img_ids]

    n_words = 1000
    existing_descriptors = None

    # Check if descriptor files exists
    if os.path.exists(os.path.join(out_dir, f'descriptors.p')):
        with open(os.path.join(out_dir, f'descriptors.p'), 'rb') as in_file:
            existing_descriptors = pickle.load(in_file)

    # Check if vocab files exists
    if os.path.exists(os.path.join(out_dir, f'vocab_{n_words}.npy')):

        with open(os.path.join(out_dir, f'vocab_{n_words}.npy'), 'rb') as in_file:
            vocab = Vocabulary.load(in_file)
    else:
        # Create vocabulary
        vocab = Vocabulary(n_words)
        descriptors = vocab.train(image_paths, existing_descriptors)
        # Save vocabulary
        vocab.save(os.path.join(out_dir, f'vocab_{n_words}.npy'))

    # Save descriptors
    if not existing_descriptors:
        with open(os.path.join(out_dir, f'descriptors.p'), 'wb') as out_file:
            pickle.dump(descriptors, out_file)
    del existing_descriptors

    # Featurize images using BOW
    featurizer = FeatureExtractor(vocab)
    features = []
    for path in tqdm(image_paths, desc="Extracting BoW feature vector"):
        img = open_image(path)
        bow = featurizer.featurize(img)
        features.append(bow)

    dataset = {
        'ids': img_ids,
        'X': features,
        'y': labels,
        'vocab_size': n_words,
    }

    with open(os.path.join(out_dir, f'processed_{n_words}.p'), 'wb') as out_file:
        pickle.dump(dataset, out_file)
