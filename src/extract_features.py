import os
import pandas as pd
import pickle
from tqdm import tqdm
from vocabulary import Vocabulary
from feature_extractor import FeatureExtractor
from helper.utils import open_image

img_dir = "data/images"
out_dir = "output"

if __name__ == '__main__':
    data = pd.read_csv("data/multiclass_train.csv")

    train_img_ids = []
    train_labels = []
    for _, row in data.iterrows():
        train_img_ids.append(row['id'])
        train_labels.append(row['attribute_ids'])
    
    train_image_paths = [os.path.join(img_dir, '{}.png'.format(id))
                   for id in train_img_ids]

    test_img_ids = []
    test_labels = []
    for _, row in data.iterrows():
        test_img_ids.append(row['id'])
        test_labels.append(row['attribute_ids'])

    test_image_paths = [os.path.join(img_dir, '{}.png'.format(id))
                   for id in train_img_ids]

    # 500 1000 2500
    n_words = 2500
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
        descriptors = vocab.train(train_image_paths, existing_descriptors)
        # Save vocabulary
        vocab.save(os.path.join(out_dir, f'vocab_{n_words}.npy'))

    # Save descriptors
    if not existing_descriptors:
        with open(os.path.join(out_dir, f'descriptors.p'), 'wb') as out_file:
            pickle.dump(descriptors, out_file)
    del existing_descriptors

    # Featurize images using BOW
    featurizer = FeatureExtractor(vocab)
    train_features = []
    for path in tqdm(train_image_paths, desc="Extracting train BoW feature vector"):
        img = open_image(path)
        bow = featurizer.featurize(img)
        train_features.append(bow)

    test_features = []
    for path in tqdm(test_image_paths, desc="Extracting test BoW feature vector"):
        img = open_image(path)
        bow = featurizer.featurize(img)
        test_features.append(bow)

    dataset = {
        'ids': train_img_ids,
        'X_train': train_features,
        'y_train': train_labels,
        'X_test': test_features,
        'y_test': test_labels,
        'vocab_size': n_words,
    }

    with open(os.path.join(out_dir, f'processed_{n_words}.p'), 'wb') as out_file:
        pickle.dump(dataset, out_file)
