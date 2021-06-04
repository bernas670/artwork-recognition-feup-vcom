from os import error
import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils import open_image


class Vocabulary:
    def __init__(self, n_words, words=None):
        self.words = words
        self.size = n_words

    def train(self, image_paths=None, descriptors=None, detector=cv.KAZE_create(), verbose=True):
        if image_paths is None and descriptors is None:
            raise Exception('One of image_paths or descriptors needs to be specified')

        all_descriptors = descriptors
        if descriptors is None:
            all_descriptors = self.extract_descriptors(
                image_paths, detector, verbose=verbose)

        bow_trainer = cv.BOWKMeansTrainer(self.size)
        bow_trainer.add(np.float32(all_descriptors))

        if verbose:
            print("Clustering (k={}) ...".format(self.size))

        self.words = bow_trainer.cluster()

        return all_descriptors

    def extract_descriptors(self, image_paths, detector, verbose=True):
        all_descriptors = []
        for path in tqdm(image_paths, disable=not verbose, desc="Computing feature descriptors"):
            img = open_image(path)

            if img is None:
                continue
            keypoints, descriptors = detector.detectAndCompute(img, None)

            if descriptors is not None:
                all_descriptors.extend(descriptors)
        return all_descriptors

    def save(self, path):
        np.save(path, self.words)

    @classmethod
    def load(cls, path):
        words = np.load(path, allow_pickle=False)
        return cls(words.shape[0], words=words)
