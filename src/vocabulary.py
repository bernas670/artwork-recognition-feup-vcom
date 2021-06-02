import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils import openImage


class Vocabulary:
    def __init__(self, nWords, words=None):
        self.words = words
        self.nWords = nWords

    def train(self, imagePaths, detector=cv.KAZE_create(), verbose=True):
        allDescriptors = []

        for path in tqdm(imagePaths, disable=not verbose, desc="Computing feature descriptors"):
            img = openImage(path)
            if img is None:
                continue
            keypoints, descriptors = detector.detectAndCompute(img, None)

            if descriptors is not None:
                allDescriptors.extend(descriptors)

        bowTrainer = cv.BOWKMeansTrainer(self.nWords)
        bowTrainer.add(np.float32(allDescriptors))

        if verbose:
            print("Clustering (k={}) ...".format(self.nWords))

        self.words = bowTrainer.cluster()

    def save(self, path):
        np.save(path, self.words)

    @classmethod
    def load(cls, path):
        words = np.load(path, allow_pickle=False)
        return cls(words.shape[0], words=words)
