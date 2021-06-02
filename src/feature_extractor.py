import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils import openImage
from vocabulary import Vocabulary


class FeatureExtractor:
    def __init__(self, vocab: Vocabulary, detector=cv.KAZE_create(), matcher=cv.FlannBasedMatcher()):
        self.vocab = vocab
        self.detector = detector
        self.bowExtractor = cv.BOWImgDescriptorExtractor(detector, matcher)
        self.bowExtractor.setVocabulary(vocab.words)

    def featurize(self, img):
        keypoints = self.detector.detect(img, None)
        bow = self.bowExtractor.compute(img, keypoints)

        if bow is not None:
            return bow.squeeze()

        # TODO: check how to deal with images without features
        return np.zeros((self.vocab.nWords))
