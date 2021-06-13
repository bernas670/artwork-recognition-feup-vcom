import cv2 as cv
import numpy as np
from vocabulary import Vocabulary


class FeatureExtractor:
    def __init__(self, vocab: Vocabulary, detector=cv.KAZE_create(), matcher=cv.FlannBasedMatcher()):
        self.vocab = vocab
        self.detector = detector
        self.bow_extractor = cv.BOWImgDescriptorExtractor(detector, matcher)
        self.bow_extractor.setVocabulary(vocab.words)

    def featurize(self, img):
        keypoints = self.detector.detect(img, None)
        bow = self.bow_extractor.compute(img, keypoints)

        if bow is not None:
            return bow.squeeze()

        # TODO: check how to deal with images without features
        return np.zeros((self.vocab.size))
