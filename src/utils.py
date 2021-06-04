import cv2 as cv
import matplotlib.pyplot as plt


def plot_confusion_matrix(matrix, nClasses):
    fig = plt.figure(figsize=(nClasses, nClasses))
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    fig.colorbar(cax)
    plt.tight_layout()
    plt.savefig("plots/confusion_matrix_plot.png")


def open_image(path):
    image = cv.imread(path)

    if image is None:
        print("  !! Error reading image ", path)
        return None

    return image
