import os
import pandas as pd
from glob import glob

if __name__ == '__main__':
    df_imgs = set(pd.read_csv('data/multilabel_small.csv')['id'])
    img_paths = set(glob('data/images-task2-small/*.png'))

    print("Images in df: ", len(df_imgs))
    print("Total images: ", len(img_paths))
    count = 0

    for img_path in img_paths:

        id = img_path[img_path.rfind('/')+1:-4]

        if id not in df_imgs:
            os.remove(img_path)
            count += 1

    print("Images to remove: ", count)
    print(len(df_imgs) + count)
