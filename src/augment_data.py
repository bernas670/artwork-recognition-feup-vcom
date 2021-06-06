import re
import os
import glob

from pandas.core.frame import DataFrame
from keras_preprocessing.image import ImageDataGenerator


def update_dataframe(key, df, out_dir):
    # Care for indeces
    for path_name in glob.glob(os.path.join(out_dir, f'{key}_*.png')):
        match = re.findall(f'/({key}.+).png', path_name)[0]
        df = df.append(
            {'id': match, 'attribute_ids': key}, ignore_index=True)
    return df


def generate_images(generator, dataframe, image_dimensions, prefix, num_images, src_dir, out_dir):

    gen = generator.flow_from_dataframe(
        dataframe=dataframe,
        directory=src_dir,
        target_size=image_dimensions,
        x_col='id',
        y_col='attribute_ids',
        class_mode='categorical',
        shuffle=True,
        batch_size=1,
        save_to_dir=out_dir,
        save_format='png',
        save_prefix=prefix
    )

    for _ in range(num_images):
        gen.next()


def viagra_no_dataset(dataframe, target, image_dims, src_dir="data/images", out_dir="data/augmented"):

    augmented_dataframe = DataFrame(columns=['id', 'attribute_ids'])
    counts = dataframe['attribute_ids'].value_counts().to_dict()

    data_up_generator = ImageDataGenerator(
        horizontal_flip=True,
        brightness_range=[0.6, 1.3],
        zoom_range=[0.9, 1.0],
        width_shift_range=[-25, 25],
        height_shift_range=[-25, 25],
        rotation_range=20,
    )

    data_down_generator = ImageDataGenerator()

    for key, value in counts.items():

        # add existing images to the augmented dataset
        images_from_class = dataframe[dataframe['attribute_ids'] == key]

        generate_images(
            generator=data_down_generator,
            dataframe=images_from_class,
            image_dimensions=image_dims,
            prefix=key,
            num_images=min(target, value),
            src_dir=src_dir,
            out_dir=out_dir
        )

        if value < target:
            generate_images(
                generator=data_up_generator,
                dataframe=images_from_class,
                image_dimensions=image_dims,
                prefix=key,
                num_images=(target - value),
                src_dir=src_dir,
                out_dir=out_dir
            )

        # update dataframe with classes outputed images
        augmented_dataframe = update_dataframe(
            key, augmented_dataframe, out_dir)

    # save dataframe
    augmented_dataframe.to_csv('data/augmented_labels.csv')
    return augmented_dataframe
