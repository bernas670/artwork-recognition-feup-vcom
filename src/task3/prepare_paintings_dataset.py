import pandas as pd
import glob
import os
import uuid
from PIL import Image

dataframe = pd.read_csv('training_set copy.csv')

def remove_invalid_images():

  for path_name in glob.glob('data/paintings_dataset/training_set/painting/*'):

    try:
      im = Image.open(path_name)
      im.thumbnail((128, 128))
    except:
        name =  os.path.splitext(os.path.basename(path_name))[0]
        dataframe = dataframe.drop(dataframe[dataframe['id'] == name].index)
  dataframe.to_csv('training_set.csv')
  dataframe = pd.read_csv('validation_set copy.csv')

  for path_name in glob.glob('data/paintings_dataset/validation_set/painting/*'):

    try:
      im = Image.open(path_name)
      im.thumbnail((128, 128))
    except:
      name =  os.path.splitext(os.path.basename(path_name))[0]
      dataframe = dataframe.drop(dataframe[dataframe['id'] == name].index)

  dataframe.to_csv('validation_set.csv')


def create_paintings_v_sculptures_dataset():
  dataframe = pd.DataFrame(columns=['id', 'attribute_ids'])

  # Care for indeces
  for path_name in glob.glob('data/paintings_dataset/training_set/painting/*'):
    filename = str(uuid.uuid4())
    os.rename(path_name, os.path.join(os.path.dirname(path_name),  filename + '.jpg'))
    dataframe = dataframe.append({'id': filename, 'attribute_ids': '1'}, ignore_index=True)

  for path_name in glob.glob('data/paintings_dataset/training_set/sculpture/*'):
    filename = str(uuid.uuid4())
    os.rename(path_name, os.path.join(os.path.dirname(path_name),  filename + '.jpg'))
    dataframe = dataframe.append({'id': filename, 'attribute_ids': '2'}, ignore_index=True)

  dataframe.to_csv('training_set.csv', index=False)

  dataframe = pd.DataFrame(columns=['id', 'attribute_ids'])

  for path_name in glob.glob('data/paintings_dataset/validation_set/painting/*'):
    filename = str(uuid.uuid4())
    os.rename(path_name, os.path.join(os.path.dirname(path_name),  filename + '.jpg'))
    dataframe = dataframe.append({'id': filename, 'attribute_ids': '1'}, ignore_index=True)

  for path_name in glob.glob('data/paintings_dataset/validation_set/sculpture/*'):
    filename = str(uuid.uuid4())
    os.rename(path_name, os.path.join(os.path.dirname(path_name),  filename + '.jpg'))
    dataframe = dataframe.append({'id': filename, 'attribute_ids': '2'}, ignore_index=True)

  dataframe.to_csv('validation_set.csv', index=False)