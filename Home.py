
# Module 2: Introduction to TensorFlow - https://colab.research.google.com/dri...
# Module 3: Core Learning Algorithms - https://colab.research.google.com/dri...
# Module 4: Neural Networks with TensorFlow - https://colab.research.google.com/dri...
# Module 5: Deep Computer Vision - https://colab.research.google.com/dri...
# Module 6: Natural Language Processing with RNNs -  https://colab.research.google.com/dri...
# Module 7: Reinforcement Learning -  https://colab.research.google.com/dri...

# from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from six.moves import urllib

import tensorflow.compat.v2.feature_column as fc

#Make sure tf is on the correct version
print(tf.version)


# Load dataset
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv') # training data
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv') # testing data

#.pop() removes a part of the data set
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')

#.head() shows the first five data points in the data set
print(dftrain.head()) 
# print(y_train)

#LINEAR REGRESSION

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

#Need to make feature columns for linear regression
feature_columns = []

for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

print(feature_columns)


#Creating the input function for training the data.
def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)