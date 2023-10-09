from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import time
import dice_ml
from dice_ml.utils import helpers  # helper functions
from tensorflow import keras
import tensorflow as tf


df = pd.read_csv('assets/adult/data.csv')
target = df["income"]

continuous = ["age","hours_per_week"]
categorical = ["workclass","education","marital_status","occupation","race","gender"]


X_train, X_test, y_train, y_test = train_test_split(
                                                    df,
                                                    target,
                                                    test_size=0.2,
                                                    shuffle=False)

X_num = X_train[continuous].astype(np.float32, copy=False)
xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
rng = (-1., 1.)
X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]

X_cat = X_train[categorical]
ohe = OneHotEncoder(categories='auto', sparse=False).fit(X_cat)
X_cat_ohe = ohe.transform(X_cat)

X_train = np.c_[X_cat_ohe, X_num_scaled].astype(np.float32, copy=False)

# Dataset for training an ML model
d = dice_ml.Data(dataframe=df,
                 continuous_features=continuous,
                 outcome_name='income')

# Fitting a dense neural network model
ann_model = keras.Sequential()
ann_model.add(keras.layers.Dense(20, input_shape=(X_train.shape[1],), kernel_regularizer=keras.regularizers.l1(0.001), activation=tf.nn.relu))
ann_model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
ann_model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])
ann_model.fit(X_train, y_train, validation_split=0.20, epochs=10, verbose=0, class_weight={0:1,1:2})

# Generate the DiCE model for explanation
m = dice_ml.Model(model=ann_model,backend='TF2',func="ohe-min-max")
# Using method=random for generating CFs
exp = dice_ml.Dice(d, m, method="gradient")

X_test = X_test.drop('income', axis=1)
# The number of instances for cf generation
num_instances = 100

start_time = time.time()
print("Start...")
dice_exp = exp.generate_counterfactuals(X_test.head(num_instances), total_CFs=5, desired_class="opposite")
total_time = time.time() - start_time

print(total_time)
print(dice_exp.visualize_as_dataframe(show_only_changes=False))
