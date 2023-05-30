import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from time import time
from alibi.datasets import fetch_adult
from alibi.explainers import CounterfactualProto
from alibi.utils import ohe_to_ord, ord_to_ohe
import pandas as pd
import time

def nn_ohe():
    x_in = Input(shape=(29,))
    x = Dense(60, activation='relu')(x_in)
    x = Dropout(.2)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x = Dense(60, activation='relu')(x)
    x = Dropout(.2)(x)
    x_out = Dense(2, activation='softmax')(x)

    nn = Model(inputs=x_in, outputs=x_out)
    nn.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return nn

def main():
    df = pd.read_csv("assets/adult/data.csv")
    continuous = ["age","hours_per_week"]
    categorical = ["workclass","education","marital_status","occupation","race","gender"]
    target = df["income"].values
    print("targe shape:",target.shape)

    category_map = { 0: ['Private', 'Self-Employed', 'Other/Unknown', 'Government'],
                         1: ['HS-grad', 'Some-college', 'Assoc', 'School', 'Doctorate', 'Prof-school', 'Bachelors', 'Masters'],
                         2: ['Married', 'Single', 'Divorced', 'Widowed', 'Separated'], 
                         3: ['Blue-Collar', 'White-Collar', 'Service', 'Other/Unknown', 'Professional', 'Sales'], 
                         4: ['White', 'Other'], 
                         5: ['Female', 'Male']
                        }
    
    data = df[["workclass","education","marital_status","occupation","race","gender","age","hours_per_week"]]

    cat_vars_ord = {0: 4, 4: 8, 12: 5, 17: 6, 23: 2, 25: 2}
    print(cat_vars_ord)

    X_num = df[continuous].astype(np.float32, copy=False)
    xmin, xmax = X_num.min(axis=0), X_num.max(axis=0)
    rng = (-1., 1.)
    X_num_scaled = (X_num - xmin) / (xmax - xmin) * (rng[1] - rng[0]) + rng[0]

    X_cat = df[categorical]
    ohe = OneHotEncoder(categories='auto', sparse=False).fit(X_cat)
    X_cat_ohe = ohe.transform(X_cat)
    print("check OHE", np.sum(X_cat_ohe, axis=1).sum())

    X = np.c_[X_cat_ohe, X_num_scaled].astype(np.float32, copy=False)
    
    X_train, X_test, y_train, y_test = train_test_split(
                                                        X,
                                                        target,
                                                        test_size=0.2,
                                                        random_state=0,
                                                        stratify=target
                                                        )

    nn = nn_ohe()
    nn.summary()
    nn.fit(X_train, to_categorical(y_train), batch_size=256, epochs=30, verbose=0)

    x_test = X_test[0].reshape((1,) + X[0].shape)
    shape = x_test.shape
    beta = .01
    c_init = 1.
    c_steps = 5
    max_iterations = 1000
    rng = (-1., 1.)  # scale features between -1 and 1
    rng_shape = (1,8)
    feature_range = ((np.ones(rng_shape) * rng[0]).astype(np.float32), 
                    (np.ones(rng_shape) * rng[1]).astype(np.float32))

    print("X shape", X.shape)
    print("rng shape",rng_shape)
    print("feature range",feature_range)
    print(cat_vars_ord)

    cf = CounterfactualProto(
                             nn,
                             shape,
                             beta=beta,
                             cat_vars=cat_vars_ord,
                             ohe=True,  # OHE flag
                             max_iterations=max_iterations,
                             feature_range=feature_range,
                             c_init=c_init,
                             c_steps=c_steps
                            )

    cf.fit(X_train, d_type='abdm',disc_perc=[25, 50, 75]);
    
    start_time = time.time()
    for x in X_test:
        x = x.reshape((1,) + X_test[0].shape)
        explanation = cf.explain(x)
    total_time = time.time() - start_time
    print("total time:", total_time)

if __name__ == "__main__":
    main()
