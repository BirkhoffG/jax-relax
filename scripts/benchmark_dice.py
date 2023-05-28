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

df = pd.read_csv('assets/adult/data.csv')
target = df["income"]

continuous = ["age","hours_per_week"]
categorical = ["workclass","education","marital_status","occupation","race","gender"]

train_dataset, test_dataset, y_train, y_test = train_test_split(
                                                    df,
                                                    target,
                                                    test_size=0.2,
                                                    random_state=0,
                                                    stratify=target)
x_train = train_dataset.drop('income', axis=1)
x_test = df.drop('income', axis=1)

# Dataset for training an ML model
d = dice_ml.Data(dataframe=df,
                 continuous_features=continuous,
                 outcome_name='income')

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])
transformations = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical)])

clf = Pipeline(steps=[('preprocessor', transformations),
                      ('classifier', RandomForestClassifier())])
model = clf.fit(x_train, y_train)

# Using sklearn backend
m = dice_ml.Model(model=model, backend="sklearn")
# Using method=random for generating CFs
exp = dice_ml.Dice(d, m, method="random")

start_time = time.time()
exp.generate_counterfactuals(x_test[:], total_CFs=x_test.shape[0], desired_class="opposite")
total_time = time.time() - start_time

print(total_time)
