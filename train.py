#!/usr/bin/env python
# coding: utf-8

import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Parameters
C = 1.0
n_splits = 5
output_file = f"model_C={C}.bin"

# Data preparation
leads = pd.read_csv("Leads.csv")

# Handle 'Select' values by replacing them with NaN
leads.replace("Select", np.nan, inplace=True)

# Drop columns with a high percentage of missing values (more than 30%)
missing_percentage = leads.isnull().sum() / len(leads) * 100
cols_to_drop = missing_percentage[missing_percentage > 30].index
leads.drop(cols_to_drop, axis=1, inplace=True)

# Impute missing values in numerical columns with the median
numerical_cols = list(leads.select_dtypes(include=np.number).columns)
for col in numerical_cols:
    leads[col] = leads[col].fillna(leads[col].median())

# Impute missing values in categorical columns with the mode
categorical_cols = list(leads.select_dtypes(include=object).columns)
for col in categorical_cols:
    leads[col] = leads[col].fillna(leads[col].mode()[0])


leads_full_train, leads_test = train_test_split(leads, test_size=0.2, random_state=1)


# Training function
def train(df_train, y_train, C=1.0):
    dicts = df_train[categorical_cols + numerical_cols].to_dict(orient="records")

    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    model = LogisticRegression(C=C, max_iter=1000)
    model.fit(X_train, y_train)

    return dv, model


# Prediction function
def predict(df, dv, model):
    dicts = df[categorical_cols + numerical_cols].to_dict(orient="records")

    X = dv.transform(dicts)
    y_pred = model.predict_proba(X)[:, 1]

    return y_pred


# Validation
print(f"doing validation with C={C}")
kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
scores = []
fold = 0

for train_idx, val_idx in kfold.split(leads_full_train):
    leads_train = leads_full_train.iloc[train_idx]
    leads_val = leads_full_train.iloc[val_idx]

    y_train = leads_train.Converted.values
    y_val = leads_val.Converted.values

    dv, model = train(leads_train, y_train, C=C)
    y_pred = predict(leads_val, dv, model)

    auc = roc_auc_score(y_val, y_pred)
    scores.append(auc)

    print(f"auc on fold {fold} is {auc}")
    fold += 1

print("validation results:")
print("C=%s %.3f +- %.3f" % (C, np.mean(scores), np.std(scores)))

# Training the final model
print("training the final model")
dv, model = train(leads_full_train, leads_full_train.Converted.values, C=1.0)
y_pred = predict(leads_test, dv, model)

y_test = leads_test.Converted.values
auc = roc_auc_score(y_test, y_pred)

print(f"auc={auc}")

# Save the model
with open(output_file, "wb") as f_out:
    pickle.dump((dv, model), f_out)

print(f"the model is saved to {output_file}")
