
# %%
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

import pickle

data = pd.read_csv("mushrooms.csv")

encoder = OneHotEncoder()

def one_hot_encoder():

    new_data = pd.DataFrame(index = range(len(data.index)))

    for column in data:
        encoded_data = pd.DataFrame(encoder.fit_transform(data[[column]]).toarray())
        encoded_data.columns = encoder.get_feature_names_out()
        new_data = new_data.join(encoded_data.astype(int))

    return new_data

new_data = one_hot_encoder():
new_data.drop("class_p", axis = 1, inplace = True)
new_data.rename(columns = {"class_e": "edibility"}, inplace = True)
new_data

X = new_data.iloc[:, 1:]
y = new_data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred

cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm/np.sum(cm), annot = True, cmap = "Blues", fmt = ".2%")
plt.xlabel("Prediction")
plt.show()

print(classification_report(y_test, y_pred))