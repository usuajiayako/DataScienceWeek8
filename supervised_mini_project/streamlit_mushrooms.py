
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

# %%
# loading data
data = pd.read_csv("mushrooms.csv")
data.head()

# %% [markdown]
# Check data

# %%
data.describe()

# %%
data.info()

# %%
data["stalk-root"].unique()

# %% [markdown]
# ### Data Visualisation

# %%
# Edible/Poisonous
print(data["class"].value_counts())

sns.set_palette("pastel")
plt.pie(data["class"].value_counts(), labels = ["Edible", "Poisonous"], autopct = "%1.1f%%")
plt.title("Edible or Poisonous")
plt.show()

# %%
# Feature x category
feature_data = pd.DataFrame(columns = data.columns, index = range(1))

for feature in feature_data:
    feature_data[feature][0] = len(data[feature].unique())

sns.barplot(data = feature_data.iloc[:, 1: -1])
plt.xticks(rotation = 90)
plt.title("Feature x Category")
plt.show()

# %%
# One Hot Encoding function
encoder = OneHotEncoder()

def one_hot_encoder():

    new_data = pd.DataFrame(index = range(len(data.index)))

    for column in data:
        encoded_data = pd.DataFrame(encoder.fit_transform(data[[column]]).toarray())
        encoded_data.columns = encoder.get_feature_names_out()
        new_data = new_data.join(encoded_data.astype(int))

    return new_data

new_data = one_hot_encoder()

# %%
new_data.drop("class_p", axis = 1, inplace = True)
new_data.rename(columns = {"class_e": "edibility"}, inplace = True)
new_data

# %% [markdown]
# ### Logistic Regression

# %%
#define X and y
X = new_data.iloc[:, 1:]
y = new_data.iloc[:, 0]

#splitting into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

# %%
#define model
model = LogisticRegression()
model.fit(X_train, y_train)

# %%
# prediction
y_pred = model.predict(X_test)
y_pred

# %%
# confusion matrix
cm = confusion_matrix(y_test, y_pred)

# plot cm
sns.heatmap(cm/np.sum(cm), annot = True, cmap = "Blues", fmt = ".2%")
plt.xlabel("Prediction")
plt.show()

# %%
print(classification_report(y_test, y_pred))

# %%
# save model
save = pickle.dumps(model)
saved_model = open("saved_model.pkl", "wb")
pickle.dump(model, saved_model)
saved_model.close()

# %%
loaded_model =  pickle.load(open("saved_model.pkl", "rb"))
loaded_model
# %%
