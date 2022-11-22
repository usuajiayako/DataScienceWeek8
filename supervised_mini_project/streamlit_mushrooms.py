
# %%
# importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("/Users/ayakobland/DataScienceBootCamp8/supervised_mini_project/mushrooms.csv")

encoder = OneHotEncoder()

def one_hot_encoder():

    new_data = pd.DataFrame(index = range(len(data.index)))

    for column in data:
        encoded_data = pd.DataFrame(encoder.fit_transform(data[[column]]).toarray())
        encoded_data.columns = encoder.get_feature_names_out()
        new_data = new_data.join(encoded_data.astype(int))

    return new_data

new_data = one_hot_encoder()
new_data.drop("class_p", axis = 1, inplace = True)
new_data.rename(columns = {"class_e": "edibility"}, inplace = True)

X = new_data.iloc[:, 1:]
y = new_data.iloc[:, 0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


# %%
########################### STREAMLIT ########################################
import streamlit as st

def main():
    st.title("Mushroom Edibility Test")
    st.sidebar.title("Feature/Category Selection")
    st.image("mushroom.png")
main()

# sidebar
cap_shape = st.sidebar.selectbox("Cap shape", ["bell", "conical", "convex", "flat", "knobbed", "sunken"])
cap_surface = st.sidebar.selectbox("Cap surface", ["fibrous", "grooves", "scaly", "smooth"])
cap_color = st.sidebar.selectbox("Cap color", ["brown", "buff", "cinnamon", "gray" , "green", "pink", "purple", "red", "white", "yellow"])
bruises = st.sidebar.selectbox("Bruises", ["yes", "no"]) 
odor = st.sidebar.selectbox("Odor", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
gill_attachment = st.sidebar.selectbox("Gill attachment", ["attached", "descending", "free", "notched"])
gill_spacing = st.sidebar.selectbox("Gill spacing", ["close", "crowded", "distant"])
gill_size = st.sidebar.selectbox("Gill size", ["broad", "narrow"])
gill_color = st.sidebar.selectbox("Gill_color", ["black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"])
stalk_shape = st.sidebar.selectbox("Stalk shape", ["enlarging", "tapering"])
stalk_root = st.sidebar.selectbox("Stalk root", ["bulbous", "club", "cup", "equal", "rhizomorphs", "rooted", "missing"])
stalk_surface_above_ring = st.sidebar.selectbox("Stalk surface above ring", ["fibrous", "scaly", "silky", "smooth"])
stalk_surface_below_ring = st.sidebar.selectbox("Stalk surface below ring", ["fibrous", "scaly", "silky", "smooth"])
stalk_color_above_ring = st.sidebar.selectbox("Stalk color above ring", ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"])
stalk_color_below_ring = st.sidebar.selectbox("Stalk color below ring", ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"])
veil_type = st.sidebar.selectbox("Veil type", ["partial", "universal"])
veil_color = st.sidebar.selectbox("Veil color", ["brown", "orange", "white", "yellow"])
ring_number = st.sidebar.selectbox("Ring number", ["none", "one", "two"])
ring_type = st.sidebar.selectbox("Ring type", ["cobwebby", "evanescent", "flaring", "large", "none", "pendant", "sheathing", "zone"])
spore_print_color = st.sidebar.selectbox("Spore print color", ["black", "brown", "buff", "chocolate", "green", "orange", "purple", "white", "yellow"])
population = st.sidebar.selectbox("Population", ["abundant", "clustered", "numerous", "scattered", "several", "solitary"])
habitat = st.sidebar.selectbox("Habitat", ["grasses", "leaves", "meadows", "paths", "urban", "waste", "woods"])

result = [cap_shape, cap_surface, cap_color, bruises, odor, gill_attachment, gill_spacing, gill_size, gill_color, stalk_shape, stalk_root, stalk_surface_above_ring, stalk_surface_below_ring, stalk_color_above_ring, stalk_color_below_ring, veil_type, veil_color, ring_number, ring_type, spore_print_color, population, habitat]

# Creating test dataframe
test_data = pd.DataFrame(columns = new_data.columns, index = ["Test data"])
test_data = test_data.fillna(0)

# Filling the result
# cap_shape
if cap_shape == "bell": test_data["cap-shape_b"] = 1
if cap_shape == "conical": test_data["cap-shape_c"] = 1
if cap_shape == "convex": test_data["cap-shape_x"] = 1
if cap_shape == "flat": test_data["cap-shape_f"] = 1
if cap_shape == "knobbed": test_data["cap-shape_k"] = 1
if cap_shape == "sunken": test_data["cap-shape_s"] = 1

# cap_surface
if cap_surface == "fibrous": test_data["cap-surface_f"] = 1
if cap_surface == "grooves": test_data["cap-surface_g"] = 1
if cap_surface == "scaly": test_data["cap-surface_y"] = 1
if cap_surface == "smooth": test_data["cap-surface_s"] = 1

# cap_color
if cap_color == "brown": test_data["cap-color_n"] = 1
if cap_color == "buff": test_data["cap-color_b"] = 1
if cap_color == "cinnamon": test_data["cap-color_c"] = 1
if cap_color == "gray": test_data["cap-color_g"] = 1
if cap_color == "green": test_data["cap-color_r"] = 1
if cap_color == "pink": test_data["cap-color_p"] = 1
if cap_color == "purple": test_data["cap-color_u"] = 1
if cap_color == "red": test_data["cap-color_e"] = 1
if cap_color == "white": test_data["cap-color_w"] = 1
if cap_color == "yellow": test_data["cap-color_y"] = 1

# bruises
if bruises == "yes": test_data["bruises_t"] = 1
if bruises == "no": test_data["bruises_f"] = 1

st.write(test_data)
st.write(result[0])


