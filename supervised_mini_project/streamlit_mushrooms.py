
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
    st.sidebar.title("Mushroom description")
    st.image("mushroom.png")
main()

# sidebar
cap_shape = st.sidebar.selectbox("Cap shape", ["bell", "conical", "convex", "flat", "knobbed", "sunken"])
cap_surface = st.sidebar.selectbox("Cap surface", ["fibrous", "grooves", "scaly", "smooth"])
cap_color = st.sidebar.selectbox("Cap color", ["brown", "buff", "cinnamon", "gray" , "green", "pink", "purple", "red", "white", "yellow"])
bruises = st.sidebar.selectbox("Bruises", ["yes", "no"]) 
odor = st.sidebar.selectbox("Odor", ["almond", "anise", "creosote", "fishy", "foul", "musty", "none", "pungent", "spicy"])
gill_attachment = st.sidebar.selectbox("Gill attachment", ["attached", "free"])
gill_spacing = st.sidebar.selectbox("Gill spacing", ["close", "crowded"])
gill_size = st.sidebar.selectbox("Gill size", ["broad", "narrow"])
gill_color = st.sidebar.selectbox("Gill color", ["black", "brown", "buff", "chocolate", "gray", "green", "orange", "pink", "purple", "red", "white", "yellow"])
stalk_shape = st.sidebar.selectbox("Stalk shape", ["enlarging", "tapering"])
stalk_root = st.sidebar.selectbox("Stalk root", ["bulbous", "club", "equal", "rooted", "missing"])
stalk_surface_above_ring = st.sidebar.selectbox("Stalk surface above ring", ["fibrous", "scaly", "silky", "smooth"])
stalk_surface_below_ring = st.sidebar.selectbox("Stalk surface below ring", ["fibrous", "scaly", "silky", "smooth"])
stalk_color_above_ring = st.sidebar.selectbox("Stalk color above ring", ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"])
stalk_color_below_ring = st.sidebar.selectbox("Stalk color below ring", ["brown", "buff", "cinnamon", "gray", "orange", "pink", "red", "white", "yellow"])
veil_type = st.sidebar.selectbox("Veil type", ["partial"])
veil_color = st.sidebar.selectbox("Veil color", ["brown", "orange", "white", "yellow"])
ring_number = st.sidebar.selectbox("Ring number", ["none", "one", "two"])
ring_type = st.sidebar.selectbox("Ring type", ["evanescent", "flaring", "large", "none", "pendant"])
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

# odor
if odor == "almond": test_data["odor_a"] = 1
if odor == "anise": test_data["odor_l"] = 1
if odor == "creosote": test_data["odor_c"] = 1
if odor == "fishy": test_data["odor_y"] = 1
if odor == "foul": test_data["odor_f"] = 1
if odor == "musty": test_data["odor_m"] = 1
if odor == "none": test_data["odor_n"] = 1
if odor == "pungent": test_data["odor_p"] = 1
if odor == "spicy": test_data["odor_s"] = 1

# gill_attachment
if gill_attachment == "attached": test_data["gill-attachment_a"] = 1
if gill_attachment == "free": test_data["gill-attachment_f"] = 1

# gill_spacing
if gill_spacing == "close": test_data["gill-spacing_c"] = 1
if gill_spacing == "crowded": test_data["gill-spacing_w"] = 1

# gill_size
if gill_size == "broad": test_data["gill-size_b"] = 1
if gill_size == "narrow": test_data["gill-size_n"] = 1

# gill_color
if gill_color == "black": test_data["gill-color_k"] = 1
if gill_color == "brown": test_data["gill-color_n"] = 1
if gill_color == "buff": test_data["gill-color_b"] = 1
if gill_color == "chocolate": test_data["gill-color_h"] = 1
if gill_color == "gray": test_data["gill-color_g"] = 1
if gill_color == "green": test_data["gill-color_r"] = 1
if gill_color == "orange": test_data["gill-color_o"] = 1
if gill_color == "pink": test_data["gill-color_p"] = 1
if gill_color == "purple": test_data["gill-color_u"] = 1
if gill_color == "red": test_data["gill-color_e"] = 1
if gill_color == "white": test_data["gill-color_w"] = 1
if gill_color == "yellow": test_data["gill-color_y"] = 1

# stalk_shape 
if stalk_shape == "enlarging": test_data["stalk-shape_e"] = 1
if stalk_shape == "tapering": test_data["stalk-shape_t"] = 1

# stalk_root
if stalk_root == "bulbous": test_data["stalk-root_b"] = 1
if stalk_root == "club": test_data["stalk-root_c"] = 1
if stalk_root == "equal": test_data["stalk-root_e"] = 1
if stalk_root == "rooted": test_data["stalk-root_r"] = 1
if stalk_root == "missing": test_data["stalk-root_?"] = 1

# stalk_surface_above_ring
if stalk_surface_above_ring == "fibrous": test_data["stalk-surface-above-ring_f"] = 1
if stalk_surface_above_ring == "scaly": test_data["stalk-surface-above-ring_y"] = 1
if stalk_surface_above_ring == "silky": test_data["stalk-surface-above-ring_k"] = 1
if stalk_surface_above_ring == "smooth": test_data["stalk-surface-above-ring_s"] = 1

# stalk_surface_below_ring
if stalk_surface_below_ring == "fibrous": test_data["stalk-surface-below-ring_f"] = 1
if stalk_surface_below_ring == "scaly": test_data["stalk-surface-below-ring_y"] = 1
if stalk_surface_below_ring == "silky": test_data["stalk-surface-below-ring_k"] = 1
if stalk_surface_below_ring == "smooth": test_data["stalk-surface-below-ring_s"] = 1

# stalk_color_above_ring
if stalk_color_above_ring == "brown": test_data["stalk-color-above-ring_n"] = 1
if stalk_color_above_ring == "buff": test_data["stalk-color-above-ring_b"] = 1
if stalk_color_above_ring == "cinnamon": test_data["stalk-color-above-ring_c"] = 1
if stalk_color_above_ring == "gray": test_data["stalk-color-above-ring_g"] = 1
if stalk_color_above_ring == "orange": test_data["stalk-color-above-ring_o"] = 1
if stalk_color_above_ring == "pink": test_data["stalk-color-above-ring_p"] = 1
if stalk_color_above_ring == "red": test_data["stalk-color-above-ring_e"] = 1
if stalk_color_above_ring == "white": test_data["stalk-color-above-ring_w"] = 1
if stalk_color_above_ring == "yellow": test_data["stalk-color-above-ring_y"] = 1

# stalk_color_below_ring
if stalk_color_below_ring == "brown": test_data["stalk-color-below-ring_n"] = 1
if stalk_color_below_ring == "buff": test_data["stalk-color-below-ring_b"] = 1
if stalk_color_below_ring == "cinnamon": test_data["stalk-color-below-ring_c"] = 1
if stalk_color_below_ring == "gray": test_data["stalk-color-below-ring_g"] = 1
if stalk_color_below_ring == "orange": test_data["stalk-color-below-ring_o"] = 1
if stalk_color_below_ring == "pink": test_data["stalk-color-below-ring_p"] = 1
if stalk_color_below_ring == "red": test_data["stalk-color-below-ring_e"] = 1
if stalk_color_below_ring == "white": test_data["stalk-color-below-ring_w"] = 1
if stalk_color_below_ring == "yellow": test_data["stalk-color-below-ring_y"] = 1

# veil_type
if veil_type == "partial": test_data["veil-type_p"] = 1

# veil_color
if veil_color == "brown": test_data["veil-color_n"] = 1
if veil_color == "orange": test_data["veil-color_o"] = 1
if veil_color == "white": test_data["veil-color_w"] = 1
if veil_color == "yellow": test_data["veil-color_y"] = 1

# ring_number
if ring_number == "none": test_data["ring-number_n"] = 1
if ring_number == "one": test_data["ring-number_o"] = 1
if ring_number == "two": test_data["ring-number_t"] = 1

# ring_type
if ring_type == "evanescent": test_data["ring-type_e"] = 1
if ring_type == "flaring": test_data["ring-type_f"] = 1
if ring_type == "large": test_data["ring-type_l"] = 1
if ring_type == "none": test_data["ring-type_n"] = 1
if ring_type == "pendant": test_data["ring-type_p"] = 1

# spore-print-color
if spore_print_color == "black": test_data["spore-print-color_k"] = 1
if spore_print_color == "brown": test_data["spore-print-color_n"] = 1
if spore_print_color == "buff": test_data["spore-print-color_b"] = 1
if spore_print_color == "chocolate": test_data["spore-print-color_h"] = 1
if spore_print_color == "green": test_data["spore-print-color_r"] = 1
if spore_print_color == "orange": test_data["spore-print-color_o"] = 1
if spore_print_color == "purple": test_data["spore-print-color_u"] = 1
if spore_print_color == "white": test_data["spore-print-color_w"] = 1
if spore_print_color == "yellow": test_data["spore-print-color_y"] = 1

# population
if population == "abundant": test_data["population_a"] = 1
if population == "clustered": test_data["population_c"] = 1
if population == "numerous": test_data["population_n"] = 1
if population == "scattered": test_data["population_s"] = 1
if population == "several": test_data["population_v"] = 1
if population == "solitary": test_data["population_y"] = 1

# habitat
if habitat == "grasses": test_data["habitat_g"] = 1
if habitat == "leaves": test_data["habitat_l"] = 1
if habitat == "meadows": test_data["habitat_m"] = 1
if habitat == "paths": test_data["habitat_p"] = 1
if habitat == "urban": test_data["habitat_u"] = 1
if habitat == "waste": test_data["habitat_w"] = 1
if habitat == "woods": test_data["habitat_d"] = 1

# test botton
if st.button("TEST!!"):
    prediction = model.predict(test_data.iloc[:, 1:])
    if prediction[0] == 1:
        st.write("Edible!!")
    else:
        st.write("Poisonous!!")




# %%
