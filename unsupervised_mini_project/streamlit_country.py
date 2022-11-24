import streamlit as st
import pandas as pd

data = pd.read_csv("saved.csv")

def main():
    st.title("Country in Need")
    st.sidebar.title("Sidebar")
    # st.image("")
main()


st.dataframe(data)
country = st.text_input('Enter a country name')