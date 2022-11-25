import streamlit as st
import pandas as pd

data = pd.read_csv("unsupervised_mini_project/saved.csv")

def main():
    st.title("Country in Need")
    st.sidebar.title("Sidebar")
    # st.image("")
main()


st.dataframe(data["country"])
country = st.slider('Pick the number of country', 0, 166)
st.subheader("Country Details")
st.write(data.iloc[country, 0:-3])

text = ""
text += f'<span style = "color:black">{"Help need level for "}{data.iloc[country,0]}{" is "}</span>'

if data.iloc[country, -3] == 0:
    text += f'<span style = "color:red">{"High"} </span>'
if data.iloc[country, -3] == 1:
    text += f'<span style = "color:green">{"Medium"} </span>'
if data.iloc[country, -3] == 2:
    text += f'<span style = "color:blue">{"Low"} </span>'

st.write(text, unsafe_allow_html = True)

