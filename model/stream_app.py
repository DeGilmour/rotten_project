import streamlit as st
from genre_predictor import PreProcessing, TrainModel, TestModel
st.set_option('deprecation.showPyplotGlobalUse', False)
rotten_ds = PreProcessing().rotten_ds

st.dataframe(rotten_ds[['movie_title','movie_info', 'tomatometer_rating']])


st.write("Todos os generos")
st.pyplot(rotten_ds[['genres']])