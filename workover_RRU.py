import streamlit as st
import pandas as pd
import numpy as np

def workover_RRU(df):
    if df is not None:
        st.write(df)
        st.write(df.columns)
    else:
        st.error('File Not uploaded yet. Please upload the field data first in order to analyse workover/RRU')