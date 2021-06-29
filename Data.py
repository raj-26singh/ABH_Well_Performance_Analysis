import pandas as pd
import numpy as np
import streamlit as st
#import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")


def data(welldatadf):
    st.write('''## Data Visualisation''')
    st.write('It makes it simple to have a peek in your data')
    if welldatadf is not None:
        
        # Simple dataframe display
        cols = st.beta_columns(2)
        cols[0].write('''### Well-Data dataframe''')
        cols[0].write(welldatadf)
        #cols[1].write('''### powerdata dataframe''')
        #cols[1].write(powercurvedf)
        
        # PowerCurve Display
        st.write('''### Data plots''')

        # Only a subset of options make sense
        x_options = welldatadf.columns
        y_options = welldatadf.columns
        # Allow use to choose
        cols = st.beta_columns(2)
        x_axis = cols[0].selectbox('Choose Value for X axis', x_options)
        y_axis = cols[1].selectbox('Choose Value for primary Y axis', y_options)
        y2_axis = cols[1].selectbox('Choose Value for secondary Y axis',y_options)
        # plot the value
        st.write(x_axis,'  vs  ', y_axis,'&', y2_axis)
        '''fig = px.scatter(welldatadf,
                        x=x_axis,
                        y= y_axis,
                        hover_name='PIP dyna (psi)')'''
        '''fig2 = px.bar(welldatadf,
                        x=x_axis,
                        y= y2_axis,
                        hover_name='PIP dyna (psi)')'''
        for i in range(len(x_options)):
            if x_options[i]==x_axis:
                x_ind = i
        for i in range(len(y_options)):
            if x_options[i]==y_axis:
                y_ind = i
        for i in range(len(y_options)):
            if x_options[i]==y2_axis:
                y2_ind = i

        x_val = np.array(welldatadf.iloc[:,x_ind])
        y_val = np.array(welldatadf.iloc[:,y_ind])
        y2_val = np.array(welldatadf.iloc[:,y2_ind])

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=x_val, y=y_val,name="Primary Y-axis"),secondary_y=False,)
        fig.add_trace(go.Scatter(x=x_val, y=y2_val,name="Secondary Y-axis"),secondary_y=True,)
        fig.update_yaxes(title_text="<b>Primary</b> Y-axis ", secondary_y=False)
        fig.update_yaxes(title_text="<b>Secondary</b> Y-axis ", secondary_y=True)
        #fig.show()
        
        st.plotly_chart(fig, use_container_width=True)
        #st.plotly_chart(fig2, use_container_width=True)
        
        '''fig = plt.figure()
        
        ax1 = fig.add_subplot(111)
        ax1.plot(x=x_val, y=y_val)
        ax1.set_ylabel('Primary Y-axis')

        ax2 = ax1.twinx()
        ax2.plot(x=x_val, y=y2_val)
        ax2.set_ylabel('Secondary Y-axis')
        
        st.pyplot(plt)'''
        
    else:
        st.error('File Not uploaded yet. Please upload the well data first in order to visualize them')
        