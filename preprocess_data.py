import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import math
import warnings
warnings.filterwarnings("ignore")

def preprocess_data(df):
      
    st.write(df.shape)

    df1 = df #.drop(columns=['Date']) 'S. No.','Theoritical Liquid rate with 80% eff (bpd)','PI (dyna)','Dyna Eff SL (surface) in','PI (echoshot)'])
    convert1 = {'THP (psi)':float}
    df1 = df1.astype(convert1)
    
    df2 = df1[df1['Pump Submergence (mTVD)'].notna()]            #Only Non-NAN valued rows of submergence column
    x2 = df2[['Pump depth (mTVD)','SPM','Oil rate (bpd)',"Pump Bore Size (in)",'GOR (scf/bbl)','CHP (psi)','THP (psi)','PIP dyna (psi)']]  
    y2 = df2['Pump Submergence (mTVD)']  
    df1 = df1[df1['Pump Submergence (mTVD)'].isna()]            #Only NAN valued rows of submergence column
    x1 = df1[['Pump depth (mTVD)','SPM','Oil rate (bpd)',"Pump Bore Size (in)",'GOR (scf/bbl)','CHP (psi)','THP (psi)','PIP dyna (psi)']]  
    y1 = df1['Pump Submergence (mTVD)']
    xg = XGBRegressor()
    xg.fit(x2,y2)
    y1_hat = xg.predict(x1)
    df1['Pump Submergence (mTVD)'] = y1_hat

    df0 = df1.append(df2,ignore_index=True)
    df3 = df0[df0['PIP echoshot (psi)'].isna()] 
    df3['PIP echoshot (psi)'] = df3['CHP (psi)'] + (0.379*3.28*df3['Pump Submergence (mTVD)'])
    df4 = df0[df0['PIP echoshot (psi)'].notna()]
    df51 = df3.append(df4)

    #st.write(df51.shape)
    #st.dataframe(df51)
    xgb1 = XGBRegressor()
    df52 = df51[df51["Dyna Eff SL (surface) in"].notna()]
    x31 = df52[['Pump depth (mTVD)','SPM','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)']]
    y31 = df52['Dyna Eff SL (surface) in']
    df53 = df51[df51['Dyna Eff SL (surface) in'].isna()]
    x32 = df53[['Pump depth (mTVD)','SPM','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)']]
    #y32 = df53['Dyna Eff SL (Pump) in']
    xgb1.fit(x31,y31)
    df53['Dyna Eff SL (surface) in'] = xgb1.predict(x32)
    df54 = df52.append(df53)
    #st.write(df54)

    xgb1 = XGBRegressor()
    df52 = df51[df51["Dyna Eff SL (Pump) in"].notna()]
    x31 = df52[['Pump depth (mTVD)','SPM','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)']]
    y31 = df52['Dyna Eff SL (Pump) in']
    df53 = df51[df51['Dyna Eff SL (Pump) in'].isna()]
    x32 = df53[['Pump depth (mTVD)','SPM','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)']]
    #y32 = df53['Dyna Eff SL (Pump) in']
    xgb1.fit(x31,y31)
    y_hat3 = xgb1.predict(x32)
    #st.write(len(y31))
    #st.write(len(y_hat3))
    #st.write(df53['Dyna Eff SL (Pump) in'].shape)
    df53['Dyna Eff SL (Pump) in'] = y_hat3
    df54 = df52.append(df53)
    #st.write(df54.shape)
    #df54 = df54.drop(columns=['Max Surface SL','Max Downhole SL','S. No.'])
    #st.write(df54)

    xgb1 = XGBRegressor()
    df521 = df54[df54['Dyna load (pump) lb'].notna()]
    x311 = df521[['Pump depth (mTVD)','SPM','Dyna Eff SL (Pump) in','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)']]
    y311 = df521['Dyna load (pump) lb']
    df531 = df54[df54['Dyna load (pump) lb'].isna()]
    x321 = df531[['Pump depth (mTVD)','SPM','Dyna Eff SL (Pump) in','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)']]
    #y32 = df53['Dyna Eff SL (Pump) in']
    xgb1.fit(x311,y311)
    df531['Dyna load (pump) lb'] = xgb1.predict(x321)
    df54 = df521.append(df531)
    

    df54['Theoritical Liquid rate with 80% eff (bpd)'] = (0.8*df54['SPM']*3.14*3.25*3.25*df54['Dyna Eff SL (Pump) in']*60*24)/(4*144*12*5.615)
    #df54['']

    xgb1 = XGBRegressor()
    df521 = df54[df54['PI (dyna)'].notna()]
    x311 = df521[['Pump depth (mTVD)','SPM','Dyna Eff SL (Pump) in','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)','Theoritical Liquid rate with 80% eff (bpd)']]
    y311 = df521['PI (dyna)']
    df531 = df54[df54['PI (dyna)'].isna()]
    x321 = df531[['Pump depth (mTVD)','SPM','Dyna Eff SL (Pump) in','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)','Theoritical Liquid rate with 80% eff (bpd)']]
    #y32 = df53['Dyna Eff SL (Pump) in']
    xgb1.fit(x311,y311)
    df531['PI (dyna)'] = xgb1.predict(x321)
    df54 = df521.append(df531)

    df5 = df54
    #df5 = df5.dropna(subset=[n for n in df if (n != 'PI (echoshot)' and n != 'Remarks' and n != 'Dyna Eff SL (surface) in' and n != 'Max Surface SL' and n != 'Max Downhole SL' and n != 'PIP dyna (psi)' and n != 'PI (dyna)')])
    #df5 = df5.drop(columns=['Max SL surface','Max SL downhole'])
    convert = {'PIP echoshot (psi)':float,'PIP dyna (psi)':float}
    df5 = df5.astype(convert)
    
    
    #st.write(df5.shape)

    xgb = XGBRegressor()
    df6 = df5[df5['PI (echoshot)'].notna()]
    x3 = df6[['Pump depth (mTVD)','SPM','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)','PI (dyna)']]
    y3 = df6['PI (echoshot)']
    df7 = df5[df5['PI (echoshot)'].isna()]
    x4 = df7[['Pump depth (mTVD)','SPM','GOR (scf/bbl)',"Pump Bore Size (in)",'CHP (psi)','THP (psi)','Pump Submergence (mTVD)','PI (dyna)']]
    y4 = df7['PI (echoshot)']
    xgb.fit(x3,y3)
    y_hat2 = xgb.predict(x4)
    df7['PI (echoshot)'] = y_hat2
    
    df8 = df6.append(df7)
    #st.write(df8.shape)
    #st.write(df8)
    #st.write(df8.shape)
    #df8 = df8.dropna()
    #st.write(df8[df8['PI (dyna)']>0.0].shape)
    #df8 = df8[df8['PI (dyna)']>0.0]
    #st.write(df8.shape)
    df8 = df8.dropna(axis=0,subset=[n for n in df8 if (n != 'Pump Submergence (mTVD)' and
                                                        n != 'PI (echoshot)' and 
                                                        n != 'Remarks' and 
                                                        n != 'Dyna Eff SL (surface) in' and 
                                                        n != 'Dyna Eff SL (Pump) in' and
                                                        n != 'Max Surface SL' and 
                                                        n != 'Max Downhole SL' and 
                                                        n != 'PIP dyna (psi)' and 
                                                        n != 'PI (dyna)' and
                                                        n != 'PIP echoshot (psi)' and
                                                        n != 'Theanditical Liquid rate with 80% eff (bpd)' and
                                                        n != 'Date' and
                                                        n != 'S. No.' and
                                                        n != 'Pump depth (mTVD)' and
                                                        n != 'SPM' and
                                                        n != 'Pump Bande Size (in)' and
                                                        n != 'Oil rate (bpd)' and
                                                        n != 'Gand (scf/bbl)' and
                                                        n != 'Liq Rate Slippage (bpd)')])
    #df8 = df8[df8['CHP (psi)'].notna()]
    #df8 = df8[df8['THP (psi)'].notna()]

    st.write(df8.shape)
    st.write(df8)

    return df8

