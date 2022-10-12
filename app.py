from tkinter.tix import Meter
from turtle import color
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import metrics
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as qo
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions
import json
from urllib.request import urlopen
import seaborn
from sklearn.decomposition import PCA

st.title("KNN by Rushabh")
uploaded_file = st.file_uploader(label="Upload CSV file",type=['csv','xlsx'])

# Dataset cleaning
@st.cache
def cleaning(file):
    if file is not None:
        df = pd.read_csv(file)
        category_cols = []
        num_cols=[c for c in list(df.columns) if df[c].dtype == 'int64' or df[c].dtype == 'float64']
        for i in num_cols:
            if df[i].isnull().sum().all() == True:
                df[i]=df[i].fillna(df[i].mean())

        threshold = 10
        for each in df.columns:
            if df[each].nunique() < threshold:
                category_cols.append(each)
        for each in category_cols:
            df[each] = df[each].astype('category')

        for i in category_cols:
            if df[i].isnull().sum().all()== True:
                df[i]=df[i].fillna(df[i].mode()[0])
        for i in df.columns:
            if df[i].isnull().sum().all()==True:
                df = df.dropna()

        return df

if uploaded_file is not None:
    
    file=cleaning(uploaded_file)
    uploaded_file=file.to_csv('file1.csv')
    data = pd.read_csv('file1.csv')
    st.header("Dataset")
    
    st.write("Data set contains "+ str(data.shape[0]) +" rows")
    st.write("Data set contains "+ str(data.shape[1]) +" columns")
    a1=st.number_input('Pick a number of rows to see', 0,data.shape[0]+1 )
    if a1>=1:
        st.dataframe(data.head(a1))
        list1=[]  
        for i in data.columns:
            list1.append(i)
        target=st.selectbox("What is the target column?",(list1))
        x=data.loc[:, data.columns != target]
        y=data[target]
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)

        nn=st.number_input('Pick a KNN number', 1, 1000)
        weights=st.selectbox("What is the weight type",("uniform","distance"))

        if data[target].dtype=='float':
            model = neighbors.KNeighborsRegressor(n_neighbors = nn)
            model.fit(X_train, y_train)  #fit the model
            y_pred=model.predict(X_test) #make prediction on test set
            score = r2_score(y_test, y_pred)


        else:
            knn = KNeighborsClassifier(n_neighbors=nn,weights=weights)
            knn.fit(X_train,y_train)
            y_pred=knn.predict(X_test)

        check1 = st.checkbox("Accuracy Score")
        if check1:
            if data[target].dtype=='float64':
                st.header("Accuracy Score")
                st.subheader(score)
            else:
                st.header("Accuracy Score")
                st.subheader(metrics.accuracy_score(y_test,y_pred))

        check2 = st.checkbox("HeatMap")
        if check2:
            fig, ax = plt.subplots()
            sns.heatmap(data.corr(), ax=ax)
            st.header("Heatmap")
            st.write(fig)
        
        check5 = st.checkbox("KNN-Visualization")
        if check5:
            column1=st.selectbox("What is the  column1 to be used for visualization?",(list1))
            column2=st.selectbox("What is the  column2 to be used for visualization?",(list1))
            if column1!=column2:
                x = data[[column1,column2]].values
                data[target] = data[target].astype('category')
                data['Types_cat'] = data[target].cat.codes
                labelencoder = LabelEncoder()
                data['Types_Cat'] = labelencoder.fit_transform(data[target])
                
                y = data['Types_Cat'].astype(int).values
                
                knn.fit(x,y)
                fig=plt.figure(figsize=(10, 5))
                plot_decision_regions(x, y, clf=knn, legend=2)
                plt.xlabel(column2)
                plt.ylabel(column1)
                st.pyplot(fig)
            