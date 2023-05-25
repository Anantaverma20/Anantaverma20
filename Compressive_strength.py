#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


# In[2]:


from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score, cross_val_predict, train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


import pandas as pd
import numpy as np
# Read the Excel file
excel_file = pd.ExcelFile("E:\\TOPCEM internship\\Environment data 1.xlsx")

# Get the sheet names
sheet_names = excel_file.sheet_names

# Iterate over each sheet and perform operations
for sheet_name in sheet_names:
    # Read the sheet into a DataFrame
    df = excel_file.parse(sheet_name)


# In[4]:


# Select a specific sheet by name
sheet_name = 'Final Product OPC1'
df = excel_file.parse(sheet_name)

# Perform operations on the selected sheet
print(df.head())


# In[5]:


# Select a specific sheet by name
sheet_name = 'Final Product OPC1'
data = excel_file.parse(sheet_name)
# Perform operations on the selected sheet
print(df.head())


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


df.isnull().sum()


# In[9]:


corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
graph = sns.heatmap(df[top_corr_features].corr(),annot = True, cmap ="RdYlGn")

scaler = MinMaxScaler()
scaler.fit(df)
df_scaled = scaler.transform(df)
df_scaled
# In[10]:


del df['3-Days Mpa']
del df['7-Days Mpa']


# In[11]:


df.head()


# In[12]:


scaler = MinMaxScaler() #scaling the numeric data columns
features = [['Fineness M2/kg','IST Min','FST Min','LC MM', 'AC %', 'NC', 'SO3 %', 'IR %', 'MgO %','LOI %','28-Days Mpa' ]]
for feature in features:
    df[feature] = scaler.fit_transform(df[feature])


# In[13]:


df.head()


# In[14]:


# spilitting target variables and predicting variables
y = df.iloc[:,[5]].values
x = df.iloc[:,[0,1,2,3,4,6,7,8,9,10]].values


# In[15]:


#splitting the dataset as 20-80 test_train data
xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size= 0.15, random_state=42)


# In[16]:


model = LinearRegression()
model.fit(xTrain, yTrain)


# In[17]:


y_pred = model.predict(xTest)


# In[18]:


y_pred


# In[19]:


print("mean squared error is:", mean_squared_error(yTest, y_pred))
print("mean absolute error is:", mean_absolute_error(yTest, y_pred))


# In[20]:


import statistics

std_dev = statistics.stdev(data['28-Days Mpa'])
print("Standard Deviation:", std_dev)
mean = data['28-Days Mpa'].mean()
mean


# In[21]:


original_pred = y_pred * std_dev + mean


# In[22]:


original_pred


# In[23]:


pip install streamlit


# In[24]:


import streamlit as st 


# In[25]:


st.title('Streamlit App Demo')
st.header('Header')
st.subheader('Sub Header')
st.text('Welcome to the Streamlit App') 
st.dataframe(df)

# In[29]:





# In[ ]:




