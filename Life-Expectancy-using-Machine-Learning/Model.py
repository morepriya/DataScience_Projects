#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats.mstats import winsorize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import pickle


# In[2]:


df = pd.read_csv("C:/Users/priya/Desktop/Django/ML/Model and Data/Life Expectancy Data.csv")
df.head(20)


# In[3]:


df.isnull().sum()


# In[4]:


df.shape


# In[5]:


df.dtypes


# In[6]:


df.describe()


# In[7]:


df.info()


# In[8]:


country_list = df.Country.unique()
fill_list = ['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality','infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B','Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure','Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']


# In[9]:


for country in country_list:
    df.loc[df['Country'] == country,fill_list] = df.loc[df['Country'] == country,fill_list].interpolate()
df.dropna(inplace=True)


# In[10]:


df.shape


# In[11]:


df.isnull().sum()


# In[12]:


df.rename(columns={" BMI ":"BMI","Life expectancy ":"Life_Expectancy","Adult Mortality":"Adult_Mortality",
                   "infant deaths":"Infant_Deaths","percentage expenditure":"Percentage_Exp","Hepatitis B":"HepatitisB",
                  "Measles ":"Measles"," BMI ":"BMI","under-five deaths ":"Under_Five_Deaths","Diphtheria ":"Diphtheria",
                  " HIV/AIDS":"HIV/AIDS"," thinness  1-19 years":"thinness_1to19_years"," thinness 5-9 years":"thinness_5to9_years","Income composition of resources":"Income_Comp_Of_Resources",
                   "Total expenditure":"Tot_Exp"},inplace=True)


col_dict = {'Life_Expectancy':1 , 'Adult_Mortality':2 ,
        'Alcohol':3 , 'Percentage_Exp': 4, 'HepatitisB': 5,
       'Measles' : 6, 'BMI': 7, 'Under_Five_Deaths' : 8, 'Polio' : 9, 'Tot_Exp' :10,
       'Diphtheria':11, 'HIV/AIDS':12, 'GDP':13, 'Population' :14,
       'thinness_1to19_years' :15, 'thinness_5to9_years' :16,
       'Income_Comp_Of_Resources' : 17, 'Schooling' :18, 'Infant_Deaths':19}


# In[13]:


for variable in col_dict.keys():
    q75, q25 = np.percentile(df[variable], [75 ,25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    print("Number of outliers in {} : {} ".format(variable,len((np.where((df[variable] > max_val) | (df[variable] < min_val))[0]))))


# In[14]:


plt.figure(figsize=(20,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.boxplot(df[variable],whis=1.5)
                     plt.title(variable)

plt.show()


# In[15]:


plt.figure(figsize=(20,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.scatter(df["Life_Expectancy"], df[variable])
                     plt.title(variable)

plt.show()


# In[16]:


winsorize(df["Life_Expectancy"],(0.01,0), inplace=True)
winsorize(df["Adult_Mortality"],(0,0.03), inplace=True)
winsorize(df["Infant_Deaths"],(0,0.10), inplace=True)
winsorize(df["Alcohol"],(0,0.01), inplace=True)
winsorize(df["Percentage_Exp"],(0,0.12), inplace=True)
winsorize(df["HepatitisB"],(0.11,0), inplace=True)
winsorize(df["Measles"],(0,0.19), inplace=True)
winsorize(df["Under_Five_Deaths"],(0,0.12), inplace=True)
winsorize(df["Polio"],(0.09,0), inplace=True)
winsorize(df["Tot_Exp"],(0,0.01), inplace=True)
winsorize(df["Diphtheria"],(0.10,0), inplace=True)
winsorize(df["HIV/AIDS"],(0,0.16), inplace=True)
winsorize(df["GDP"],(0,0.13), inplace=True)
winsorize(df["Population"],(0,0.14), inplace=True)
winsorize(df["thinness_1to19_years"],(0,0.04), inplace=True)
winsorize(df["thinness_5to9_years"],(0,0.04), inplace=True)
winsorize(df["Income_Comp_Of_Resources"],(0.05,0), inplace=True)
winsorize(df["Schooling"],(0.02,0.01), inplace=True)


# In[17]:


for variable in col_dict.keys():
    q75, q25 = np.percentile(df[variable], [75 ,25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    print("Number of outliers in {} : {} ".format(variable,len((np.where((df[variable] > max_val) | (df[variable] < min_val))[0]))))


# In[18]:


plt.figure(figsize=(20,30))

for variable,i in col_dict.items():
                     plt.subplot(5,4,i)
                     plt.scatter(df["Life_Expectancy"], df[variable])
                     plt.title(variable)

plt.show()


# In[19]:


# Adult Mortality, Income_Comp_Of_Resources, Schooling, Alcohol


# In[20]:


plt.figure(figsize=(20,8))

plt.subplot(1,3,1)
plt.scatter(df["Schooling"], df["Adult_Mortality"])
plt.title("Schooling vs AdultMortality")

plt.subplot(1,3,2)
plt.scatter(df["Schooling"], df["Income_Comp_Of_Resources"])
plt.title("Schooling vs Income_Comp_Of_Resources")

plt.subplot(1,3,3)
plt.scatter(df["Adult_Mortality"], df["Income_Comp_Of_Resources"])
plt.title("AdultMortality vs Income_Comp_Of_Resources")

plt.show()


# In[21]:


plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
plt.scatter(df["Alcohol"], df["Adult_Mortality"])
plt.title("Alcohol vs AdultMortality")

plt.subplot(1,3,2)
plt.scatter(df["Alcohol"], df["Income_Comp_Of_Resources"])
plt.title("Alcohol vs Income_Comp_Of_Resources")

plt.subplot(1,3,3)
plt.scatter(df["Alcohol"], df["Schooling"])
plt.title("Alcohol vs Schooling")
plt.show()


# In[22]:


round(df[['Status','Life_Expectancy']].groupby(['Status']).mean(),2)


# In[23]:


import scipy.stats as stats
stats.ttest_ind(df.loc[df['Status']=='Developed','Life_Expectancy'],df.loc[df['Status']=='Developing','Life_Expectancy'])


# In[24]:


repl={"Status":{"Developing":0,"Developed":1}}
df.replace(repl, inplace=True)
df = df.apply(pd.to_numeric, errors='coerce')

cols=['Country', 'Year', 'Life_Expectancy',
       'Infant_Deaths',  'Percentage_Exp', 'HepatitisB', 'Measles',
       'BMI', 'Under_Five_Deaths', 'Polio', 'Tot_Exp', 'Diphtheria',
       'GDP', 'Population', 'thinness_1to19_years',
       'thinness_5to9_years']
 

# In[25]:


#X = df[['Status','Schooling','Income_Comp_Of_Resources','HIV/AIDS','Adult_Mortality','Alcohol']]
X = df.drop(cols, axis = 1 )
y = df['Life_Expectancy']
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
model = LinearRegression(fit_intercept=True, normalize=True).fit(X_train, Y_train)
predictions= model.predict(X_test)

#nsamples, nx, ny = X.shape
#d2_train_dataset = X.reshape((nsamples,nx*ny))

#regressor =LinearRegression()
#regressor.fit(d2_train_dataset,y)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[1,1,1,1,1,1]]))
