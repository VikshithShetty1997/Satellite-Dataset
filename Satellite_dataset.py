#!/usr/bin/env python
# coding: utf-8

# Conjunctions in space occur when two or more Resident Space Objects(RSOs) in the Earth's orbit pass dangerously close to one another, resulting inpossible collision scenarios. With the rapid increase in the number of RSOs in recenttimes, the number of predicted conjunctions has significantly increased. The interpretation and visualisation of around 250,000 conjunctions per day is achallenging data analytics problem.

# In[1]:


import warnings
warnings.filterwarnings('ignore')


# In[40]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import datetime 
import datetime as dt


# Assumption: Column TCA(Time of closes approach) is in object form so will convert it into DateTime Format Iam Assuming Date to be from 1st of november 2023 Looking at the column TCA is in time format "Hour:Minute.Second" and i will take a time difference of 24 hours to be one day and continuing the date for every 24 hour difference

# In[41]:


#Import the Dataset
data=pd.read_csv("celestrak.csv")
data


# In[42]:


#Check for Dimensions of a Data
data.shape


# In[43]:


# Check for any missing Values
data.info()


# There is no missing or Null values in the Dataset so continuing with Data Analysis

# Column TCA(Time of closes approach) is in object form so will convert it into DateTime Format
# Iam Assuming Date to be from 1st of november 2023 
# Looking at the column TCA is in time format "Hour:Minute.Second" and i will take a time difference of 24 hours to be one day and continuing the date for every 24 hour difference

# In[44]:


# Setting a start date 
start_date = datetime.datetime(2023, 11, 1)
start_date


# In[45]:


# Converting the time format to decimal using lambda function
data['TCA'] = data['TCA'].apply(lambda x: float(x.split(':')[0]) + float(x.split(':')[1])/60)
data['TCA']


# In[46]:


#Combining date and time using lamda function
data['TCA'] = data['TCA'].apply(lambda x: start_date + datetime.timedelta(hours=x))
data['TCA']


# In[47]:


# Checking for the modified Table
data


# In[48]:


# Check for datatype after modification
data.info()


# A) Derive high level analytics from the whole data set for a single day. In other words, derive general analytics of the whole set of conjunction scenarios (for e.g. number of conjunctions among active satellites). The analytics should be intuitive and represented in an easily understandable format

# In[50]:


# Selecting November 1 as day for analysis of the conjection data
df_nov1 = data[data['TCA'].dt.date == dt.date(2023, 11, 1)]
df_nov1


# In[51]:


# Statistical Analysis of the selected Dataset 
df_nov1[["DSE_2","TCA_RELATIVE_SPEED","TCA_RELATIVE_SPEED","DILUTION"]].describe()


# In[16]:


# Set the style of seaborn
sns.set(style='whitegrid')


# In[17]:


# Set the style of seaborn
sns.set(style='whitegrid')

# Create a figure and a set of subplots
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

# Plot the distribution of TCA_RANGE
sns.histplot(data=df_nov1, x='TCA_RANGE', kde=True, ax=axs[0, 0])
axs[0, 0].set_title('Distribution of TCA_RANGE')

# Plot the distribution of TCA_RELATIVE_SPEED
sns.histplot(data=df_nov1, x='TCA_RELATIVE_SPEED', kde=True, ax=axs[0, 1])
axs[0, 1].set_title('Distribution of TCA_RELATIVE_SPEED')

# Plot the distribution of MAX_PROB
sns.histplot(data=df_nov1, x='MAX_PROB', kde=True, ax=axs[1, 0])
axs[1, 0].set_title('Distribution of MAX_PROB')

# Plot the distribution of DILUTION
sns.histplot(data=df_nov1, x='DILUTION', kde=True, ax=axs[1, 1])
axs[1, 1].set_title('Distribution of DILUTION')

# Adjust the layout
plt.tight_layout()

# Show the plots
plt.show()


# From these plots, we can observe the following:
# The TCA_RANGE variable seems to be right-skewed, indicating that most of the values are concentrated on the lower end.
# The TCA_RELATIVE_SPEED variable also appears to be right-skewed.
# The MAX_PROB variable shows a distribution that is heavily concentrated towards lower values.
# The DILUTION variable has a distribution that is somewhat uniform but with a peak at the lower end.

# In[53]:


# Combine the NORAD_CAT_ID columns for both satellites
all_satellites = pd.concat([df_nov1['OBJECT_NAME_1'], df_nov1['OBJECT_NAME_2']])
all_satellites


# In[54]:


# Calculate the total number of unique satellites
num_unique_satellites = all_satellites.nunique()
num_unique_satellites


# In[18]:


# Total number of conjunctions on 1st of November
num_conjunctions = df_nov1.shape[0]
num_conjunctions


# In[56]:


# Calculate the number of conjunctions each satellite has been involved in and displaying top 5 data
num_conjunctions_per_satellite = all_satellites.value_counts()
num_conjunctions_per_satellite.head()


# In[19]:


# Number of conjunctions among active satellites
num_active_conjunctions = df_nov1[df_nov1['OBJECT_NAME_1'].str.contains('\[\+\]') & df_nov1['OBJECT_NAME_2'].str.contains('\[\+\]')].shape[0]
num_active_conjunctions


# In[20]:


# Maximum probability of conjunctions
max_prob = df_nov1['MAX_PROB'].max()
max_prob


# In[21]:


# Minimum probability of conjunctions
min_prob = df_nov1['MAX_PROB'].min()
min_prob


# In[22]:


# Average probability of conjunctions
avg_prob = df_nov1['MAX_PROB'].mean()
avg_prob


# In[23]:


# Create a dataframe for the analytics
df_analytics = pd.DataFrame({'Total Conjunctions': [num_conjunctions],
                              'Active Conjunctions': [num_active_conjunctions]})

# Plot the analytics
df_analytics.plot(kind='bar')
plt.title('Conjunction Analytics for November 1st, 2023')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks([])  # remove x-axis labels
plt.show()


# In[24]:


# Create a dataframe for the analytics
df_analytics = pd.DataFrame({'Max Probability': [max_prob],
                              'Min Probability': [min_prob],
                              'Average Probability': [avg_prob]})

# Plot the analytics
df_analytics.plot(kind='bar')
plt.title('Probablity Conjunction Analytics for November 1st, 2023')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks([])  # remove x-axis labels
plt.show()


# In[28]:


# Plot the distribution of the number of conjunctions per satellite
plt.figure(figsize=(10, 6))
sns.histplot(num_conjunctions_per_satellite, bins=50, kde=True)
plt.xlabel('Number of Conjunctions')
plt.ylabel('Number of Satellites')
plt.title('Distribution of Number of Conjunctions per Satellite')
plt.show()


# 

# 

# B) Represent the conjunctions data of a single satellite or a satellite constellation. The analytics should be intuitive, represented in an easily understandable format and should enable decision making from a satellite operator’s point of view

# In[29]:


# Filter the data for the selected satellite
satellite_data = data[(data['NORAD_CAT_ID_1'] == 47746) | (data['NORAD_CAT_ID_2'] == 47746)]
satellite_data


# In[30]:


# Calculate the number of conjunctions for the selected satellite
num_conjunctions = satellite_data.shape[0]
num_conjunctions


# In[31]:


# Number of conjunctions among active satellites
num_active_conjunctions = satellite_data[satellite_data['OBJECT_NAME_1'].str.contains('\[\+\]') & satellite_data['OBJECT_NAME_2'].str.contains('\[\+\]')].shape[0]
num_active_conjunctions


# In[32]:


# Maximum probability of conjunctions
max_prob = satellite_data['MAX_PROB'].max()
max_prob


# In[33]:


# Minimum probability of conjunctions
min_prob = satellite_data['MAX_PROB'].min()
min_prob


# In[34]:


# Average probability of conjunctions
avg_prob = satellite_data['MAX_PROB'].mean()
avg_prob


# In[35]:


# Create a dataframe for the analytics
df_analytics_starlink2160 = pd.DataFrame({'Total Conjunctions': [num_conjunctions],
                                          'Active Conjunctions': [num_active_conjunctions],
                                          'Max Probability': [max_prob],
                                          'Min Probability': [min_prob],
                                          'Average Probability': [avg_prob]})

# Plot the analytics
df_analytics_starlink2160.plot(kind='bar')
plt.title('Conjunction Analytics for STARLINK-2160 [+]')
plt.xlabel('Metrics')
plt.ylabel('Values')
plt.xticks([])  # remove x-axis labels
plt.show()


# 

# Question2

# Use the whole dataset that spans about five days. Derive analytics and visualise the
# data/analytics accounting for the evolution from the first day (for e.g. the number of
# conjunctions of the RSO having NORAD ID 12345 over 7 days of analysis)

# In[62]:


# Filter the data for the RSO having NORAD ID 12345
rsodata = data[(data['NORAD_CAT_ID_1'] == 43925) | (data['NORAD_CAT_ID_2'] == 43925)]
rsodata


# Here is the visualization of the number of conjunctions for the RSO with NORAD ID 43925 over 7 days:

# In[71]:


# Count the number of conjunctions per day
conjunctions_per_day = rsodata['TCA'].dt.date.value_counts().sort_index()
conjunctions_per_day


# In[69]:


# Plot the number of conjunctions per day
plt.figure(figsize=(10, 6))
plt.plot(conjunctions_per_day.index, conjunctions_per_day.values, marker='o')
plt.title('Number of conjunctions of the RSO having NORAD ID 43925 over 7 days of analysis')
plt.xlabel('Date')
plt.ylabel('Number of conjunctions')
plt.grid()
plt.show()


# From the line chart, we can observe that the number of conjunctions per day varies. There are certain days with a higher number of conjunctions, while on other days, the number of conjunctions is relatively lower.

# In[72]:


# Filter the maximum probability of collision for the RSO with NORAD ID 43925
max_prob = rsodata['MAX_PROB']

# Plot the distribution of the maximum probability of collision
plt.figure(figsize=(10, 6))
sns.histplot(max_prob, kde=True)
plt.xlabel('Maximum Probability of Collision')
plt.ylabel('Frequency')
plt.title('Distribution of Maximum Probability of Collision for RSO with NORAD ID 43925')
plt.grid(True)
plt.show()


# 

# From the plot, we can observe that the maximum probability of collision for the RSO with NORAD ID 43925 is mostly concentrated around lower values, indicating that the RSO has a lower risk of collision. However, there are a few instances where the maximum probability of collision is relatively high, suggesting that there are certain situations where the risk of collision for the RSO increases significantly.
# 
# 

# In[73]:


# Filter the relative speed of conjunctions for the RSO with NORAD ID 43925
relative_speed = rsodata['TCA_RELATIVE_SPEED']

# Plot the distribution of the relative speed of conjunctions
plt.figure(figsize=(10, 6))
sns.histplot(relative_speed, kde=True)
plt.xlabel('Relative Speed of Conjunctions (km/s)')
plt.ylabel('Frequency')
plt.title('Distribution of Relative Speed of Conjunctions for RSO with NORAD ID 43925')
plt.grid(True)
plt.show()


# The distribution appears to be skewed to the right, indicating that there are a few instances where the RSO has a higher relative speed of conjunctions.Relative speed of the conjuctions are very high on around 2 to 4 and 13 to 14 

# In[74]:


# Filter the range of conjunctions for the RSO with NORAD ID 43925
range_of_conjunctions = rsodata['TCA_RANGE']

# Plot the distribution of the range of conjunctions
plt.figure(figsize=(10, 6))
sns.histplot(range_of_conjunctions, kde=True)
plt.xlabel('Range of Conjunctions (km)')
plt.ylabel('Frequency')
plt.title('Distribution of Range of Conjunctions for RSO with NORAD ID 43925')
plt.grid(True)
plt.show()




# We can observe that the majority of conjunctions for the RSO with NORAD ID 43925 occur within a certain range. The distribution appears to be skewed to the right

# In[68]:


# Identifing the RSOs that have the highest number of conjunctions with the RSO with NORAD ID 43925
conjunctions_with_other_rso = rsodata['NORAD_CAT_ID_2'].value_counts().head(10)
conjunctions_with_other_rso


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




