# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predicting the sale price of Bulldozers using Machine Learning
# In thins notebook, we're going to go through an example machine learning project with the goal of predicting the sale price of bulldozers.  
# ## 1. Problem defition
# > How well can we predict the future sale price of a bulldozer, given bulldozers have been sold for?
# ## 2. Data
# The data is downloaded from the Kaggle Bulebook for Bulldozers compertition:  https://www.kaggle.com/c/bluebook-for-bulldozers/data    
# There are 3 main datasets:
#
# * Train.csv is the training set, which contains data through the end of 2011.
# * Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
# * Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.  
#
# ## 3. Evaluation
# The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.  
# For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation  
# **Note:** The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project be to build a machine learning model which minimises RMSLE.  
#
# ## 4. Features  
# Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary on Google Sheets: https://docs.google.com/spreadsheets/d/1jykeJFBVQPb350mZzoRO0fZdWcgDkt3B/edit#gid=1967137473

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()


# %%
# Import training and validation sets
df = pd.read_csv('data/TrainAndValid.csv', 
                 low_memory=False)

# %%
df.info()

# %%
df.isna().sum()

# %%
df.columns

# %%
fig, ax = plt.subplots()
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000]);

# %%
df.saledate[:1000]

# %%
df.SalePrice.plot.hist();

# %% [markdown]
# ### Parsing dates
# When we work with time series data, we want to enrich the time & date component as much as possible.  
#   
# We can do thay by telling pandas which of our columns has dates in it using the `parse_dates` parameter.

# %%
# Import data ag ain but this time parse dates
df = pd.read_csv('data/TrainAndValid.csv',
                 low_memory=False,
                 parse_dates=['saledate'])

# %%
df.saledate.dtype

# %%
df.saledate[:1000]

# %%
fig, ax = plt.subplots()
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000]);

# %%
df.head().T

# %% [markdown]
# ### Sort DataFrame by saledate
# When working with time series data, it's a good idea to sort it by date.

# %%
# Sort DataFrame in date order
df.sort_values(by=['saledate'], inplace=True, ascending=True)
df.saledate.head(20)

# %% [markdown]
# ### Make a copy of the original DataFrame
# We make a copy of the original dataframe so when we manipulate the copy, we've still got our original data.

# %%
# Make a copy
df_tmp = df.copy()

# %%

# %% [markdown]
# Add datetime parameters for `saledate` column

# %%
df_tmp['saleYear'] = df_tmp.saledate.dt.year
df_tmp['saleMonth'] = df_tmp.saledate.dt.month
df_tmp['saleDay'] = df_tmp.saledate.dt.day
df_tmp['saleDayOfWeek'] = df_tmp.saledate.dt.dayofweek
df_tmp['saleDayOfYear'] = df_tmp.saledate.dt.dayofyear

# %%
df_tmp[:1].T

# %%
# Now we've enriched our DataFrame with date time features, we can remove `saledate`
df_tmp.drop('saledate', axis=1, inplace=True)

# %%
# Check the values of different columns
df_tmp.state.value_counts()

# %% [markdown]
# ## 5.Modelling
# We've done enough EDA (we could always do more) but let's start to do some model-driven EDA.
# Exploratory Data Analysis

# %%
# Let's vuild a machine learning model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42)

# %% [markdown]
# ### Convert string to categories
# One way we can turn all of our data into numbers is by converting them into pandas catgories.  
# We can check the different compatible with pandas here:   
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_object_dtype.html

# %%
pd.api.types.is_string_dtype(df_tmp['UsageBand'])

# %%
# Find the columns which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)

# %%
# This will turn all of the string value into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()

# %%
df_tmp.info()

# %%
# 類別, 類別計數, 類別代碼
df_tmp.state.cat.categories, df_tmp.state.value_counts(), df_tmp.state.cat.codes

# %% [markdown]
# Thanks to pandas Categories we now have a way to access all of our data in the form of nymbers.
# But we still have a bynch of missing data...

# %%
# Check missing data
df_tmp.isnull().sum()/len(df_tmp)

# %% [markdown]
# ### Save preprocessed data

# %%
# Export current tmp dataframe
df_tmp.to_csv('data/train_tmp.csv',
              index=False)

# %%
# Import preprocessed data
df_tmp = pd.read_csv('data/train_tmp.csv',
                     low_memory=False)
df_tmp.head().T

# %% [markdown]
# ## Fill missing values
# ### Fill numerical misssing values first

# %%
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)

# %%
# Check for which numeric columns have null values
for label, content in df_tmp.items():
    # data 為數值的欄位
    if pd.api.types.is_numeric_dtype(content):
        # 加總各column的 null, 如果大於 0 就 print
        if pd.isnull(content).sum():
            print(label)


# %%
# Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label + '_is_missing'] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())


# %%
# Demonstrate how median is more robust than mean
hundreds = np.full((1000, ), 100)
hundreds_billion = np.append(hundreds, 1000000000)
np.mean(hundreds), np.mean(hundreds_billion), np.median(hundreds), np.median(hundreds_billion)

# %%
# Check if there's any null numeric values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)

# %%
# Check to see how many examples missing
df_tmp.auctioneerID_is_missing.value_counts()

# %% [markdown]
# ### Filling and turning categorical variables into numbers

# %%
# Check for columns which aren't numeric
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)


# %%
# Turn categorical variables into number and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label + '_is_missing'] = pd.isnull(content)
        # Turn categories into numbers and add +1
        # Categorical 會將 missing data 分配在 -1 所以將全部 +1
        df_tmp[label] = pd.Categorical(content).codes +1


# %%
df_tmp.info()

# %%
df_tmp.head().T

# %%
df_tmp.isna().sum()

# %% [markdown]
# Now that all of ata is numeric as well as our dataframe has no missing values, we should be able to build a machine learning model.

# %%
df_tmp.head()

# %%
len(df_tmp)

# %%
# %%time
# Instantiate model
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42)

# Fit the model
model.fit(df_tmp.drop('SalePrice', axis=1), df_tmp['SalePrice'])

# %%
# Score the model
model.score(df_tmp.drop('SalePrice', axis=1), df_tmp['SalePrice'])

# %% [markdown]
# **Question:** Why doesn't the above metric hold water? (why isn't the metric reliable)

# %% [markdown]
# ### Splitting data into train/validation sets

# %%
df_tmp.head()

# %%
# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]
len(df_val), len(df_train)

# %%
#Split data into X & y
X_train, y_train = df_train.drop('SalePrice', axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop('SalePrice', axis=1), df_val.SalePrice

# %%
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape

# %% [markdown]
# ### Building an evaluation function

# %%
# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    '''
    Caculates root mean squared log error between predictions and
    true labels.
    '''
    return np.sprt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluation model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {'Training MAE': mean_absolute_error(y_train, train_preds),
              'Valid MAE': mean_absolute_error(y_train, val_preds),
              'Training RMSLE': rmesl(y_train, train_preds),
              'Valid RMSLE': rmsle(y_train, val_preds),
              'Training R^2': r2_score(y_train, train_preds),
              'Valid R^2': r2_score(y_train, val_preds)}
    return scores

# %%

# %%

# %%
