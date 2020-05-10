# -*- coding: utf-8 -*-
---
jupyter:
  jupytext:
    formats: ipynb,py:percent,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Predicting the sale price of Bulldozers using Machine Learning
In thins notebook, we're going to go through an example machine learning project with the goal of predicting the sale price of bulldozers.  
## 1. Problem defition
> How well can we predict the future sale price of a bulldozer, given bulldozers have been sold for?
## 2. Data
The data is downloaded from the Kaggle Bulebook for Bulldozers compertition:  https://www.kaggle.com/c/bluebook-for-bulldozers/data    
There are 3 main datasets:

* Train.csv is the training set, which contains data through the end of 2011.
* Valid.csv is the validation set, which contains data from January 1, 2012 - April 30, 2012 You make predictions on this set throughout the majority of the competition. Your score on this set is used to create the public leaderboard.
* Test.csv is the test set, which won't be released until the last week of the competition. It contains data from May 1, 2012 - November 2012. Your score on the test set determines your final rank for the competition.  

## 3. Evaluation
The evaluation metric for this competition is the RMSLE (root mean squared log error) between the actual and predicted auction prices.  
For more on the evaluation of this project check: https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation  
**Note:** The goal for most regression evaluation metrics is to minimize the error. For example, our goal for this project be to build a machine learning model which minimises RMSLE.  

## 4. Features  
Kaggle provides a data dictionary detailing all of the features of the dataset. You can view this data dictionary on Google Sheets: https://docs.google.com/spreadsheets/d/1jykeJFBVQPb350mZzoRO0fZdWcgDkt3B/edit#gid=1967137473

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.impute import SimpleImputer
# from sklearn.preprocessing import OneHotEncoder
from jupyterthemes import jtplot
from IPython.core.display import display, HTML

display(HTML("<style>.container { width:80% !important; }</style>"))
jtplot.style()
```


```python
# Import training and validation sets
df = pd.read_csv('data/TrainAndValid.csv', 
                 low_memory=False)
```

```python
df.info()
```

```python
df.isna().sum()
```

```python
df.columns
```

```python
fig, ax = plt.subplots()
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000]);
```

```python
df.saledate[:1000]
```

```python
df.SalePrice.plot.hist();
```

### Parsing dates
When we work with time series data, we want to enrich the time & date component as much as possible.  
  
We can do thay by telling pandas which of our columns has dates in it using the `parse_dates` parameter.

```python
# Import data ag ain but this time parse dates
df = pd.read_csv('data/TrainAndValid.csv',
                 low_memory=False,
                 parse_dates=['saledate'])
```

```python
df.saledate.dtype
```

```python
df.saledate[:1000]
```

```python
fig, ax = plt.subplots()
ax.scatter(df['saledate'][:1000], df['SalePrice'][:1000]);
```

```python
df.head().T
```

### Sort DataFrame by saledate
When working with time series data, it's a good idea to sort it by date.

```python
# Sort DataFrame in date order
df.sort_values(by=['saledate'], inplace=True, ascending=True)
df.saledate.head(20)
```

### Make a copy of the original DataFrame
We make a copy of the original dataframe so when we manipulate the copy, we've still got our original data.

```python
# Make a copy
df_tmp = df.copy()
```

Add datetime parameters for `saledate` column

```python
df_tmp['saleYear'] = df_tmp.saledate.dt.year
df_tmp['saleMonth'] = df_tmp.saledate.dt.month
df_tmp['saleDay'] = df_tmp.saledate.dt.day
df_tmp['saleDayOfWeek'] = df_tmp.saledate.dt.dayofweek
df_tmp['saleDayOfYear'] = df_tmp.saledate.dt.dayofyear
```

```python
df_tmp[:1].T
```

```python
# Now we've enriched our DataFrame with date time features, we can remove `saledate`
df_tmp.drop('saledate', axis=1, inplace=True)
```

```python
# Check the values of different columns
df_tmp.state.value_counts()
```

## 5.Modelling
We've done enough EDA (we could always do more) but let's start to do some model-driven EDA.
Exploratory Data Analysis

```python
# Let's vuild a machine learning model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42)
```

### Convert string to categories
One way we can turn all of our data into numbers is by converting them into pandas catgories.  
We can check the different compatible with pandas here:   
https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.types.is_object_dtype.html

```python
pd.api.types.is_string_dtype(df_tmp['UsageBand'])
```

```python
# Find the columns which contain strings
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        print(label)
```

```python
# This will turn all of the string value into category values
for label, content in df_tmp.items():
    if pd.api.types.is_string_dtype(content):
        df_tmp[label] = content.astype('category').cat.as_ordered()
```

```python
df_tmp.info()
```

```python
# 類別, 類別計數, 類別代碼
df_tmp.state.cat.categories, df_tmp.state.value_counts(), df_tmp.state.cat.codes
```

Thanks to pandas Categories we now have a way to access all of our data in the form of nymbers.
But we still have a bynch of missing data...

```python
# Check missing data
df_tmp.isnull().sum()/len(df_tmp)
```

### Save preprocessed data

```python
# Export current tmp dataframe
df_tmp.to_csv('data/train_tmp.csv',
              index=False)
```

```python
# Import preprocessed data
df_tmp = pd.read_csv('data/train_tmp.csv',
                     low_memory=False)
df_tmp.head().T
```

## Fill missing values
### Fill numerical misssing values first

```python
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        print(label)
```

```python
# Check for which numeric columns have null values
for label, content in df_tmp.items():
    # data 為數值的欄位
    if pd.api.types.is_numeric_dtype(content):
        # 加總各column的 null, 如果大於 0 就 print
        if pd.isnull(content).sum():
            print(label)
```


```python
# # Fill numeric rows with the median
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            # Add a binary column which tells us if the data was missing or not
            df_tmp[label + '_is_missing'] = pd.isnull(content)
            # Fill missing numeric values with median
            df_tmp[label] = content.fillna(content.median())
```


```python
# Demonstrate how median is more robust than mean
hundreds = np.full((1000, ), 100)
hundreds_billion = np.append(hundreds, 1000000000)
np.mean(hundreds), np.mean(hundreds_billion), np.median(hundreds), np.median(hundreds_billion)
```

```python
# Check if there's any null numeric values
for label, content in df_tmp.items():
    if pd.api.types.is_numeric_dtype(content):
        if pd.isnull(content).sum():
            print(label)
```

### Filling and turning categorical variables into numbers

```python
# Check for columns which aren't numeric
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        print(label)
```


```python
# Turn categorical variables into number and fill missing
for label, content in df_tmp.items():
    if not pd.api.types.is_numeric_dtype(content):
        # Add binary column to indicate whether sample had missing value
        df_tmp[label + '_is_missing'] = pd.isnull(content)
        # Turn categories into numbers and add +1
        # Categorical 會將 missing data 分配在 -1 所以將全部 +1
        df_tmp[label] = pd.Categorical(content).codes +1
```


### Splitting data into train/validation sets

```python
df_tmp.head()
```

```python
df_tmp.info()
```

```python
# Split data into training and validation
df_val = df_tmp[df_tmp.saleYear == 2012]
df_train = df_tmp[df_tmp.saleYear != 2012]
len(df_val), len(df_train)
```

```python
model = RandomForestRegressor(n_jobs=-1, 
                              random_state=42)
```

```python
#Split data into X & y
X_train, y_train = df_train.drop('SalePrice', axis=1), df_train.SalePrice
X_valid, y_valid = df_val.drop('SalePrice', axis=1), df_val.SalePrice
```

```python
X_train.shape, y_train.shape, X_valid.shape, y_valid.shape
```

Now that all of data is numeric as well as our dataframe has no missing values, we should be able to build a machine learning model.

```python
%%time
# Fit the model
model.fit(df_tmp.drop('SalePrice', axis=1), df_tmp['SalePrice'])
```

```python
# Score the model
model.score(df_tmp.drop('SalePrice', axis=1), df_tmp['SalePrice'])
```

**Question:** Why doesn't the above metric hold water? (why isn't the metric reliable)


### Building an evaluation function

```python
# Create evaluation function (the competition uses RMSLE)
from sklearn.metrics import mean_squared_log_error, mean_absolute_error, r2_score

def rmsle(y_test, y_preds):
    '''
    Caculates root mean squared log error between predictions and
    true labels.
    '''
    return np.sqrt(mean_squared_log_error(y_test, y_preds))

# Create function to evaluation model on a few different levels
def show_scores(model):
    train_preds = model.predict(X_train)
    val_preds = model.predict(X_valid)
    scores = {'Training MAE': mean_absolute_error(y_train, train_preds),
              'Valid MAE': mean_absolute_error(y_valid, val_preds),
              'Training RMSLE': rmsle(y_train, train_preds),
              'Valid RMSLE': rmsle(y_valid, val_preds),
              'Training R^2': r2_score(y_train, train_preds),
              'Valid R^2': r2_score(y_valid, val_preds)}
    return scores
```

## Testing our model on a subset (to tune the hyperparameters)

```python
len(X_train)
```

```python
# Change max_samples value
model = RandomForestRegressor(n_jobs=-1,
                              random_state=42,
                              max_samples=10000)
```

```python
%%time
# Cutting down on the max number of samples each estimator can see improves training time
model.fit(X_train, y_train)
```

```python
 show_scores(model)
```

### Hyerparameter tuning with RandomizedSearchCV

```python
%%time 
from sklearn.model_selection import RandomizedSearchCV

# Different RandomForestRegressor hyperparameters
rf_grid = {'n_estimators': np.arange(10, 100, 10),
           'max_depth': [None, 3, 5, 10],
           'min_samples_split': np.arange(2, 20, 2),
           'min_samples_leaf': np.arange(1, 20 ,2),
           'max_features': [0.5, 1, 'sqrt', 'auto'],
           'max_samples':[10000]}

# Instantiate RandomizedSearchCV model
rs_model = RandomizedSearchCV(RandomForestRegressor(n_jobs=-1,
                                                    random_state=42),
                              param_distributions=rf_grid,
                              n_iter=2,
                              cv=5,
                              verbose=True)

# Fit the RandomizedSearchCV model
rs_model.fit(X_train, y_train)
```

```python
# Find the best model hyperparameters
rs_model.best_params_
```

```python
# Evaluate the RandomizedSearch model
show_scores(rs_model)
```

## Train a model with the best hyperparamters
**Note:** These were found after 100 iterations of `RandomizedSearchCV`

```python
%%time
# Most ideal hyperparameters
ideal_model = RandomForestRegressor(n_estimators=40,
                                    min_samples_leaf=1,
                                    min_samples_split=14,
                                    max_features=0.5,
                                    n_jobs=-1,
                                    max_samples=None,
                                    random_state=42)
# Fix the ideal model
ideal_model.fit(X_train, y_train)
```

```python
# Scores for ideal_model (trained on all the data)
show_scores(ideal_model)
```

```python
# Scores on re_model (only trained on -10,000 examples)
show_scores(rs_model)
```

### Make predictions on test data

```python
# Import the test data
df_test = pd.read_csv('data/Test.csv',
                      low_memory=False,
                      parse_dates=['saledate'])
df_test.head()
```

### Preprocessing the data (getting the test dataset in the same format as our training dataset)

```python
def preprocess_data(df):
    '''
    Performs transformations on df and returns transformed df.
    '''
    df['saleYear'] = df.saledate.dt.year
    df['saleMonth'] = df.saledate.dt.month
    df['saleDay'] = df.saledate.dt.day
    df['saleDayOfWeek'] = df.saledate.dt.dayofweek
    df['saleDayOfYear'] = df.saledate.dt.dayofyear
    
    df.drop('saledate', axis=1, inplace=True)
    
    # # Fill numeric rows with the median
    for label, content in df.items():
        if pd.api.types.is_numeric_dtype(content):
            if pd.isnull(content).sum():
                # Add a binary column which tells us if the data was missing or not
                df[label + '_is_missing'] = pd.isnull(content)
                # Fill missing numeric values with median
                df[label] = content.fillna(content.median())
        else:
            # Add binary column to indicate whether sample had missing value
            df[label + '_is_missing'] = pd.isnull(content)
            # Turn categories into numbers and add +1
            # Categorical 會將 missing data 分配在 -1 所以將全部 +1
            df[label] = pd.Categorical(content).codes +1
    return df
```

```python
# Process the test data
df_test = preprocess_data(df_test)
df_test.head()
```

Finally now our test dataframe has the same features as our training dataframe, we can make predictions!

```python
# Find how the columns differ using sets
set(X_train.columns) - set(df_test.columns)
```

```python
# Match test dataset columns to training dataset
df_test['auctioneerID_is_missing'] = False
df_test.head()
```

```python
# Make predictions on the test data
test_preds = ideal_model.predict(df_test)
```

```python
# Create DataFrame compatible with Kaggle submission requirements
df_preds = pd.DataFrame()
df_preds['SalesID'] = df_test.SalesID
df_preds['SalePrice'] = test_preds
df_preds
```

### Feature importance

Feature importance seeks to figure out which different attributes of the data were most importance.

  When it comes to predicting the **target variable** (SalePrice).

```python
# Find feature importance of our best model
ideal_model.feature_importances_
```

```python
# Helper function for plotting feature importance
def plot_features(columns, importances, n=20):
    df = (pd.DataFrame({'features': columns,
                        'feature_importances': importances})
          .sort_values('feature_importances', ascending=False)
          .reset_index(drop=True))
    # Plot the dataframe
    fig, ax = plt.subplots()
    ax.barh(df['features'][:n], df['feature_importances'][:20])
    ax.set_ylabel('Features')
    ax.set_xlabel('Feature importance')
    ax.invert_yaxis()
```

```python
plot_features(X_train.columns, ideal_model.feature_importances_)
```

```python
df['ProductSize'].value_counts()
```
