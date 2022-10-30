#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import itertools
import joblib
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix
from dython.nominal import associations

pd.set_option("display.max_columns", 100)
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_colwidth", 1000)


# # Data Preparation

# In[2]:


csv_filename = "loan_data_2007_2014.csv"
feather_filename = "loan_data_2007_2014.feather"


# ## Data Reading
# Read data with `.csv` file, and save into `.feather` file

# In[3]:


if not os.path.exists(feather_filename):
    # Read csv file
    df = pd.read_csv("loan_data_2007_2014.csv")

    # change to feather
    df.to_feather("loan_data_2007_2014.feather")


# ## Move to Feather Extension
# For fastest reading the data

# In[4]:


raw_df = pd.read_feather(feather_filename)


# ## Data Overview
# 

# In[5]:


raw_df.head()


# # Data Description

# In[13]:


data_dict = pd.read_excel("LCDataDictionary1.xlsx", sheet_name="LoanStats")


# ## Description

# In[14]:


data_dict[["Colomn", "Description"]]


# ## DataFrame Check

# In[15]:


raw_df.info()


# # EDA

# In[16]:


title_font = dict(size=20, weight="bold")

def plot_count(df, y, title, **sns_kwargs):
    value_counts = df[y].value_counts()
    percentage = value_counts / value_counts.sum()
    percentage = percentage.apply("{:.2%}".format)

    plt.figure(figsize=(14, 10))
    plt.title(title, fontdict=title_font)
    sns.countplot(data=df, y=y, order=value_counts.index, **sns_kwargs)
    plt.ylabel("")
    plt.show()

    print(percentage)


def plot_distribution(df, x, title, **sns_kwargs):
    plt.figure(figsize=(14, 10))
    plt.title(title, fontdict=title_font)
    sns.histplot(data=df, x=x, kde=True, **sns_kwargs)
    plt.ylabel("")
    plt.show()


def plot_boxplot(df, x, y, title, **sns_kwargs):
    plt.figure(figsize=(14, 10))
    plt.title(title, fontdict=title_font)
    sns.boxplot(data=df, x=x, y=y, **sns_kwargs)
    plt.ylabel("")
    plt.show()


# ## Loan Status
# Most loans are in an "on going" status. At the success rate, loans that are successfully repaid have a higher ratio than loans that are overdue.

# In[84]:


plot_count(raw_df, title="Loan Status",  y="loan_status")


# ## Determining Loan Status
# Our goal here is to determine which loans are likely to default, so that the categories we can take are between those that are successful and those that fail. Therefore we will only take 2 categories namely:
# - Approved, consisting of: Fully Paid
# - Rejected, consisting of: Charged Off, Default, and Does not meet the credit policy
# 
# We can't use `current` because the loan is still ongoing, as well as `late` and `in_grace_period`.

# In[85]:


# Decide which category to use
approved_cat = ["Fully Paid"]
dissaproved_cat = [
    
    "Charged Off",
    "Default",
    "Does not meet the credit policy. Status:Fully Paid",
    "Does not meet the credit policy. Status:Charged Off"
]


# In[86]:


# change to a new label
def label_loan_status(value):
    if value in approved_cat:
        return 1
    return 0

label_loan_status("Fully Paid")


# In[87]:


# Filter and apply function
inter_df = raw_df[raw_df["loan_status"].isin(approved_cat + dissaproved_cat)].copy()
inter_df["loan_status"] = inter_df["loan_status"].apply(label_loan_status)


# In[88]:


plot_count(inter_df, y="loan_status", title="Loan Status")


# ## Correlation between Variables

# In[89]:


# Calculate the correlation each variables
correlations = (inter_df.select_dtypes(exclude=object)
                         .corr()
                         .dropna(how="all", axis=0)
                         .dropna(how="all", axis=1)
)


# In[23]:


correlations["loan_status"].abs().sort_values(ascending=False)


# In[90]:


# Filter correlation between vmin - vmax
vmin, vmax = 0.1, 0.99

unstack_corr = correlations.unstack()
pos_corr = (unstack_corr > vmin) & (unstack_corr < vmax)
neg_corr = (unstack_corr > -vmax) & (unstack_corr < -vmin)
high_corr = unstack_corr[pos_corr | neg_corr]

trimmed_corr = high_corr.sort_values(ascending=False).unstack()


# In[91]:


# Create a mask to form the lower triangular matrix
mask = np.zeros_like(trimmed_corr)
mask[np.triu_indices_from(mask)] = True


# In[92]:


# Show heatmap
plt.figure(figsize=(20, 20))
plot = sns.heatmap(
    trimmed_corr, 
    annot=True, 
    mask=mask,
    fmt=".2f", 
    cmap="viridis", 
    annot_kws={"size": 14})

plot.set_xticklabels(plot.get_xticklabels(), size=18)
plot.set_yticklabels(plot.get_yticklabels(), size=18)
plt.show()


# From the heatmap above, there are several variables that have an influence on loan status, including:

# In[27]:


affect_loan = high_corr.loc["loan_status"].abs().sort_values(ascending=False)
affect_loan


# While the features that are correlated with the previous one we need to identify. We use a limit value of 0.9 to look for features that are strongly correlated.

# In[28]:


threshold = 0.9
affect_collision = (high_corr.abs()
                             .loc[high_corr > threshold]
                             .loc[affect_loan.index, affect_loan.index]
                             .sort_values(ascending=False)
)
affect_collision


# Based on the magnitude of its influence on the loan status, the correlated features will be selected based on the most influential.

# In[29]:


left_index = affect_collision.index.get_level_values(0)
right_index = affect_collision.index.get_level_values(1)

def remove_collide_index(left_index, right_index):
    include, exclude = [], []

    for left, right in zip(left_index, right_index):
        if left not in include and left not in exclude:
            include.append(left)
        if right not in include and right not in exclude:
            exclude.append(right)
        
    return include, exclude


include_affect_col, exclude_affect_col = remove_collide_index(left_index, right_index)
include_affect_col, exclude_affect_col


# Influential numeric features that we will use

# In[30]:


affect_num_cols = affect_loan[~affect_loan.index.isin(exclude_affect_col)].index.to_list()
affect_num_cols


# ## Loan Status and Principal Loan Size Paid
# Principal is the amount of the principal loan lent to the debtor. In other words, it is the original amount of money lent. Most people who experience default have not been able to pay the principal payment until maturity, it can be seen from the distribution of payments below. The average is almost 0.

# In[32]:


plot_distribution(df=inter_df, x="total_rec_prncp", hue="loan_status", title="")


# ## Loan Status and Total Unbilled Money
# Charged off recoveries are the total money that cannot be paid to the borrowing company because the maturity period has passed so that the borrowing company can release the right to collect the debt by selling it to another company. From this it is clear that it is people with bad loan status who have the most charge off recoveries.

# In[33]:


plot_distribution(df=inter_df, x="recoveries", hue="loan_status", title="")


# ## Loan Status and Loan Amount
# The average loan amount is in the 'bad' status.

# In[93]:


x, y = "loan_status", "loan_amnt"
plot_boxplot(df=inter_df, x=x, y=y, title="Total Loan Distribution")
inter_df.groupby(x)[y].describe()


# ## Loan Status and Total Payments Received
# It is clear that the highest total payments are on loans with 'good' status.

# In[94]:


x, y = "loan_status", "total_pymnt"
plot_boxplot(df=inter_df, x=x, y=y, title="Distribution of Total Payments Received")
inter_df.groupby(x)[y].describe()


# ## Purpose of Taking a Loan
# More than half of borrowers have a goal of closing previous loans. When viewed from the type, the purpose for consumption is more than the purpose for business, renovation and education.

# In[95]:


plot_count(inter_df, y="purpose", title="Loan Purpose")


# ## Borrower's Country of Origin
# Most of the borrowers come from California.
# 

# In[96]:


plot_count(df=inter_df, y="addr_state", title="Borrower's Country of Origin")


# ## Loan Rate
# Loans are graded from the letters of the alphabet A to G, the closer to G the higher the interest rate paid.

# In[97]:


x, y = "int_rate", "grade"
order = inter_df[y].sort_values().unique()
plot_boxplot(inter_df, x=x, y=y, title="Loan Rate", order=order)
plot_count(df=inter_df, y=y, title="")
inter_df.groupby(y)[x].describe()


# ## Home Ownership Status
# Most of the borrowers delegate their house as collateral for the loan, while only a few of the borrowers own their own house.

# In[39]:


y = "home_ownership"
order = inter_df[y].sort_values().unique()
plot_count(df=inter_df, y=y, title="")


# # Data Preprocessing

# ## Eliminate unused features
# After seeing the info and description of the data above, there are features that we don't need to use because they are not so significant to be used as features in predictions.

# In[98]:


# Detailed information about data columns and rows
data_stat = pd.DataFrame()
data_stat.index = inter_df.columns
data_stat["unique_value"] = inter_df.nunique()
data_stat["missing_rate"] = inter_df.isna().mean()
data_stat["dtype"] = inter_df.dtypes
data_stat


# Column with unusable data

# In[99]:


# Column where all data is missing
miss_col = data_stat[data_stat["missing_rate"] == 1].index.to_list()
print("Column where all data is missing:")
print(miss_col)
print()

# Column where all data is very unique
vari_col = data_stat[data_stat["unique_value"] == inter_df.shape[0]].index.to_list()
print("Column where all data is very unique:")
print(vari_col)
print()

# Column with many categorizes
cat_col_stat = data_stat[data_stat["dtype"] == "object"]
vari_cat_col = cat_col_stat[cat_col_stat["unique_value"] > 1000].index.to_list()
print("Column with many categorizes:")
print(vari_cat_col)
print()

# Column with only one value
single_valued_col = data_stat[data_stat["unique_value"] == 1].index.to_list()
print("Column with only one value:")
print(single_valued_col)
print()

removed_features = miss_col + vari_col + vari_cat_col + single_valued_col


# In[100]:


# Delete Unused Features
pre_df = inter_df.loc[:, ~inter_df.columns.isin(removed_features)].copy()
pre_df.shape


# ## Categorical or object Features

# In[101]:


# Column with object or categorical value
cat_features = pre_df.select_dtypes(include=object).columns
cat_features


# ### Column with a date value

# In[44]:


date_cols = ["issue_d", "earliest_cr_line", "last_pymnt_d", "last_credit_pull_d", "next_pymnt_d"]

for col in date_cols:
    print(pre_df[col].value_counts().iloc[:5])
    print()


# There is no strong correlation between dates and each date has little correlation with loan status. However, we will remove the date feature that correlates less than 0.1 with the loan status

# In[102]:


# Correlation between dates and loan status
used_cols = date_cols + ["loan_status"] 
complete_correlation = associations(
    pre_df[used_cols], 
    filename='date_correlation.png',
    figsize=(10,10)
)


# In[104]:


# Date's features or column that we will use
affect_date_cols = ["issue_d", "last_pymnt_d", "last_credit_pull_d", "next_pymnt_d"]
affect_date_cols


# In[105]:


# Delete date's colum or features where does'nt have a powerfull correlation with loan status
unused_cols = ["earliest_cr_line"]
pre_df = pre_df.drop(columns=unused_cols, errors="ignore")
pre_df.head()


# ### Unused Categorical Columns

# In[48]:


other_cat_cols = cat_features[~cat_features.isin(date_cols)]
other_cat_cols


# In[49]:


pre_df.loc[:, other_cat_cols].head()


# Some unused categorical columns are:
# - desc and title because they are text.
# - zip_code because the 3 digits behind it are censored
# - sub_grade because it already has a similar column, namely grade

# In[50]:


unused_cols = ["desc", "zip_code", "sub_grade", "title"]
pre_df = pre_df.drop(columns=unused_cols, errors="ignore")
pre_df.head()


# In[51]:


other_cat_cols = cat_features[~cat_features.isin(date_cols + unused_cols)]
other_cat_cols


# There is a strong correlation between emp_title and loan status, followed by grade and term. Other less influential features will not be used for predictions.

# In[106]:


# Correlation between categorical features and loan status
used_cols = other_cat_cols.to_list() + ["loan_status"]
complete_correlation = associations(
    pre_df[used_cols], 
    filename='cat_correlation.png',
    figsize=(10,10)
)


# The grade and term features have little correlation with the loan status.

# In[107]:


#Categorical features we will use
affect_cat_cols = ["grade", "term"]
affect_cat_cols


# In[108]:


# Remove less influential features
used_cols = ["emp_title", "grade", "term"]
unused_cols = other_cat_cols[~other_cat_cols.isin(used_cols)]
pre_df = pre_df.drop(columns=unused_cols, errors="ignore")
pre_df.head()


# ## Target correlated features

# In[109]:


# The columns we will use
predictor_cols = affect_num_cols + affect_cat_cols + affect_date_cols
predictor_cols


# ## Imputation Missing Value
# The next_pyment_d feature has the most missing value because it is possible that borrowers who have paid off their debts will no longer have a repayment schedule.

# In[56]:


pre_df[predictor_cols].isna().mean().sort_values(ascending=False)


# In[110]:


# Fill data with "no"
pre_df["next_pymnt_d"] = pre_df["next_pymnt_d"].fillna("no")
top_next_pyment_d = pre_df["next_pymnt_d"].value_counts().head()
top_next_pyment_d


# Do the same for the last_pymnt_d and last_credit_pull_d columns

# In[58]:


pre_df["last_pymnt_d"] = pre_df["last_pymnt_d"].fillna("no")
pre_df["last_credit_pull_d"] = pre_df["last_credit_pull_d"].fillna("no")


# Fill in missing value numeric data using mode value

# In[59]:


mode = pre_df["inq_last_6mths"].mode().values[0]
pre_df["inq_last_6mths"] = pre_df["inq_last_6mths"].fillna(mode)


# Check again if there is still missing data

# In[60]:


pre_df[predictor_cols].isna().mean().sort_values(ascending=False)


# # Modelling

# ## Define Labels and Data Features
# The label is the performance level of the loan which is in the `loan_status` column. Since the column has several categories, we have selected and combined them into 2 categories namely `good` and `bad`.

# Previously, we need to separate labels and features from the data so that we can then separate the data.

# In[61]:


label = pre_df["loan_status"].copy()
features = pre_df[predictor_cols].copy()

print("Label shape:")
print(label.shape)

print("Features shape:")
print(features.shape)


# ## Train Plot

# ### Preprocessing

# In[62]:


num_features = features.select_dtypes(exclude="object")
cat_features = features.select_dtypes(include="object")


# In[111]:


# Normalization numeric features
num_features = (num_features - num_features.mean()) / num_features.std()
num_features


# In[112]:


# OneHotEncode Categoric features
cat_features = pd.get_dummies(cat_features)
cat_features


# In[113]:


# combine features
features_full = pd.concat([num_features, cat_features], axis=1)


# In[66]:


features_full.shape


# ### Data Separate

# In[67]:


X_train, X_test, y_train, y_test = train_test_split(features_full, label, test_size=0.2, random_state=42, stratify=label)


# In[68]:


X_train.shape, y_train.shape


# ### Modeling

# In[69]:


logres = LogisticRegression(max_iter=500, solver="sag", class_weight="balanced", n_jobs=-1)
logres


# In[73]:


logres.fit(X_train, y_train)


# ## Save modelling

# In[74]:


joblib.dump(logres, "logres.z")


# In[75]:


logres = joblib.load("logres.z")


# # Model Evaluation

# ## Model Baseline
# We will make the simplest prediction model by predicting all the data for the most categories. This is done so that we get a benchmark, what is the minimum performance that our machine learning model must pass later.

# In[76]:


test_label_counts = y_test.value_counts()
test_label_counts


# In[77]:


test_label_counts.max() / test_label_counts.sum()


# ## Matrix classification

# ## Train

# In[78]:


logres.score(X_train, y_train)


# In[79]:


report = classification_report(y_true=y_train, y_pred=logres.predict(X_train))
print(report)


# ## Test

# In[80]:


logres.score(X_test, y_test)


# In[81]:


report = classification_report(y_true=y_test, y_pred=logres.predict(X_test))
print(report)


# ## Confusion Matrix

# In[82]:


conf = confusion_matrix(y_true=y_test, y_pred=logres.predict(X_test))


# In[83]:


plt.figure(figsize=(10, 10))
sns.heatmap(conf, annot=True, fmt="g")
plt.show()


# In[ ]:




