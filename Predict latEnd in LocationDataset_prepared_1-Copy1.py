#!/usr/bin/env python
# coding: utf-8

# # Predicting latEnd in LocationDataset_prepared

# ### Notebook automatically generated from your model

# Model Ridge (L2) regression (s1), trained on 2026-03-14 20:07:40.

# #### Generated on 2026-03-19 06:24:06.846835

# prediction
# This notebook will reproduce the steps for a REGRESSION on  LocationDataset_prepared.
# The main objective is to predict the variable latEnd

# #### Warning

# The goal of this notebook is to provide an easily readable and explainable code that reproduces the main steps
# of training the model. It is not complete: some of the preprocessing done by the DSS visual machine learning is not
# replicated in this notebook. This notebook will not give the same results and model performance as the DSS visual machine
# learning model.

# Let's start with importing the required libs :

# In[ ]:


import sys
import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
import dataiku.core.pandasutils as pdu
from dataiku.doctor.preprocessing import PCA
from collections import defaultdict, Counter


# And tune pandas display options:

# In[ ]:


pd.set_option('display.width', 3000)
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)


# #### Importing base data

# The first step is to get our machine learning dataset:

# In[ ]:


# We apply the preparation that you defined. You should not modify this.
preparation_steps = []
preparation_output_schema = {'columns': [{'name': 'ID', 'type': 'bigint'}, {'name': 'eventTimeStart', 'type': 'date'}, {'name': 'day_of_week', 'type': 'bigint'}, {'name': 'hour', 'type': 'bigint'}, {'name': 'eventTimeEnd', 'type': 'date'}, {'name': 'time_difference', 'type': 'bigint'}, {'name': 'latStart', 'type': 'double'}, {'name': 'lonStart', 'type': 'double'}, {'name': 'latEnd', 'type': 'double'}, {'name': 'lonEnd', 'type': 'double'}, {'name': 'distance_moved', 'type': 'double'}, {'name': 'speed', 'type': 'double'}], 'userModified': False}

ml_dataset_handle = dataiku.Dataset('LocationDataset_prepared')
ml_dataset_handle.set_preparation_steps(preparation_steps, preparation_output_schema)
get_ipython().run_line_magic('time', 'ml_dataset = ml_dataset_handle.get_dataframe(limit = 100000)')

print ('Base data has %i rows and %i columns' % (ml_dataset.shape[0], ml_dataset.shape[1]))
# Five first records",
ml_dataset.head(5)


# #### Initial data management

# The preprocessing aims at making the dataset compatible with modeling.
# At the end of this step, we will have a matrix of float numbers, with no missing values.
# We'll use the features and the preprocessing steps defined in Models.
# 
# Let's only keep selected features

# In[ ]:


ml_dataset = ml_dataset[['hour', 'latEnd', 'eventTimeEnd', 'latStart', 'lonStart', 'time_difference', 'lonEnd', 'eventTimeStart', 'distance_moved', 'speed', 'day_of_week']]


# Let's first coerce categorical columns into unicode, numerical features into floats.

# In[ ]:


# astype('unicode') does not work as expected

def coerce_to_unicode(x):
    if sys.version_info < (3, 0):
        if isinstance(x, str):
            return unicode(x,'utf-8')
        else:
            return unicode(x)
    else:
        return str(x)


categorical_features = []
numerical_features = ['hour', 'eventTimeEnd', 'latStart', 'lonStart', 'time_difference', 'lonEnd', 'eventTimeStart', 'distance_moved', 'speed', 'day_of_week']
text_features = []
from dataiku.doctor.utils import datetime_to_epoch
for feature in categorical_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in text_features:
    ml_dataset[feature] = ml_dataset[feature].apply(coerce_to_unicode)
for feature in numerical_features:
    if ml_dataset[feature].dtype == np.dtype('M8[ns]') or (hasattr(ml_dataset[feature].dtype, 'base') and ml_dataset[feature].dtype.base == np.dtype('M8[ns]')):
        ml_dataset[feature] = datetime_to_epoch(ml_dataset[feature])
    else:
        ml_dataset[feature] = ml_dataset[feature].astype('double')


# We renamed the target variable to a column named target

# In[ ]:


ml_dataset['__target__'] = ml_dataset['latEnd']
del ml_dataset['latEnd']


# Remove rows for which the target is unknown.
ml_dataset = ml_dataset[~ml_dataset['__target__'].isnull()]


# #### Cross-validation strategy

# The dataset needs to be split into 2 new sets, one that will be used for training the model (train set)
# and another that will be used to test its generalization capability (test set)

# This is a simple cross-validation strategy.

# In[ ]:


train, test = pdu.split_train_valid(ml_dataset, prop=0.8)
print ('Train data has %i rows and %i columns' % (train.shape[0], train.shape[1]))
print ('Test data has %i rows and %i columns' % (test.shape[0], test.shape[1]))


# #### Features preprocessing

# The first thing to do at the features level is to handle the missing values.
# Let's reuse the settings defined in the model

# In[ ]:


drop_rows_when_missing = ['eventTimeEnd', 'eventTimeStart']
impute_when_missing = [{'feature': 'hour', 'impute_with': 'MEAN'}, {'feature': 'latStart', 'impute_with': 'MEAN'}, {'feature': 'lonStart', 'impute_with': 'MEAN'}, {'feature': 'time_difference', 'impute_with': 'MEAN'}, {'feature': 'lonEnd', 'impute_with': 'MEAN'}, {'feature': 'distance_moved', 'impute_with': 'MEAN'}, {'feature': 'speed', 'impute_with': 'MEAN'}, {'feature': 'day_of_week', 'impute_with': 'MEAN'}]

# Features for which we drop rows with missing values"
for feature in drop_rows_when_missing:
    train = train[train[feature].notnull()]
    test = test[test[feature].notnull()]
    print ('Dropped missing records in %s' % feature)

# Features for which we impute missing values"
for feature in impute_when_missing:
    if feature['impute_with'] == 'MEAN':
        v = train[feature['feature']].mean()
    elif feature['impute_with'] == 'MEDIAN':
        v = train[feature['feature']].median()
    elif feature['impute_with'] == 'CREATE_CATEGORY':
        v = 'NULL_CATEGORY'
    elif feature['impute_with'] == 'MODE':
        v = train[feature['feature']].value_counts().index[0]
    elif feature['impute_with'] == 'CONSTANT':
        v = feature['value']
    train[feature['feature']] = train[feature['feature']].fillna(v)
    test[feature['feature']] = test[feature['feature']].fillna(v)
    print ('Imputed missing values in feature %s with value %s' % (feature['feature'], coerce_to_unicode(v)))


# We can now handle the categorical features (still using the settings defined in Models):

# Let's rescale numerical features

# In[ ]:


rescale_features = {'hour': 'AVGSTD', 'eventTimeEnd': 'AVGSTD', 'latStart': 'AVGSTD', 'lonStart': 'AVGSTD', 'time_difference': 'AVGSTD', 'lonEnd': 'AVGSTD', 'eventTimeStart': 'AVGSTD', 'distance_moved': 'AVGSTD', 'speed': 'AVGSTD', 'day_of_week': 'AVGSTD'}
for (feature_name, rescale_method) in rescale_features.items():
    if rescale_method == 'MINMAX':
        _min = train[feature_name].min()
        _max = train[feature_name].max()
        scale = _max - _min
        shift = _min
    else:
        shift = train[feature_name].mean()
        scale = train[feature_name].std()
    if scale == 0.:
        del train[feature_name]
        del test[feature_name]
        print ('Feature %s was dropped because it has no variance' % feature_name)
    else:
        print ('Rescaled %s' % feature_name)
        train[feature_name] = (train[feature_name] - shift).astype(np.float64) / scale
        test[feature_name] = (test[feature_name] - shift).astype(np.float64) / scale


# #### Modeling

# Before actually creating our model, we need to split the datasets into their features and labels parts:

# In[ ]:


X_train = train.drop('__target__', axis=1)
X_test = test.drop('__target__', axis=1)

y_train = np.array(train['__target__'])
y_test = np.array(test['__target__'])


# Now we can finally create our model!

# In[ ]:


from sklearn.linear_model import RidgeCV
clf = RidgeCV(fit_intercept=True)


# ... And train the model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from dataiku.doctor.utils.skcompat import dku_fit  # Make the linear models regressors normalization compatible with all sklearn versions. Simply call clf.fit for non-linear models.\n\ndku_fit(clf, X_train, y_train)\n')


# Build up our result dataset

# In[ ]:


get_ipython().run_line_magic('time', '_predictions = clf.predict(X_test)')
predictions = pd.Series(data=_predictions, index=X_test.index, name='predicted_value')

# Build scored dataset
results_test = X_test.join(predictions, how='left')
results_test = results_test.join(test['__target__'], how='left')
results_test = results_test.rename(columns= {'__target__': 'latEnd'})


# #### Results

# You can measure the model's accuracy:

# In[ ]:


c =  results_test[['predicted_value', 'latEnd']].corr()
print ('Pearson correlation: %s' % c['predicted_value'][1])


# That's it. It's now up to you to tune your preprocessing, your algo, and your analysis !
# 
