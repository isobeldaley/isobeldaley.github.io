---
layout: post
title:      "Outliers: A Story of Over-Zealous Exclusion?"
date:       2019-08-04 21:50:19 +0000
permalink:  outliers_a_story_of_over-zealous_exclusion
---


##### During a recent project, I was asked to create a model to predict house prices using data from King County, Washington. Whilst completing an initial exploration, I came across a house that allegedly contained 33 bedrooms, but only 2 bathrooms.  Imagine the queues for a shower in the morning.  As I dug deeper, it became clear that the data was suffering from an illness : outliers.  This post explores the maladie, the cures, the side-effects, and provides a real-life example (that did not go to plan).  


## What is an Outlier?

In life, we're all outliers.  That's a great thing.  Compile an exhaustive list of your characteristics and you'll find that along one dimension or another, you stand out from the crowd.  It's this diversity that keeps life interesting.  

For data scientists, however, an outlier is not always  welcome news.  Defined as a data point that is either much bigger or much smaller than the other datapoints, outliers can wreak havoc with our models.  

## Why are they a problem?

Suppose you've been asked to calculate the mean of the following dataset:

```
data = [5, 4, 6, 3, 7, 2, 6, 2, 5, 7, 3, 999, 2, 3]
```

From a visual inspection of this data, it is clear that one value is significantly different from the others.  What should you do? 

To decide, let's first calculate the mean, including the outlier, and then see how it differs when excluded:

```
data = np.array([5, 4, 6, 3, 7, 2, 6, 2, 5, 7, 3, 999, 2, 3])
np.mean(data)

data_no_outlier = np.array([5, 4, 6, 3, 7, 2, 6, 2, 5, 7, 3, 2, 3])
np.mean(data_no_outlier)
```

When the 999 outlier is included, the mean is calculated as 75.3.  Excluding the outlier, the mean is calculated as 4.23.    From this very simple example, we can see that the outlier has a disproportionate effect on the mean.  

Similarly, when creating more complex models, outliers can exert an inordinate influence on the estimation of a model's parameters.  This can be a problem, as it can compromise the accuracy of a model's predictions.  


## How can they be addressed?

To address the challenges posed by outliers, there are two options:

1. Keep them
2. Remove them

This seems simple.  Unfortunately, it's often not.  

By keeping them we ensure that the scope of our model is not reduced.   Moreover, we can be certain that we haven't inadvertently excluded certain observations in order to make our data fit the model.  However, the accuracy of our model's predictions may be reduced.

By removing the offending data points, our model is likely to be more robust in its estimates.  Unfortunately it is also likely to have a reduced scope.  That is, be capable of providing robust predictions, but for a smaller set of scenarios.  

There is therefore often no clear cut answer.  Instead, the context of an outlier must be assessed, and a common sense solution applied.

## An example... that did not go to plan

Let's return to my house price project.  I wanted to explore the impact of excluding outliers, by estimating a multiple regression model before and after exclusion.  We'll walk through this together, and compare the results.  

### Estimating the more before excluding outliers

First, let's import our data:

```
kc_data = pd.read_csv("kc_house_data.csv")
```

For the purposes of this blog post, let's keep this simple.  Assume that all missing values have been addressed, transformations have been applied, and the file contains only the predictors that we're going to be using for this model.  If you would like more details on these steps, please refer to my [github repository](https://github.com/isobeldaley/dsc-v2-mod1-final-project-online-ds-ft-071519/tree/version2).

Next, we split the data into our dependent variable (y) and predictors (X).  The data is then split further into a training set (containing 80% of all observations) and a test set (20%).  The training set will be used to estimate our model, whilst the test set will be used to evaluate it.  

```
X = predictors
y = kc_data['price']

## Split the data into a training set (80%) and a test set (20%)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

We are now ready to estimate our model.  We will use the Statsmodule OLS function to complete this task.

```
import statsmodels.api as sm

X_train = sm.add_constant(X_train)

model = sm.OLS(y_train, X_train).fit()

model.summary()
```

![Regression Results](https://i.imgur.com/iMPZcMo.png?3)

As can be seen, the model has an R-Squared and adjusted R-Squared of 0.71.  Not bad.  How does our model perform when outliers are excluded? Let's see.  

### Estimating the more after excluding outliers

As above, the data is imported.  However, before splitting the data, all outliers are excluded.  To isolate outliers, I have chosen to apply the following often-used rule, and exclude data points which:

*Fall more than 1.5 times the interquartile range above the 3rd quartile, or below the 1st quartile*

This is achieved using a for-loop, which iterates through each variable, dropping any observation that exceeds the specified threshold:

```
for col in cols:
        Q1 = np.quantile(kc_data[col], 0.25)
        Q3 = np.quantile(kc_data[col], 0.75)
        IQR = Q3 - Q1
        drop_col = kc_data.loc[(kc_data[col] < Q1 - 1.5*IQR) | (kc_data[col] > Q3 + 1.5*IQR)].index
        kc_data.drop(drop_col, inplace=True) 
```

Once complete, the data can be split, and the model estimated as above.  The results are given below:

![](https://i.imgur.com/MPesV2J.png?1)

As can be seen, the R squared has fallen a fraction to 0.70.  This is not what we expected.  This suggests, there is little difference in the explanatory power of our model once outliers have been excluded.

So has our outlier exclusion been in vain? Let's investigate further.

### Comparing the Root Mean Squared Error

In theory, the model excluding outliers should generate more accurate predictions.  Luckily we have reserved 20% of our data in order to test this.  One way to do this, is to calculate the Root Mean Squared Error (RMSE) for each model.  

The RMSE computes the mean difference between the actual and predicted value of a target variable, and takes the square root.  This statistic can be computed for both the test and training set.  A big difference between the RMSE for the test set, and the RMSE for the training set gives us an indication that the model does not generalise beyond the training data set in generating accurate predictions.  We might expect this to be the case for the model that includes outliers.

In Python, SciKit learn provides a handy function to make calculation of the RMSE simple.  However, before we can do that, we must first generate some predictions for each model, using SciKit:

```
from sklearn.linear_model import LinearRegression

y_pred_test = regressor.predict(X_test)

y_pred_train = regressor.predict(X_train_no_const)
```

Now, RMSE for the test and training set can be caluculated, and more importantly the difference can be derived:

```
from sklearn import metrics
RMSE_train = np.sqrt(metrics.mean_squared_error(y_pred_train, y_train)))
RMSE_test = np.sqrt(metrics.mean_squared_error(y_pred_test, y_test)))
Difference = RMSE_train - RMSE_test

```

Running this code, we find that the absolute difference for the set including outliers was 0.0006.  Meanwhile, the  difference for the set excluding outliers was 0.001.  That is, almost double.

This is very interesting.  It suggests the model including outliers provided more accurate predictions.  This was not what we expected.

### Why did this happen?

Everything happens for a reason.  In this case, we suspect the reason may be an over-agressive exclusion of outliers.  In doing this, we restricted variation in our data, and this led to sub-optimal coefficient estimates.  

As a next step, we could apply a less agressive approach to outlier exclusion and analyse the results. 







