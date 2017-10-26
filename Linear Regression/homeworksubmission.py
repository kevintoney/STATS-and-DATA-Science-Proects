
#In height_weight files, height is the explanatory variable.
#weight is the response variable. 

#the question is how does height affect weight?
#file1 has the height and weights from people
#file 2 has the same heights with different weights. Did 
#the two datasets get recorded at different times?

##############include an intercept in file 1. 

#####File 1
import pandas as pd
import numpy as np
from pandas import Series, DataFrame
from matplotlib import pylab as plt
import statsmodels.formula.api as smf
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression

### Load the dataset into dataframe, ensure read-in correctly
weightData = pd.read_csv("C:/Users/kevin/Desktop/Fall 2017/STAT 420/Homework/Linear Regression assignment/height_weight1.csv")
weightData.head()

weightData.dtypes

weightData.columns

### There is no missing data. 

weightData.dtypes

### exploratory data analysis
### obtain summary statistics
### assess regression assumptions

weightData.describe()
#plot the data. The data seems to follow a positive relationship. 
plt.plot(weightData.weight, weightData.height, '.')
plt.show()

# check normality of response variable
plt.hist(weightData.weight, 50)
plt.show()
#the response variable looks somewhat normal. 

### split data into training, test
weightTrain, weightTest = train_test_split(weightData, test_size=.2, random_state=123)
#the test data is 0.2 times the training data. 
#there are 100 observations in the test data. 
weightTrain.shape
weightTest.shape

### basic linear regression (without variable selection)

# adding '1' as a predictor forces the inclusion of an intercept
model = smf.ols(formula='weight ~ height + 1', data=weightTrain).fit()
model.summary()


### Test our residuals

residual = model.predict(weightTrain) - weightTrain.weight

# Test for normal residuals
plt.hist(residual)
# Test for heteroscedasticity
plt.plot(model.predict(weightTrain), residual, '.')
plt.show()
#the residuals look independent because there is no pattern to the data.
#also, the variance looks equal throughout the data points. 

#Also, it is good to know the R2 score is 0.36. Therefore, 
#only 36% of the data's variation is explained by our linear model. This score isn't optimal.


###Now, try removing a constant.
# subtracting '1' as a predictor forces the exclusion of an intercept
model = smf.ols(formula='weight ~ height - 1', data=weightTrain).fit()
model.summary()


### Test our residuals

residual = model.predict(weightTrain) - weightTrain.weight

# Test for normal residuals
plt.hist(residual)
#the residuals follow a normal distribution. 
# Test for heteroscedasticity
plt.plot(model.predict(weightTrain), residual, '.')
plt.show()
#the residuals look independent because there is no pattern to the data.
#also, the variance looks equal throughout the data points. 

#when we excluded the intercept, the R2 score was 0.992. 
#92 percent of our data's variation is explained by our linear model in 
#ordinary least squares regression. R2 is a good score to use in order to compare
#the two linear models. 



#############Question 2
####Data File 2
#############
weightData2 = pd.read_csv("C:/Users/kevin/Desktop/Fall 2017/STAT 420/Homework/Linear Regression assignment/height_weight2.csv")
weightData2.head()

weightData2.dtypes

weightData2.columns

### There is no missing data. 

### exploratory data analysis
### obtain summary statistics
### assess regression assumptions

weightData2.describe()
#plot the data. The data seems to follow a positive relationship. 
plt.plot(weightData2.weight, weightData2.height, '.')
plt.show()

# check normality of response variable
plt.hist(weightData2.weight, 50)
plt.show()
#the response variable looks somewhat normal, but the data is skewed to the right.  

### split data into training, test
weight2Train, weight2Test = train_test_split(weightData2, test_size=.2, random_state=123)
#the test data is 0.2 times the training data. 
#there are 100 observations in the test data. 
weight2Train.shape
weight2Test.shape

### basic linear regression (without variable selection)

# adding '1' as a predictor forces the inclusion of an intercept
model = smf.ols(formula='weight ~ height -1', data=weight2Train).fit()
model.summary()
#the R2 score is 0.991.

### Test our residuals

residual2 = model.predict(weight2Train) - weight2Train.weight

# Test for normal residuals
plt.hist(residual2)
#the residuals are normal, but skewed to the right. 
# Test for heteroscedasticity
plt.plot(model.predict(weight2Train), residual2, '.')
plt.show()
#the residuals look independent because there is no pattern to the data.
#also, the variance looks equal throughout the data points. 

#Also, it is good to know the R2 score is 0.36. Therefore, 
#only 36% of the data's variation is explained by our linear model. This score isn't optimal.

####One of our residual assumptions are not fully met. 
####The residuals are not normal and centered around 0. Therefore, our
####confidence intervals for our predictions will be wrong if we try to predict future values. 
####Since the confidence intervals will be incorrect, the 
####prediction intervals will also be incorrect. Each point 
####estimate will be incorrect. 




############### 
###Problem 3
###############

#We are looking at the cars csv data file. 
#The main research goal is to predict the price of a used car based on multiple factors. 

#the price of a used car is the response variable.
#explanatory variables are the age of car, the make, the type
#and the number of miles.
carsData = pd.read_csv("C:/Users/kevin/Desktop/Fall 2017/STAT 420/Homework/Linear Regression assignment/car.csv")
carsData.head()

carsData.dtypes

carsData.columns 

### exploratory data analysis
### obtain summary statistics
### assess regression assumptions

carsData.describe()
### There is no missing data.

#plot the data. The data seems to follow a positive relationship. 
plt.plot(carsData.Price, carsData.Miles, '.')
plt.show()
#none of the relationships have a linear relationship with the response.
#this is a multivariate regression problem. 

# check normality of response variable
plt.hist(carsData.Price, 50)
plt.show()
#the response variable doesn't look normal. 

### split data into training, test
carsTrain, carsTest = train_test_split(carsData, test_size=.1, random_state=123)
#the test data is 0.2 times the training data. 
#there are 100 observations in the test data. 
carsTrain.shape
carsTest.shape

### basic linear regression (without variable selection)

# adding '1' as a predictor forces the inclusion of an intercept
modelcars1 = smf.ols(formula='Price ~ Miles', data=carsTrain).fit()
modelcars1.summary()

modelcars2 = smf.ols(formula='Price ~ Miles + Age', data=carsTrain).fit()
modelcars2.summary()

modelcars3 = smf.ols(formula='Price ~ Miles + Age + C(Make)', data=carsTrain).fit()
modelcars3.summary()
#the third model, which accounts for all the explanatory variables,
#is the best model for our question. 
##This model is the best because the R2 score is the highest,
##also, the R2 score continued to improve as I added more variables
#into the ols model. 

### Test our residuals

residual = modelcars3.predict(carsTrain) - carsTrain.Price

# Test for normal residuals
plt.hist(residual)
# Test for heteroscedasticity
plt.plot(modelcars3.predict(carsTrain), residual, '.')
plt.show() 

#there is no homoskedascity or independence in this data. 
#the residuals aren't normally distributed eighter. 
#ordinary least squares regression is not appropriate for this data. 

dict1 = {'Age':7, 'Make': 'BMW', 'Type':2, 'Miles':67000, 'Price':2000}
carsTest = carsTest.append(dict1, ignore_index=True)
#in order to predict the vauoe of one car, try appending it 
#to the list you are working with. 
modelcars3.predict(carsTest[240:])
#Even though predicting a 7 year old BMW's value if it has 67,000 miles, isn't a good idea,
#the best model predicted the car would sell for $26,452 and 12 cents. 


