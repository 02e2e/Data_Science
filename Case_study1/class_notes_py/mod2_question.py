# Module 2 Question: 
# Using Python and the sklearn LinearRegression, Lasso, and Ridge methods, find the best fit (Lowest Means Squared Error) for each of the methods using the California Housing Data Set. To get the data, you can use the following import item:

#Approximate Answers:
# Linear Regression = 0.558
# Lasso Regression = 0.557, alpha = 0.0026
# Ridge = 0.553, alpha = 10
# Be prepared to discuss and bring questions to the live session.


# Code 
from sklearn.datasets import fetch_california_housing

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
# Load Data
cal_housing = fetch_california_housing()# have to turn off vpn to download the data 
data_dict = fetch_california_housing()
# turn into a pandas dataframe 
X = pd.DataFrame(cal_housing.data, columns=cal_housing.feature_names) 
# data = pd.DataFrame(data_dict['data'], columns = data_dict['feature_names'])
y = cal_housing.target 

# print y 
y 

# linear regression - vanilla basic
vanilla_my_model = LinearRegression()

# print it out 
vanilla_my_model.fit(X,y)
vanilla_my_model.coef_ # part of the utils to pull the coefficients of the model to assess feature importance 

# need to scale the data 
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(data=X_scaled, columns=cal_housing.feature_names)

scaled_model = LinearRegression()
scaled_model.fit(X_scaled, y)

scaled_model.coef_
# compare to the original vanilla model coeff 
vanilla_my_model.coef_

# tune lamda from slides / its alpha in the sklearn package 
for i in range(len(X_scaled.columns)): 
    print(X_scaled.columns[i], scaled_model.coef_[i])


#### lasso / l1 model ###
l1_model = Lasso(alpha=1)
l1_model.fit(X_scaled, y)
# the alpha is to high everything was driven to zero 
l1_model.alpha=0.1
l1_model.alpha

alpha = 1
for i in range(5):
    l1_model.alpha = alpha
    l1_model.fit(X_scaled,y)
    print(alpha, l1_model.coef_)
    print("---------")
    alpha = alpha/10 

### Cross Validation  ### 

cross_val_score(scaled_model, X_scaled, y) 
alpha = 1
for i in range(5):
    l1_model.alpha = alpha
    print(alpha, cross_val_score(l1_model, X_scaled, y))
    print("---------")
    alpha = alpha/10 


### Ridge #### 
l2_model = Ridge(alpha=1)
l2_model.fit(X_scaled, y)
l2_model.score(X_scaled, y)

l2_model.coef_

# again with a different alpah 
l2_model = Ridge(alpha=1E-5)
l2_model.fit(X_scaled, y)
l2_model.coef_
l2_model.score(X_scaled, y)

# loop to find the right alpha 
alpha = 1E-5
for i in range(10): 
    l2_model.alpha = alpha
    print(alpha, cross_val_score(l2_model, X_scaled, y))
    print('--------')
    alpha = alpha * 10 
    


################

# Live Session Module 2 
# Code 
from sklearn.datasets import fetch_california_housing

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

from sklearn.model_selection import cross_val_score
# Load Data
# cal_housing = fetch_california_housing()# have to turn off vpn to download the data 
data_dict = fetch_california_housing()
# turn into a pandas dataframe 

data = pd.DataFrame(data_dict['data'], columns = data_dict['feature_names'])

from sklearn.preprocessing import StandardScaler
scale = StandardScaler()

# argument scale on training data 
# some dont care
data_scaled = pd.DataFrame(scale.fit_transform(data), columns = data_dict['feature_names'])

import matplotlib.pyplot as plt 
plt.hist(data_scaled['Longitude'], bins=40)
plt.show()

y=data_dict['target'] # YOU DONT SCALE THE TARGET YOU TRY AND STANDARDIZE THAT 
plt.hist(y, bins=50)
# nice poisson curve, and then all of sudden large bar at 5, maybe split the data, and then do the fit on all those less than 5, and these are going to cause all sorts of problems. 

# always look at the targets EDA this is what he is looking for, you dont have to add labels in code you can do that in the report 

# now we need to get one of our models in 

from sklearn.linear_model import Ridge
# got to the sklearn documentation go the Ridge model and you can see L2 regularlization, they called lamdab regularzation strength alpha, 
# their slopes are w we called it m 
# alpha times the coefficient squared minimizes the object function 
# solver on auto, dont play with tol, just play with teh alpah the regualrization streignth, vary alpha while we are doing cross validation 
# we are going to take our L2 mdoel 
l2_model = Ridge(alpha=5)
# what we really want to is vary alpha but before we do that we want to cross validate
# theres an easy way to do CV -- cross_val score gives avg score, cross_Val predict gets you every value of the dataset. 
cross_val_score(l2_model, data_scaled, y, cv=5, scoring='neg_mean_squared_error') # number of folds is cv - 5 to 10 is fine adn then select our scoring
# youll notice our scoring spread widely
# in sklearn with all the metrics we have, some we wnat to minimze some we want to maximize, what they did the ones you want ot minimizae they make them negative, so by picking the least negative value because losses are always positive, so the largest loss is the smallest number in neg mean sq error
# the idea is its automaticallly going to pcik the largest value 
import numpy as np 
alphas = np.logspace(-6,6,12)
alpha 
# we are going to print out the scores and see which one is the best 
# we still need to fix the shuffling as well - the wide spread in values earlier was the clue 

    
from sklearn.model_selection import KFold
splitter = KFold(n_splits=5, shuffle=True, random_state=1)
for sample in alphas: 
    l2_model.alpha = sample
    mse = cross_val_score(l2_model, data_scaled, y, cv=splitter, scoring='neg_mean_squared_error')
    print(mse.mean(), mse.std(), sample)

from sklearn.linear_model import RidgeCV
l2_model_2 = RidgeCV(alphas=alphas, scoring='neg_mean_squared_error', cv=splitter)
l2_model_2.fit(data_scaled, y)
#what was our best alpha, the pattern with sklearn is that anything that comes out with an underscore, is part of the fit, so coeff_ intercept_, alpaha_, best_score_ 
l2_model_2.alpha_


l2_model.fit(data_scaled,y)
l2_model.alpha # was set before and not found # underfit range 
l2_model.coef_ 


# now do an l1 
#### lasso / l1 model ###
l1_model = Lasso(alpha=1)
splitter = KFold(n_splits=5, shuffle=True, random_state=1)
for sample in alphas: 
    l1_model.alpha = sample
    mse = cross_val_score(l1_model, data_scaled, y, cv=splitter, scoring='neg_mean_squared_error')
    l1_model.fit(data_scaled,y)
    print(mse.mean(), mse.std(), sample)
    print(l1_model.coef_)
    print('-------------')

# you can see the importance, before the weakest variable is zeroed out because we dont want to lose features... this tells you feature importance 
# regularization is an order of magnitude otherwise any difference is miniscual 
# one of the big mistakes si taht peopel will not put hyperparamters far enoguh apart, we ran 12 hyper paramters, times 5 so thats 60 models in a few mins... 1 10 and 100 is not smart choices -- so look at out he he casted the orders of magnitude on the alpha 
# learning rates and lambdas can skew by orders of magnitude 
# the last thing is the elastic net which combines both of them. 

# it ends up very close to the l2, both types of regularlziation l1 and l2, with strength and mixing hyperparameters will add a bit of time. 
# the l2 lambda originally was 43 and the l1 was an order of magnitude smaller, so elastic nets are not often used. most of the time we use l2, but l1 and l2 operate at different scales, and eleastic net will most likely not be the better solution than l1 or l2.. 
# top 5 variable improtances and two models l1 and l2. 
# go ahead and tune elastic net has two paramters to tune but wont give you the best scores. 
# elastic net -- l1_ratio 0.5 - if you set to 1 its the samething as l1, if you set it to zero its set to ridge regression, thats what l1_ratio does 

# this is 650 fits right now 
from sklearn.linear_model import ElasticNet
# DO NOT USE THE SGDREGRESSOR
mix = np.linspace(0,1,11) # ten increments call that our mix 
lambdas = np.logspace(-6,6,12) # minus 6 to 6 and do 12 
best_rho = -1
best_lambda = -1 # made negative because then its a sign that did somethign wrong in my coding 
best_score = 1E6 
for rho in mix: 
    for lambda_ in lambdas: 
        model= ElasticNet(alpha=lambda_, l1_ratio=rho)
        tmp = cross_val_score(model, data_scaled, y, cv=splitter)
        best_score = best_score
        if tmp.mean() < best_score: 
            best_rho = rho 
            best_lambda = lambda_

best_rho
best_score
tmp.mean()
best_lambda


# your prediction should not be less than zero, you cannot have a negative temp for a solid, all of these are solids in your case study in your prediction 
# there is somthing called grid search CV this is where the scorign problem comes in, we want the best loss this is where its odd teh smalles is the largest value of the scoring function, least amount negative 










l1_model.fit(X_scaled, y)
# the alpha is to high everything was driven to zero 
l1_model.alpha=0.1
l1_model.alpha

alpha = 1
for i in range(5):
    l1_model.alpha = alpha
    l1_model.fit(X_scaled,y)
    print(alpha, l1_model.coef_)
    print("---------")
    alpha = alpha/10 
















alpha = 1
for i in range(5):
    l1_model.alpha = alpha
    print(alpha, cross_val_score(l1_model, X_scaled, y))
    print("---------")
    alpha = alpha/10 
