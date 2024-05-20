

# Your case study is to build a linear regression model using L1 or L2 regularization (or both) the task to predict the Critical Temperature as closely as possible. In addition, include in your write-up which variable carries the most importance.

# To begin, we first load our imports necessary to run our models, normalize the data, perform cross validation, and a grid search for the regularization strength parameter alpha (aka C). 

import pandas as pd 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from tabulate import tabulate
from sklearn.metrics import mean_squared_error, accuracy_score

import pandas as pd

pd.set_option('display.max_rows', None)   # Display all rows
pd.set_option('display.max_columns', None)  # Display all columns

# The data was downloaded from the SMU ___ website and then the file paths for both files are a assigned a variable (filepath and filepath2).

filepath = "/Users/tmc/Desktop/MS_SMU_Admin/05_2024Summer/QUANTIFIYING_TW/02_module/Case_study1/train.csv" # one dot current directory, two dots means the parent directory one level up from the current directory 
filepath2 = "/Users/tmc/Desktop/MS_SMU_Admin/05_2024Summer/QUANTIFIYING_TW/02_module/Case_study1/unique_m.csv"

# Now that the the files are assigned a variable name, the data is then converted to a pandas dataframe using the pd.read_csv function. 

# excel 1
traincsv = pd.read_csv(filepath)
traincsv.info()
traincsv.dtypes
traincsv.shape
traincsv.describe()
# excel 2 
uniquemcsv = pd.read_csv(filepath2)
uniquemcsv.info()
uniquemcsv.dtypes
uniquemcsv.shape

# Once the data was loaded, the columns for each file were identified. The data is clean and there are no missing values so no imputation is needed. To verify there are no missing values we use the isnull() function.  Missing values can create errors when passing the data training a model. 
    
traincsv.columns 
uniquemcsv.columns
if traincsv.isnull().values.any() or uniquemcsv.isnull().values.any()  :
    print("There are missing values in the data.")
else:
    print("The data is clean and there are no missing values.")

# The columns 'critical_temp' and 'material' are dropped from the second dataset (uniquecsv) as the 'critical_temp' column is already located in the first the dataset (traincsv). The 'material' column is a composite of all the other features in the data so it would be redundant to include as a feature in the models. 
df2_unique = uniquemcsv.drop(columns=['critical_temp', 'material'])  

# The columns function was then used to verify that the appropriate columns were in fact dropped from the dataframe. 
df2_unique.columns

# The client has asked that both datasets be joined so that we have one joined dataset to train and evaluate our models on. The pandas pd.concat function is called to join the data frames. The columns and head functions is then used to once again verify the data was joined appropriately. The shape function was utlized to check the number of rows and columns in the dataframe (21263, 168). 

joined_df = pd.concat([traincsv, df2_unique], axis=1)
print(joined_df)
# pd.set_option('display.max_columns', None)
print(joined_df.head(10))
print(joined_df.shape)
print(joined_df.columns)

#############
#look at your data
joined_df.describe()
joined_df.info()
joined_df.dtypes
joined_df.shape
# look at correlation 
###############

# The joined dataframe (joined_df) is now ready for use, however we still had the target column in the dataframe, so the target variable was created and the target column 'critical_temp' isolated and dropped using the drop function. Note for the target variable we use double brackets on the line target = joined_df[['critical_temp']] so that the target is a pandas dataframe rather than pandas series, which allows us to use the columns function to print the column name 'critical_temp'. 

target = joined_df[['critical_temp']] # our target variable
joined_features = joined_df.drop(columns=['critical_temp']) 
print(joined_features.columns)
print(target.columns)

# The target and the features (joined_features) are now in their in own respective dataframes. Before moving forward, the data is then visualized to view the distribution (normal, skewed, bimodal, central tendancy, spread, outliers) and extract insights into the feature relationships. 

# The histogram function was called with 50 bins and the histogram for the target was generated. The histogram indicates that the target 'critical_temp' is right skewed with a majority of the temperatures clustered at the lower end with a long tail. The takeaway here is that that the critical temperature for most of the materials are usually on the lower end with a only a few higher critical temperature values. The temperatures above 90 __ on right end of the tail of the distribution could possibly impact the performance of the regression models. However, we will proceed with no transformation on the target. 

plt.hist(target, bins=50)
plt.xlabel("Critical Temperature")
plt.ylabel("Frequency")
plt.title("Distribution of Critical Temperature")
plt.show()

# We have 167 columns and visualizing them via a histogram individually is not the not the most efficient way to gain insights into the data. Instead the describe function is utilized which provides summary statistics for each feature in the dataframe. The output is transposed to provide easier viewing but it still did not look quite right for a report, so after some investigating a package called tabulate was utilized to create a table that was more appropriate for the report and the number of decimal points were reduced to two. 


# plt.hist(joined_features, bins=10)
# plt.show()
# Calculate correlations for your DataFrame
# lets review the correlation structure of the variables to the target
correlations = joined_df.corr()

# Filter for correlations with the target variable above 0.8
target_correlations = correlations['critical_temp']
target_correlations = target_correlations[target_correlations.abs() > 0.5]
top_correlations = target_correlations.sort_values(ascending=False)[:20]

# Format the result for better display (with only two columns)
top_correlations_df = pd.DataFrame(top_correlations).reset_index()
top_correlations_df.columns = ['Feature', 'Correlation']

# Print the table with top correlations
print(tabulate(top_correlations_df, headers='keys', tablefmt='psql', floatfmt=".2f"))


joined_features.describe().T

summary_stats = joined_features.describe().T.applymap(lambda x: f"{x:.2f}")  # Format data
print(tabulate(summary_stats, headers='keys', tablefmt='psql', floatfmt=".2f"))  

summary_stats_top10 = summary_stats.iloc[:10] 
print(tabulate(summary_stats_top10, headers='keys', tablefmt='psql', floatfmt=".2f"))  

# The resulting table illustrates that the features need to be normalized as we have a wide range between min and max values as well as high standard deviations. To address the wide spread in values and variation in averages for the features the standard scaler package was utilized so that the models will not give undue importance to features with larger values. The scaling will transform the features into a comparable range with a mean of zero and a standard deviation of one. By scaling the data our models should theoretically perform better as the models assumes that the features are centered around zero and hae a similar scale.  

scale = StandardScaler()
X_scaled = pd.DataFrame(scale.fit_transform(joined_features))
# plt.hist(X_scaled, bins=10)
# plt.show()
summary_stats_scaled = X_scaled.describe().T.applymap(lambda x: f"{x:.2f}")  # Format data
print(tabulate(summary_stats_scaled, headers='keys', tablefmt='psql', floatfmt=".2f"))  


summary_stats_scaled_top10 = summary_stats_scaled.iloc[:10] 
print(tabulate(summary_stats_scaled_top10, headers='keys', tablefmt='psql', floatfmt=".2f"))  




# In creating a linear model using lasso (l1 regularization), the alpha hyperparameter is the most critical parameter to explore. The alpha parameter controls the regularlization strength with higher values reducing the least important coefficients to zero, hence feature selection. The max-iter is another parameter that can be used to optimize the algorithm and was adjusted so that model convereged. 
# To perform a grid search on the regularlization strength, the GridSearchCV class was utilized with the scoring metric negative mean squared error to minimize the MSE. The score generated is then converted back to MSE by taking the absolute value to make it more interpratble. 

#############################################################
# 01 Lasso 
#############################################################
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.linear_model import Lasso
alphas = np.logspace(-6, 1, 50) # going from 10^6 to 10^1 with 50 samples in logspace
# Define the Lasso model
# l1_model = Lasso(max_iter=2000) 
l1_model = Lasso(alpha=1, max_iter=2000, random_state=1) 

# Create the parameter grid for GridSearchCV
param_grid = {'alpha': alphas}
grid_search = GridSearchCV(l1_model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
grid_search.fit(X_scaled, target)

# Get the best alpha and its corresponding model
best_alpha = grid_search.best_params_['alpha']
best_model = grid_search.best_estimator_
best_score = abs(grid_search.best_score_)

# Determine top 5 features
feature_importances = pd.DataFrame({'Feature': joined_features.columns, 'Importance': np.abs(best_model.coef_)})
top_5_features = feature_importances.nlargest(5, 'Importance') 


results_data = [
    ["Best Alpha", best_alpha],
    ["Best Model Coefficients", best_model.coef_],
    ["Best Model MSE", best_score],
]
headers = ["Metric: abs(MSE)", "Score"]
print(tabulate(results_data, headers=headers, tablefmt="fancy_grid"))

table_data = top_5_features.values.tolist() 
headers = top_5_features.columns.tolist()  # Get column names as headers

print(f"Best alpha: {best_alpha:.6f}")
print(f"Best model MSE: {best_score:.4f}")
print("\nTop 5 Features:")
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))  

table_data = [
    ["Best Alpha", best_alpha],
    ["Best Model MSE", best_score]
]
# Add top 5 features to the table
for _, row in top_5_features.iterrows():
    table_data.append([row['Feature'], row['Importance']])
headers = ["Metric/Best Alpha/Feature", "Value/Importance"]

print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))
#############################################################
# ANALYZE RESULTS
# The cross validated score represents the average performance of the best model with the best alpha (regularlization strength parameter) across 5 folds. 
#############################################################

#############################################################
# L2 Ridge -- start here Monday to complete 
#############################################################
# Define the Ridge model
ridge_model = Ridge(max_iter=2000, random_state=1)

# Create the parameter grid for GridSearchCV
# alphas = np.logspace(-6, 1, 50)  # going from 10^6 to 10^1 with 50 
# alphas = np.logspace(-6, 2, 50)  # going from 10^(-6) to 10^2 with 50 
alphas = np.logspace(-6, 2.5, 50)  # going from 10^(-6) to 10^(2.5) with 50 samples in logspace


param_grid = {'alpha': alphas}

# Create the GridSearchCV object
grid_search = GridSearchCV(ridge_model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_scaled, target)

# Get the best alpha and its corresponding model
best_alpha_ridge = grid_search.best_params_['alpha']
best_model_ridge = grid_search.best_estimator_
best_score_ridge = abs(grid_search.best_score_)
########

# Determine top 5 features
feature_importances_ridge = pd.DataFrame({'Feature': joined_features.columns, 'Importance': np.abs(best_model_ridge.coef_.ravel())})
top_5_features_ridge = feature_importances_ridge.nlargest(5, 'Importance') 


results_data_ridge = [
    ["Best Alpha", best_alpha_ridge],
    ["Best Model Coefficients", best_model_ridge.coef_],
    ["Best Model MSE", best_score_ridge],
]
headers = ["Metric: abs(MSE)", "Score"]
print(tabulate(results_data_ridge, headers=headers, tablefmt="fancy_grid"))

table_data_ridge = top_5_features_ridge.values.tolist() 
headers= top_5_features_ridge.columns.tolist()  # Get column names as headers

print(f"Best alpha: {best_alpha_ridge:.6f}")
print(f"Best model MSE: {best_score_ridge:.4f}")
print("\nTop 5 Features:")
print(tabulate(table_data_ridge, headers=headers, tablefmt="fancy_grid"))  

table_data_ridge = [
    ["Best Alpha", best_alpha_ridge],
    ["Best Model MSE", best_score_ridge]
]
# Add top 5 features to the table
for _, row in top_5_features_ridge.iterrows():
    table_data_ridge.append([row['Feature'], row['Importance']])
headers = ["Metric/Best Alpha/Feature", "Value/Importance"]

print(tabulate(table_data_ridge, headers=headers, tablefmt="fancy_grid"))

#############################################################
# ANALYZE RESULTS
# 
#############################################################

#############################################################
# ElasticNet
#############################################################
# Define the Elastic Net model
elastic_net_model = ElasticNet(max_iter=2000, random_state=1)

# Create the parameter grid for GridSearchCV
alphas = np.logspace(-6, 1, 50)  # going from 10^6 to 10^1 with 50 samples in logspace
l1_ratios = np.linspace(0, 1, 10)  # going from 0 to 1 with 10 samples in logspace
param_grid = {'alpha': alphas, 'l1_ratio': l1_ratios}

# Create the GridSearchCV object
grid_search = GridSearchCV(elastic_net_model, param_grid, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)

# Perform the grid search
grid_search.fit(X_scaled, target)

# Get the best alpha and its corresponding model
best_alpha = grid_search.best_params_['alpha']
best_l1_ratio = grid_search.best_params_['l1_ratio']
best_model = grid_search.best_estimator_
best_score = abs(grid_search.best_score_)

########

results_data = [
    ["Best Alpha", best_alpha],
    ["Best Model Coefficients", best_model.coef_],
    ["Best Model MSE", best_score],
]
headers = ["Metric: abs(MSE)", "Score"]
print(tabulate(results_data, headers=headers, tablefmt="fancy_grid"))

table_data = top_5_features.values.tolist() 
headers = top_5_features.columns.tolist()  # Get column names as headers

print(f"Best alpha: {best_alpha:.6f}")
print(f"Best model MSE: {best_score:.4f}")
print("\nTop 5 Features:")
print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))  

table_data = [
    ["Best Alpha", best_alpha],
    ["Best Model MSE", best_score]
]
# Add top 5 features to the table
for _, row in top_5_features.iterrows():
    table_data.append([row['Feature'], row['Importance']])
headers = ["Metric/Best Alpha/Feature", "Value/Importance"]

print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))

#############################################################
# ANALYZE RESULTS
# 
#############################################################













# we are here on Friday, we have two versions and we wanted to see that they matched to be sure that we are doing it correclty, if so, then now its just a matter of doing L2 and ElasticNet next, be sure to make sure you cover enough ground on your parameter search, and research on how we might use a plot or something to display our results. 
# in looking at both results from lasso 01 and lasso 02 everything is the same but the MSE is slighlty different, i suppose you could use both methods in your Case Study... and note the difference, but overall its the same alpha, and the same top 5 features, invesetigate why the the MSE is slightly different between the two tomorrow. 











##############
# NOTES BELOW 
##############
# original # 02 lasso 
import numpy as np 

alphas = np.logspace(-4, 1, 50)

from sklearn.model_selection import KFold

results = []
# Manual Grid Search with Cross-Validation
for sample in alphas:
    l1_model.alpha = sample
    mse_scores = cross_val_score(l1_model, X_scaled, target, cv=splitter, scoring='neg_mean_squared_error')
    l1_model.fit(X_scaled, target)
    mean_mse = -mse_scores.mean()  # Convert to positive MSE
    std_mse = mse_scores.std()

    # Store results for later analysis
    results.append((sample, mean_mse, std_mse, l1_model.coef_.copy())) 
    print(mean_mse, std_mse, sample)  # Optional: Print intermediate results

# Find the best alpha and MSE
best_result = min(results, key=lambda x: x[1])  # Find minimum mean MSE
best_alpha, best_mse, _, best_coef = best_result

# Determine top 5 features using absolute coefficients
feature_importances = pd.DataFrame({'Feature': joined_features.columns, 'Importance': np.abs(best_coef)})
top_5_features = feature_importances.nlargest(5, 'Importance')

# Display results
print("\nBest Alpha:", best_alpha)
print("Best MSE:", best_mse)
print("\nTop 5 Features:")
for _, row in top_5_features.iterrows():
    print(f"- {row['Feature']}: {row['Importance']:.4f}")
##############







# visualize the output of coefficients 
# The coefficients
print("Coefficients: \n", regr.coef_)
# The mean squared error
print("Mean squared error: %.2f" % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# The coefficient of determination: 1 is perfect prediction
print("Coefficient of determination: %.2f" % r2_score(diabetes_y_test, diabetes_y_pred))

# Plot outputs
plt.scatter(diabetes_X_test, diabetes_y_test, color="black")
plt.plot(diabetes_X_test, diabetes_y_pred, color="blue", linewidth=3)

plt.xticks(())
plt.yticks(())

plt.show()







# l2 

# elastic_net


# 5 l1 regularization 

# 6 l2 regularization 

# 7 predict the Critical Temp as closely as possible
# 8 which variable carries the most importance 
# 9 you'll need nice charts and tables - refer to your ml2 notebook for help on that 
# Summarize results in a table at the end with all the scores, findings etc. all the work you did




##### below are notes from mod1 and mod2 #####

# Module 2 Question: 
# Using Python and the sklearn LinearRegression, Lasso, and Ridge methods, find the best fit (Lowest Means Squared Error) for each of the methods using the California Housing Data Set. To get the data, you can use the following import item:

#Approximate Answers:
# Linear Regression = 0.558
# Lasso Regression = 0.557, alpha = 0.0026
# Ridge = 0.553, alpha = 10
# Be prepared to discuss and bring questions to the live session.







##################################
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
