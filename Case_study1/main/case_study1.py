

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
# alphas = np.logspace(-6, 1, 50) # going from 10^6 to 10^1 with 50 samples in logspace
alphas = np.logspace(-10, 2.5, 20)

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
# L2 Ridge -- start here Monday to complete 
#############################################################
# Define the Ridge model
ridge_model = Ridge(max_iter=2000, random_state=1)

# Create the parameter grid for GridSearchCV
# alphas = np.logspace(-6, 1, 50)  # going from 10^6 to 10^1 with 50 
# alphas = np.logspace(-6, 2, 50)  # going from 10^(-6) to 10^2 with 50 
# alphas = np.logspace(-6, 12, 20)  # going from 10^(-6) to 10^(2.5) with 10 samples in logspace
alphas = np.logspace(1, 30, 20) 

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
# ElasticNet
#############################################################
# Define the Elastic Net model

from sklearn.linear_model import ElasticNet
# elastic_net_model = ElasticNet(max_iter=2000, random_state=1)
elastic_net_model = ElasticNet(max_iter=100, random_state=1)

# Create the parameter grid for GridSearchCV
# alphas = np.logspace(-6, 1, 10)  # going from 10^6 to 10^1 with 10 samples in logspace
# alphas = np.logspace(-2, 1, 10)  
alphas = np.logspace(-4, 2, 15) 
# l1_ratios = np.linspace(0, 1, 10) 
l1_ratios = np.linspace(0, 1, 20)# going from 0 to 1 with 10 samples in logspace
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
    ["Best l1_ratio", best_l1_ratio],
    ["Best Model MSE", best_score]
]
# Add top 5 features to the table
for _, row in top_5_features.iterrows():
    table_data.append([row['Feature'], row['Importance']])
headers = ["Metric/Best Alpha/Feature", "Value/Importance"]

print(tabulate(table_data, headers=headers, tablefmt="fancy_grid"))










