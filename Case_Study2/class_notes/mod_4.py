
# missing data patterns
import pandas as pd 
import numpy as np 


raw_data = pd.csv('file_name.csv') #srb_housing.csv
# russian housing dataset, quite large, dont have it anywhere

raw_data.head()
# we see that there are missing values, imputation on some level 

raw_data.describe()
# Summary Statistics 

# 1st step = visually exam to find patterns in missing data. Professor uses Excel to look at the data to understand the missing data, looking to see if the missing data has a pattern -- determined to be missing not at random - we start to see in 2013 the data is starting to be populated but starting to be unpopulated. not missing at random , timestamp was used as third column, an example missing not at random on time stamp and product investment name 

# loop through the data to match shape, if they are not the same then we know the data is missing 

missing_data = raw_data.describe()

missing_data

# we have counts and column names 

missing_data.columns
for i in missing_data.columns:
    if missing_data.loc['count', i] != raw_data.shape[0]
        print(i)
        
# we now have a list of all the columns with missing data. we have our work cut out for us on this data imputation. 

# Missing Data Patterns Video: Imputation Demo 
raw_data.loc[:,['full_sq','life_sq']]
# relationship between the two 

raw_data.corr()
# a heat map would work but it would be too large to show in this scenario. 

raw_data.head() # while they are not roughly equal they are roughly correlated. 
# my preference is not to drop

# so we can impute

raw_data.loc[raw_data['life_sq'].isna(), ['full_sq','life_sq']]

raw_data.loc[raw_data['life_sq'].isna(), 'full_sq']
# how to substitute things in... 
raw_data.loc[raw_data['life_sq'].isna(), 'life_sq']
# returns a series
# assign these values to the next column over 
raw_data.loc[raw_data['life_sq'].isna(), 'life_sq'] = raw_data.loc[raw_data['life_sq'].isna(), 'full_sq']

#this overwrites so  you have to be careful 
# raw_data.loc[raw_data['life_sq'].isna()]

imputed_data= raw_data
raw_data = pd.read.csv('filname.csv')
# getting back orgical data 
raw_data.loc[raw_data['life_sq'].isna()] 

imputed_data.loc[imputed_data['life_sq'].isna()] # simple substitution 

raw_data['full_sq']/2 # we can add subtract divide etc, to impute just remember to save into a modified dataframe to essentially save your work. 
raw_data.loc[raw_data['life_sq']]



####################################################
# Clustering Video Demo Term Frequency
####################################################
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer 


newsgroup_train = fetch_20newsgroups(subset = 'train', 
                                     remove=('headers','fitters', 'quotes')) 
# we want to deal with the full text and this removes those, someone has already done this for us and this is how we access it. 


newsgroup_test = fetch_20newsgroups(subset = 'test', 
                                     remove=('headers','fitters', 'quotes'))

# look for an occurance of a word and count those words as a count vectorizer and add it as a feature. 

newsgroup_train.data[0]
# count vectorizer only starts with two letters 

vectorizer = CountVectorizer() # works out of the box

# we want to fit this to our newspaper data 
vectorizer.fit(newsgroup_train.data) # this will create a vocabulary for us!! # try in pcn project

vectorizer.vocabulary_ 
# indices of the words 
# note however you wont always get the same resutls, i have seen insntaces where this can change so be aware of that in this case they ramined the same. 

# transform only row 1
sample = vectorizer.transform([newsgroup_train.data[0]]).toarray() # returns in a sparse matrix form, however we wnat to inspect visually and thats hard to do with a sparse matrix format, so casting it as array allows us to look at the matrix in a traditional form instead of sparse. 
sample.shape #(1,101631)
# sparse matrix meaning we are using our entire vocabulary will have many zeros because its sparse... we know words that occured like "was" which was word 95834... single column row vector with over 100k elements in it, we see that mostly they are zeros when we print out 
sample
# not every word occurs in every example, thus sparse matrix, 


# is a 2d array but one of the dimenssions is 1. take one example and look at the number of occurance of the word was 
sample[0][95834] # returns 4 
#with  but we are just going to look at one example at that index 

# the countvectorizer is extremely efficient, but we could also do this manually. 



#####################
# Live Session
#####################
import pandas as pd 
import numpy as np 


df = pd.read_csv('/Users/tmc/Desktop/MS_SMU_Admin/05_2024Summer/QUANTIFIYING_TW/04_module/train.csv') 
df
df.columns
df.isnull()
df.describe()
df.info()


pd.set_option('display.max_rows', None)   # Display all rows
pd.set_option('display.max_columns', None)  
df.iloc[0]



# Group 1, max_floor, 



# G1 Max Floor, hospital_beds_raion,school_quota
df['G1 Max Floor, hospital_beds_raion,school_quota'].iloc[0]

df.loc[0, 'max_floor, hospital_beds_raion, school_quota'] 


columns = ['G1_Max_Floor, hospital_beds_raion, school_quota'] 
df.loc[columns]

df.loc[df['max_floor'].isna()] # simple 
df.loc[df['hospital_beds_raion'].isna()] # s
df.loc[df['school_quota'].isna()][0]


df.loc[df['max_floor'].isna(), 'max_floor'] = df.loc[df['max_floor'].isna(), 'floor']
df.loc[df[['max_floor','floor']]]
df.loc[df['floor']]
df.loc[df['max_floor']]

slearn.impute.SimpleImputer

# solution to imputing with mode from live session exercise
max_floor_mode = imputed_data['max_floor'].mode()[0]
imputed_data['max_floor'].fillna(max_floor_mode, inplace=True)