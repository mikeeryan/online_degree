#!/usr/bin/python
'''
Intro to Machine Learning - Final project

The original data is in a dictionary form which I transform to a dataframe 
for easier manipulation. Scikit algorithms uses NP arrays as inputs. 
At the end of this script I transform my dataframe back into a dictionary before pickling.
Actual pickling is done by the provided fns in the feature_format.py and tester.py scripts.
    
These files are to be shared with the instructor.
poi_id.py
my_dataset.pkl
my_feature_list.pkl
my_classifier.pkl
    (plus readme and short answers)

originals (not to be shared):
tester.py
enron61702insiderpay.pdf
poi_email_addresses.py
poi_names.txt


'''

#Set working directory
import os 
os.chdir(r"C:\Users\rf\Google Drive\Education\Python\codes\Udacity\Machine_Learning\codes")
print os.getcwd()

#everything else is not user specific
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn import linear_model 
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import StratifiedShuffleSplit

#for gridsearch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

#imports from Kate's scripts
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

#imports from the tester.py that does kfolding and scoring
from tester import dump_classifier_and_data
from tester import main

############################################################

# EDA - exploration of the data

# import data a dictionary by name
data_dict = pickle.load(open("C:/Users/rf/Google Drive/Education/Python/codes/Udacity/Machine_Learning/codes/tools/final_project_dataset.pkl", "r") )
#This pickled data set is a dictionary with 146 keys which are names of Enron employees. 
#The corresponding 146 elements are also dictionaries with 21 key:element pairs representing
#the poi label, financial and e-mail features. 

#First impression - why not turn this dataset into a dataframe with each employee getting own row
#and every feature getting own column?
#convert to dataframe (will convert back at the end)
sdf = pd.DataFrame.from_dict(data_dict, orient='index')
#OK, looks as I wanted. 

#At the first glance - lots of missing values. Let's explore this data frame properly.
print sdf.dtypes 
#All the integer columns are now strings
#object means character, 'NaN' here is a string, not placeholder
#Will need to convert back to numeric
sdf2 = sdf.convert_objects(convert_numeric=True)
print sdf2.dtypes 
#worked well but will be deprecated

#Now can look at summary stats

print 'The data frame has',sdf2.shape[1],'columns and',sdf2.shape[0],'rows.'
print sdf2.head()
print sdf2.describe()
print sdf2.mean(axis=0)
print sdf2.min(axis=0)
print sdf2.max(axis=0)

#also look at the character value - email_address
print sdf2.describe(include=['object'])
#35 missing emails though

#Remove the obvious outlier - TOTAL from the dataframe
sdf2['name']=sdf2.index
sdf3 = sdf2[(sdf2.name != 'TOTAL') & (sdf2.name != 'THE TRAVEL AGENCY IN THE PARK')]
#OK

#Let's look at some vars in depth
poi1 = sdf3.loc[:,['poi']]
poi2 = poi1.apply(pd.value_counts).fillna(0)
print "POI vs non-POI allocation."
print poi2

#comment out before submitting
#print sdf3['poi'].plot(kind='hist',bins=2,xticks=np.arange(0, 2, 1)
#    ,title='Frequency of POI',legend=True,color='grey')
#how to add counts to the graph?
#18 POI's and 127 nots
    
#Let's look at some of compensation metrics
#scatter plots - sns has jitter
#comment out before submitting
#sns.stripplot(x='poi', y='total_payments', data=sdf3, jitter=True, edgecolor='none', size=10, alpha=0.5)
#sns.despine()
#Who is that POI with the highest total payment?
maxtp = sdf3['total_payments'].max(axis=0)
print sdf3[sdf3['total_payments'] == maxtp][['name','total_payments']]
#Tada, gotcha, Ken. Such a major outlier would need to be removed if financial data is used in the models. 

#Who has the small amount?
mintp = sdf3['total_payments'].min(axis=0)
print sdf3[sdf3['total_payments'] == mintp][['name','total_payments']]
#was he a volunteer or something?

#Histogram of payments
#but gotta exclude Ken now
#commend out before submitting
#p99 = sdf3['total_payments'].describe(percentiles=[0.99])
#hist_tot = sdf3['total_payments'].hist(bins=50 , normed=True, color='red')
#dens_tot = sdf3['total_payments'].plot(kind='kde' , color='blue')
#print hist_tot.set_xlim((0,p99.ix[5]))
#Still skewed but more reasonable

#Let's look at this by POI flag
#comment out before submitting
#g = sns.FacetGrid(data=sdf3,  col="poi", margin_titles=True)
#tp_bins = np.arange(0, p99.ix[5],p99.ix[4] )
#g.map(plt.hist, 'total_payments', bins = tp_bins, color="steelblue")

#not so obvious
p99 = sdf3['total_payments'].describe(percentiles=[0.99])

#Let's look at all the POI's
print sdf3.ix[sdf3['poi']==True,['name','total_payments']].sort_values(by='total_payments', ascending=False).head()
#Hmm, the second highest earing POI made "only" 8mil while Ken made over 100mil

print sdf3.ix[sdf3['poi']==False,['name','total_payments']].sort_values(by='total_payments', ascending=False).head()
#While there are 3 guys who made over 10mil and were not POI's? That just does not make sense. 
#BHATNAGAR SANJAY left in summer of 2000 when the stock market price was very high 
#Troubles started in 2001
#so, it's not that straightforward

#########################################

#Understanding missing values
#Is data MCAR or not? what is the mix?
#export the data set for further examination
#sdf3.to_csv('sdf3.csv')

descriptive = sdf3.describe()
#descriptive.to_csv('sdf3_desc.csv')

nulls = pd.DataFrame(sdf3.isnull().sum())
nulls = nulls.sort_values(by=0, ascending=False)
print "Number of missing values by feature."
print nulls

#plus I already know what 35 emails are missing
print sdf2.describe(include=['object'])
#convert string 'NaN' to na? Leave it for now.

'''
Financial vars
A lot of the data is missing
It is missing in the only source we have "enron61702insiderpay.pdf"
"Other" is income as well - $
"Total payments" is the sum of 9 variables.
"Total stock option" is the sum of 3 variables. 

Total payments is missing for 21 people
Total stock value is missing for 20 people
There is only minor overlap among these people.
107 have both total payments and total stock
Notably, all the POI have both TP and TS numbers

Thinking these values are not missing compleately at random MCAR 
but actually are 0 so, I will impute with 0's to keep more observations. 

Email vars
34 out of 145 people do not have e-mail addresses and their emails are not available
Out of the remaining 111 people 25 do not have the email data.
So, effectively, we have have e-mail data for only 86 people. 
Noteably, 4 out of 18 POI do not have e-mail data, which is kind of bad.

Presuming everyone had an email account and used it. 
It's just the data was not released to public for some reason
Not imputing at all will result in losing these observations
Imputing with 0's will bias the models
Might be OK to impute with means. 

Bottom line: impute financial vars with 0's and e-mail vars with means. 
'''

#Creating new features etc
#Create a dummy var for poi so I have both boolean and numeric if needed
sdf3.loc[sdf3['poi'] == True,'poi_dummy'] = 1
sdf3.loc[sdf3['poi'] == False,'poi_dummy'] = 0

########################################

#load features into a list 
#remove the totals as they are simply sums of the other vars
finvars = [
'deferral_payments'
,'other'
,'salary'
,'exercised_stock_options'
,'bonus'
,'restricted_stock'
,'restricted_stock_deferred'
,'expenses'
,'loan_advances'
,'director_fees'
,'deferred_income'
,'long_term_incentive'
]

em_vars = [
'shared_receipt_with_poi'
,'from_messages'
,'from_this_person_to_poi'
,'from_poi_to_this_person'
,'to_messages'
]

other_vars = [
'email_address'
,'name'
,'poi'
,'poi_dummy'
]

#Data prepartion - removing outliers
#Keep the two CEO's - but exclude the person with missing values in all features
sdf4 = sdf3[(sdf3.name != 'LOCKHART EUGENE E')]

### Split into Training and Test data sets (not enough data for a separate Validation set)
train, test1 = train_test_split(sdf4, test_size = 0.3, random_state=2)
#Reviewer Suggestion: to use ShuffleSplit or StratifiedShuffleSplit here like done in the tester.py
#because performance scores produced by tester.py are more reliables because of the StratifiedShuffleSplit
#Response: Why reinvent the wheel? I can feed different specs into different models and test in tester.py at the end. 

#Reviewer Suggestion: keep the two CEO's in the test sample instead of leaving up to chance. 
#First remove them from both sets and then add to test - this way the step is independent of sampling
train = train[(train.name != 'LAY KENNETH L') & (train.name != 'SKILLING JEFFREY K')]
test2 = test1[(test1.name != 'LAY KENNETH L') & (test1.name != 'SKILLING JEFFREY K')]

#add back to the test set
t_ceos = sdf4[(sdf4.name == 'LAY KENNETH L') | (sdf4.name == 'SKILLING JEFFREY K')]
test = pd.concat([test2,t_ceos],axis=0)
#OK, now can continue

#returned DF's as well
#count of POI's in each?
train['poi'].sum() 
test['poi'].sum() 
#9 POI's in each test and training
#With such a small number of events, accuracy will be mostly about guessing the non-event correctly
#Expect high scores for any model really - even if predicts non-event for all cases

#split into separate df's, impute, then put back together
train_fin = train[finvars]
train_em  = train[em_vars]
train_ot  = train[other_vars]

test_fin = test[finvars]
test_em  = test[em_vars]
test_ot  = test[other_vars]

#Impute by 0's for financial data
train_fin['salary'].mean()
train_fin = train_fin.fillna(0)
train_fin['salary'].mean()

test_fin = test_fin.fillna(0)

#Impute by means for email data
train_em['from_messages'].mean()
train_em = train_em.fillna(train_em.mean())
train_em['from_messages'].mean()

test_em = test_em.fillna(train_em.mean())
#Now combine these tables back

train_out1 = train_em.combine_first(train_fin)
train_out2 = train_out1.combine_first(train_ot)
#column order gets jumbled though
test_out1 = test_em.combine_first(test_fin)
test_out2 = test_out1.combine_first(test_ot)

train_out2[train_out2['name'] == 'BADUM JAMES P'][['salary','to_messages']]
#OK, filled in by 0 or means as it should

#####################################

#Create additional email vars?
#Yes, add ratios of from/to pois of the emails
train_out2['ratio_from_poi'] = train_out2['from_poi_to_this_person'] / train_out2['from_messages']
train_out2['ratio_to_poi'] = train_out2['from_this_person_to_poi'] / train_out2['to_messages']

test_out2['ratio_from_poi'] = test_out2['from_poi_to_this_person'] / test_out2['from_messages']
test_out2['ratio_to_poi'] = test_out2['from_this_person_to_poi'] / test_out2['to_messages']
#nicely becomes 0, not null because I have already imputed before

new_vars = ['ratio_from_poi','ratio_to_poi']

#now stack the train and test sets back
frames = [train_out2, test_out2]
my_dataframe = pd.concat(frames)
print sum(my_dataframe['poi_dummy']) #16 poi left

# EDA section for selected vars
train_out3e = train_out2[em_vars]
test_out3e = test_out2[em_vars] 
 
#Seaborn's nice heatmap
#comment out before submitting
#corr_em = train_out3e.corr(method='spearman')
#print sns.heatmap(corr_em, 
#            xticklabels=corr_em.columns.values,
#            yticklabels=corr_em.columns.values)
#these are fairly strongly correlated, PCA should help to keep just the most useful variation.

####################################################

#Modeling NP arrays - separate for labels and features
#labels 
train_labels = np.array(train_out2['poi_dummy'])
test_labels = np.array(test_out2['poi_dummy'])

#keep all original vars first
keep_vars = list(finvars)
keep_vars_old = keep_vars + em_vars
#Just the original vars = 12+5 = 17
keep_vars_new = keep_vars + em_vars + new_vars
#including the newly created ones as well 12+5+2 = 19
#to understand usefullness the usefullness of the new features, need to build 
#decision trees with and without these variables and then compare the importances 

#Build a model with only the original features first
#Create the arrays for the models
#Put into DF's first, will be useful later
train_feat_df_old = train_out2[keep_vars_old]
test_feat_df_old = test_out2[keep_vars_old]

train_features_old = np.array(train_feat_df_old)
test_features_old = np.array(test_feat_df_old)
#OK, looks good, add names though? dataframes are better

#Scikit models - normally take NP arrays
#Score is accuracy

#Decision trees - min_samples_split default=2
clf_tree = tree.DecisionTreeClassifier(min_samples_split=4)
tree01_old = clf_tree.fit(train_features_old, train_labels) 
#print tree01_old.feature_importances_ 
#but they are unnamed now
#would need an extra step keep only the useful ones
imp_old = pd.DataFrame({'feature':train_feat_df_old.columns,'imp':np.round(tree01_old.feature_importances_,3)})
imp_old = imp_old.sort_values('imp',ascending=False)

#Now build a model that additionally includes the new variables
train_feat_df_new = train_out2[keep_vars_new]
test_feat_df_new = test_out2[keep_vars_new]
train_features_new = np.array(train_feat_df_new)
test_features_new = np.array(test_feat_df_new)
tree01_new = clf_tree.fit(train_features_new, train_labels) 
imp_new = pd.DataFrame({'feature':train_feat_df_new.columns,'imp':np.round(tree01_new.feature_importances_,3)})
imp_new = imp_new.sort_values('imp',ascending=False)

#Keep only the used variables - with non-0 importances
#Let's just build two models using these variables and assess the performance
imp_old = imp_old[imp_old['imp']>0]
dt_vars_old = list(imp_old['feature'])

#same for the second list
imp_new = imp_new[imp_new['imp']>0]
dt_vars_new = list(imp_new['feature'])

#Now rebuild the models with just these vars
#old features spec
train_features_df_old = train_out2[dt_vars_old]
train_features_dt_old = np.array(train_out2[dt_vars_old])
test_features_dt_old = np.array(test_out2[dt_vars_old])
tree02_old = clf_tree.fit(train_features_dt_old, train_labels) 
imp_tree02_old = pd.DataFrame({'feature':train_features_df_old.columns,'imp':np.round(tree02_old.feature_importances_,3)})
#One variable drops off again - so, need to re-create the var list for it
imp_tree02_old = imp_tree02_old[imp_tree02_old['imp']>0]
dt_vars_old = list(imp_tree02_old['feature'])
#and then re-create train and test arrays
train_features_dt_old = np.array(train_out2[dt_vars_old])
test_features_dt_old = np.array(test_out2[dt_vars_old])
#OK now can fit with these features only

#same way for the new spec
train_features_df_new = train_out2[dt_vars_new]
train_features_dt_new = np.array(train_out2[dt_vars_new])
test_features_dt_new = np.array(test_out2[dt_vars_new])
tree02_new = clf_tree.fit(train_features_dt_new, train_labels) 
imp_tree02_new = pd.DataFrame({'feature':train_features_df_new.columns,'imp':np.round(tree02_new.feature_importances_,3)})
#One variable drops off again - so, need to re-create the var list for it
imp_tree02_new = imp_tree02_new[imp_tree02_new['imp']>0]
dt_vars_new = list(imp_tree02_new['feature'])
#and then re-create train and test arrays
train_features_dt_new = np.array(train_out2[dt_vars_new])
test_features_dt_new = np.array(test_out2[dt_vars_new])
#OK now, fit with these features only


#Evaluation - I will create a score_stats fn that will output the confusion matrix and the stats
def score_stats (clf, xvar, yvar):
    from sklearn import metrics
    xhat = clf.predict(xvar)
    print "\n"
    print "Evaluation Metrics Report"
    print "# of cases:", yvar.shape[0]
    print "# of actual events:",sum(yvar)
    print "# of predicted events:",sum(xhat)
    print "# of events predicted correctly:",sum(yvar == xhat)
    print "Confusion matrix:"
    print metrics.confusion_matrix(yvar, xhat)
    print "Accuracy_score:", round(metrics.accuracy_score(yvar, xhat),4)
    print "Precision_score:", round(metrics.precision_score(yvar, xhat),4)
    print "Recall_score:", round(metrics.recall_score(yvar, xhat),4)
    print "F1_score:", round(metrics.f1_score(yvar, xhat),4)
    print "Roc_auc_score:", round(metrics.roc_auc_score(yvar, xhat),4)


# Logistic regression - reference
clf_logreg = linear_model.LogisticRegression()
log01 = clf_logreg.fit(train_features_dt_old, train_labels)
print "Logistic regression results"
score_stats(log01,test_features_dt_old,test_labels)
#0.2 and 0.11

print "Variable importances for the original variables"
print imp_tree02_old
tree02_old = clf_tree.fit(train_features_dt_old, train_labels) 
score_stats(tree02_old,test_features_dt_old,test_labels)
#0.33 and 0.22

print "Variable importances for the original and new variables"
print imp_new
tree02_new  = clf_tree.fit(train_features_dt_new, train_labels) 
score_stats(tree02_new,test_features_dt_new,test_labels)
#0.37 and 0.33 

####################################################

#Can the DT performance be improved?
#let's try to tune the DT - min_samples_split
clf_tree = tree.DecisionTreeClassifier(min_samples_split=10)
tree03 = clf_tree.fit(train_features_dt_new, train_labels) 
score_stats(tree03,test_features_dt_new,test_labels)
#min_samples_split=2 - 0.25 and 0.22
#Better min_samples_split=10 0.33 and 0.33 now
#min_samples_split=2 0.28 and 0.22
#min_samples_split=15 0 and 0
#OK, I'll take 0.33 and 0.33

#Tune max_depth
clf_tree = tree.DecisionTreeClassifier(max_depth=2)
tree04 = clf_tree.fit(train_features_dt_new, train_labels) 
score_stats(tree04,test_features_dt_new,test_labels)
#max_depth=2 - the same 0.6 and 0.33

#Can GridSearchCV help?
#pick the range of parameter values to use - cannot be everything! LoL
parameters = {'max_depth': np.arange(1, 20), 'min_samples_split' : np.arange(2, 20)}

#create the gridsearch object
sss = StratifiedShuffleSplit(train_labels)
grid = GridSearchCV(tree.DecisionTreeClassifier(random_state=2), parameters, scoring = "f1", cv=sss)

#fit and find the highest AUC
grid.fit(train_features_dt_new, train_labels)
grid_preds = grid.predict_proba(test_features_dt_new)[:, 1]
grid_performance = roc_auc_score(test_labels, grid_preds)

print "Highest AUC is:", grid_performance, "and the best parameters are:", grid.best_params_
#These parameters are the same that I found manually. 
#In fact, these results change from one iteration to another - not consistent for some reason. 

################################################

#Time to try ensemble models?
#These all looks alike - put into a fn for different sets of features
def all_clf (v_train_features, v_train_labels, v_test_features, v_test_labels, v_trees):
    clf_logreg = linear_model.LogisticRegression()
    log01 = clf_logreg.fit(v_train_features, v_train_labels)
    print "\n"
    print "LogisticRegression"
    score_stats(log01,v_test_features, v_test_labels)

    clf_tree = tree.DecisionTreeClassifier(min_samples_split=10, max_depth=2)
    tree01 = clf_tree.fit(v_train_features, v_train_labels) 
    print "\n"
    print "DecisionTreeClassifier"
    score_stats(tree01,v_test_features, v_test_labels)
    
    clfr = RandomForestClassifier(n_estimators=v_trees) #number of trees to build
    for01 = clfr.fit(v_train_features, v_train_labels)
    print "\n"
    print "RandomForestClassifier"
    score_stats(for01,v_test_features, v_test_labels)
    
    clfa = AdaBoostClassifier(n_estimators=v_trees)
    ada01 = clfa.fit(v_train_features, v_train_labels)
    print "\n"
    print "AdaBoostClassifier"
    score_stats(ada01,v_test_features,v_test_labels)
    
    clfg = GradientBoostingClassifier(n_estimators=v_trees)
    gbc01 = clfg.fit(v_train_features, v_train_labels)
    print "\n"
    print "GradientBoostingClassifier"
    score_stats(gbc01,v_test_features,v_test_labels)

#All original vars
all_clf(train_features_new, train_labels, test_features_new,  test_labels, 100)
#Just go by F1 score: logist 0.26, dtree 0.42, random 0, ada 0.36, gbc 0.4
#Winner is still the decision tree

#Just the most important ones from DT 
all_clf(train_features_dt_new, train_labels, test_features_dt_new,  test_labels, 100)
#Nice, almost exactly the same with the selected fewer vars - just 6 out of 19.  
#Dtree F1=0.42

##########################################

#This project would not be complete if I did not try to transform the original variables
#First I will standardize all the variables, then create principal components and model
#Caveat - both train and test data sets have to be standardized the means and std devs from the train set. 
#Explanation - long story short - think of the test set as new data for which we don't know the mean/stddev - e.g. just 1 obs to score
#Also, let's create separate PC for financial and email vars

#Keep all the 19 vars, but split into separate sets for PCA
train_out3f =train_out2[finvars]
test_out3f =test_out2[finvars]

train_out3e = train_out2[em_vars]
test_out3e = test_out2[em_vars] 

#Create a fn so that I can keep all PC's or just a few top ones
def pca_fn (v_components):
    sklearn_pca = sklearnPCA(n_components=v_components)
    #Fit the scaler first, so that it can be applied to the test set as well
    train_out3f_stdf = StandardScaler().fit(train_out3f)
    train_out3f_std = train_out3f_stdf.transform(train_out3f)
    train_out3f_pcf = sklearn_pca.fit(train_out3f_std)
    print "\n"
    print "Financial var PC explained variance ratio"
    print train_out3f_pcf.explained_variance_ratio_
    train_out3f_pc = train_out3f_pcf.transform(train_out3f_std) #transform
    
    #Transform but not fit the test set as well
    test_out3f_std = train_out3f_stdf.transform(test_out3f)
    test_out3f_pc = train_out3f_pcf.transform(test_out3f_std) #transform
    
    #Same scaling and PCA for email vars
    train_out3e_stdf = StandardScaler().fit(train_out3e)
    train_out3e_std = train_out3e_stdf.transform(train_out3e)
    train_out3e_pcf = sklearn_pca.fit(train_out3e_std)
    print "\n"
    print "Email var PC explained variance ratio"
    print train_out3e_pcf.explained_variance_ratio_
    train_out3e_pc = train_out3e_pcf.transform(train_out3e_std) #transform
    
    #Repeat for test
    test_out3e_std = train_out3e_stdf.transform(test_out3e)
    test_out3e_pc = train_out3e_pcf.transform(test_out3e_std) #transform

    #Concatenate together
    train_features = np.concatenate([train_out3f_pc, train_out3e_pc], axis=1) 
    test_features = np.concatenate([test_out3f_pc, test_out3e_pc], axis=1) 

    return train_features, test_features


train_features_pc, test_features_pc = pca_fn(None)
#Looks like keep the top 2 PC's from each set would be OK
all_clf(train_features_pc, train_labels, test_features_pc,  test_labels, 100)
#WoW, with all PC logistic did worse, DT (0.2222)

#With just 2 PCs?
train_features_pc, test_features_pc = pca_fn(2)
all_clf(train_features_pc, train_labels, test_features_pc,  test_labels, 100)
#All did worse, GBC won. 

#############################

#Conclusion:
#Using the original data - logistic Regression and Gradient Boosting models did equally well
#Precision = 0.4 and Recall = 0.4.
#Keeping just 7 most important variables (using Decision Tree's diagnostics) did not noticeably 
#reduce performance. 

#Standardizing the data and creating principal components helped some of the classifiers
#but not enough to beat the F1=0.4 of the Logistic regression on original data. 
#Now let's see how it will look in the tester.py

##########################################

#Final Output to pickle
#Convert the final modeling data frame to the required dictionary
my_dataset = my_dataframe.to_dict(orient='index')
#remember: keep_vars has all 19, dt_vars_new/old has just top ones
#but these may be too many vars, try a custom list instead
#remove the least important features one by one

custom_vars = [
'deferred_income'
,'bonus'
]
features_list = ['poi'] + custom_vars #put labels first
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
#labels should be 0/1, no?

#Dump and Score
#name the final classifier - clf
#DT - using the same parameters I used above
clf_tree = tree.DecisionTreeClassifier(min_samples_split=2, max_depth=2)
clf = clf_tree.fit(features, labels)
dump_classifier_and_data(clf, my_dataset, features_list)
main()
#OK - above .3 for both!

#My report:
custom_var_list = np.array(test_out2[custom_vars])
score_stats(clf,custom_var_list,test_labels)
#OK - ready for write-up


#Archive - other non-winner algorithms, optional
'''
clf_logreg = linear_model.LogisticRegression()
clf = clf_logreg.fit(features, labels)
dump_classifier_and_data(clf, my_dataset, features_list)
main()

clfr = RandomForestClassifier(n_estimators=100)
clf = clfr.fit(features, labels)
dump_classifier_and_data(clf, my_dataset, features_list)
main()

clfa = AdaBoostClassifier(n_estimators=100)
clf = clfa.fit(features, labels)
dump_classifier_and_data(clf, my_dataset, features_list)
main()

clfg = GradientBoostingClassifier(n_estimators=100)
clf = clfg.fit(features, labels) 
dump_classifier_and_data(clf, my_dataset, features_list)
main()

'''

#logistic F1 0.1762
#DT F1 0.24326
#Random forest F1 0.0888 - and was very slow
#ADA F1 0.2495 (adaboost uses decision trees as weak learners)
#GBC F1 0.2057
#Conclusion: Logistic's performance declined considerably from F1=0.4 to F1=0.17
#Which means that the train/test split was still not enough to protect from 
#overfitting/optimism. 
#Decision tree built on the 7 most important variables was stable:
#On the test sample DT=0.2857, in Kfolding done by tester.py - F1=0.24326. 

#Verdict: just decision tree on the original data. 
#Accuracy: 0.86093       Precision: 0.45648      Recall: 0.22550 F1: 0.30187     F2: 0.25089

#Ran w/o any errors

#The End.
