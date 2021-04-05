#!/usr/bin/env python
# coding: utf-8

# In[89]:


from sklearn import *
# import nltk
# nltk.download('punkt')
import numpy as np
import pandas as pd
import sklearn as sk 
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler, Normalizer, QuantileTransformer, PowerTransformer
import sklearn.model_selection
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_score
import time


#1. USING PANDAS TO EXTRACT COLUMN DATA:
#***************************************************************************
df = pd.read_csv("Train.csv")

#array of all the summaries

og_summaries = df['summary']
summaries = df["summary"]

#array of the overalls
overall = df["overall"]
print(overall)

#array of product nums
products = df["amazon-id"]

og_amazon_reviews = df['reviewText']
amazon_reviews = df['reviewText']


# In[90]:


# 2. CREATING SENTIMENT MODEL:
#***************************************************************************
#****2A. first creating sentiment model based on the amazon reviews

# Using ratings associated with reviews as indicators of sentiment
# i.e. turn any review with star rating >= 5 to 1, else 0 
review_sentiments = [int(overall[i] >= 5) for i in range(len(overall))]


for i in range(len(amazon_reviews)):
    if type(amazon_reviews[i])!=str:
        amazon_reviews.pop(i)
        review_sentiments.pop(i)
print(set([type(r) for r in amazon_reviews]))


#create train test split
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(amazon_reviews, 
                                                                       review_sentiments, 
                                                                       random_state=1, test_size=0.33)
#vectorize data 
vect1 = TfidfVectorizer()
x_train = vect1.fit_transform(x_train)
x_test = vect1.transform(x_test)

# We can experiment with different numbers of max iterations
# 100 iters is default
# iters = [1000]

start = time.time()
review_clf = LogisticRegression(max_iter=500)
review_clf.fit(x_train, y_train)
end = time.time()
print("Test accuracy: {}".format(100*review_clf.score(x_test, y_test)))
print("Time taken: {:.2f} seconds".format(end-start))

#****2B. first creating sentiment model based on the summary reviews
summary_sentiments = [int(overall[i] >= 5) for i in range(len(overall))]
for i in range(len(summaries)):
    if type(summaries[i])!=str:
        summaries.pop(i)
        summary_sentiments.pop(i)
print(set([type(s) for s in summaries]))

#create train test split
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(summaries, 
                                                                       summary_sentiments, 
                                                                       random_state=1, test_size=0.33)

# vectorize summary
vect2 = TfidfVectorizer()
x_train = vect2.fit_transform(x_train)
x_test = vect2.transform(x_test)

# We can experiment with different numbers of max iterations
# 100 iters is default
# iters = [1000]

start = time.time()
summary_clf = LogisticRegression(max_iter=500)
summary_clf.fit(x_train, y_train)
end = time.time()
print("Test accuracy: {}".format(100*summary_clf.score(x_test, y_test)))
print("Time taken: {:.2f} seconds".format(end-start))


# In[91]:


#3. GENERATE AVERAGE PRODUCT REVIEW AND SUMMARY SENTIMENT:
#***************************************************************************
df = pd.read_csv("Train.csv")

# using models generated in (2) to determine average product review and summary sentiment

summary_data = vect2.transform(df['summary'].values.astype('U'))
review_data = vect1.transform(df['reviewText'].values.astype('U'))

#predicting sentiment
summary_predictions = summary_clf.predict(summary_data)
review_predictions = review_clf.predict(review_data)

#accuracy analysis
def getAccuracy(texts, preds):
    total = len(texts)
    t = 0
    for i in range(total):
        if (overall[i] == 5 and preds[i] == 1) or (overall[i] != 5 and preds[i] == 0):
            t+=1
    return t/total

#determine sentiment predictions
print("Accuracy of summary sentiment classifier on Amazon summaries: {:.2f}".format(getAccuracy(summaries, summary_predictions)))
print("Accuracy of review sentiment classifier on Amazon reviews: {:.2f}".format(getAccuracy(amazon_reviews, review_predictions)))


# In[92]:


# 4. WE'LL NOW TRAIN SEVERAL CLASSIFIERS USING THE FOLLOWING FEATURES 
#***************************************************************************
#   1.Using just avg. summary sentiment as a feature
#   2. Using just avg. review sentiment as a feature
#   3. Using avg. summary sentiment & avg. review sentiment
#   4. Using avg. summary sentiment & avg. review sentiment & sales rank
#   5. Using avg. summary sentiment & avg. review sentiment & salesrank & price

#generate average review and summary sentiments
targets = {}
for i in range(len(products)):
    if products[i] not in targets:
        targets[products[i]] = [overall[i]]
    else:
        targets[products[i]].append(overall[i])

targets = [int(np.average(targets[prod]) > 4.5) for prod in targets]

review_predictions, summary_predictions = list(review_predictions), list(summary_predictions)

amz_review_sentiments, amz_summary_sentiments = {}, {}
for i in range(len(products)):
    if products[i] not in amz_review_sentiments:
        amz_review_sentiments[products[i]] = [review_predictions[i]]
        amz_summary_sentiments[products[i]] = [summary_predictions[i]]
    else:
        amz_review_sentiments[products[i]].append(review_predictions[i])
        amz_summary_sentiments[products[i]].append(summary_predictions[i])


avg_review_sentiments = [np.average(amz_review_sentiments[prod]) for prod in amz_review_sentiments]
avg_summary_sentiments = [np.average(amz_summary_sentiments[prod]) for prod in amz_summary_sentiments]


avg_review_sentiments = [[score] for score in avg_review_sentiments]
avg_summary_sentiments = [[score] for score in avg_summary_sentiments]


# In[93]:


#PARADIGM 1: Just summary sentiments
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(avg_summary_sentiments, targets, random_state=1)


#train model
summary_only_clf = LogisticRegression()
summary_only_clf.fit(X_train, Y_train)
print("Test accuracy using summary sentiment only: {:.2f}".format(100*summary_only_clf.score(X_test, Y_test)))


# In[6]:


#PARADIGM 2: Just review sentiments
X_train, X_test, Y_train, Y_test = sk.model_selection.train_test_split(avg_review_sentiments, targets, random_state=1)


#train model
review_only_clf = LogisticRegression()
review_only_clf.fit(X_train, Y_train)
print("Test accuracy using review sentiment only: {:.2f}".format(100*review_only_clf.score(X_test, Y_test)))


# In[7]:


#PARADIGM 3: Just review & summary 

#create new feature matrix contraining avg. review and summary sentiment
X = np.zeros((len(avg_review_sentiments), 2))

for i in range(X.shape[0]):
    X[i] = (avg_review_sentiments[i][0], avg_summary_sentiments[i][0])

kf = KFold(n_splits=10, shuffle=True, random_state=1)

targets = [[t] for t in targets]

targets = np.asarray(targets)

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]

for train_idx, test_idx in kf.split(targets):
    y_train, y_test = targets[train_idx], targets[test_idx]

#train model
review_and_summary_clf = LogisticRegression(C=0.6, class_weight={0:1.417, 1:1})
review_and_summary_clf.fit(X_train, y_train)


print("Test accuracy using review sentiment and summary sentiment: {:.2f}".format(100*review_and_summary_clf.score(X_test, y_test)))

print("Weighted F1 using review sentiment and summary sentiment: {:.2f}".format(100*f1_score(y_test, 
             review_and_summary_clf.predict(X_test), 
             average='weighted')))


# In[95]:


# Cross Validation 
cv_results = cross_val_score(review_and_summary_clf, X, 
                            targets,
                            scoring='f1_weighted',
                            cv=10)
print("For LogReg model on review and summary: ", cv_results)
print("LogReg average : ", np.average(cv_results))
print("Test accuracy using review sentiment and summary sentiment: {:.2f}".format(100*review_and_summary_clf.score(X_test, y_test)))
print("Weighted F1 using review sentiment and summary sentiment: {:.2f}".format(100*f1_score(y_test, 
             review_and_summary_clf.predict(X_test), 
             average='weighted')))


# In[96]:


# running grid search CV to find the best hyperparameters for LogReg
from sklearn.datasets import make_blobs
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
# define dataset
# define models and parameters
model = LogisticRegression(class_weight={0: 1.417, 1: 1})
solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.6, 0.3, 0.01]
# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='precision',error_score=0)
grid_result = grid_search.fit(X_train, y_train)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))


# In[97]:


#train SVM model
review_and_summary_clfsvm = svm.SVC(class_weight={0:1.417, 1:1}, kernel = 'linear')
review_and_summary_clfsvm.fit(X_train, y_train)

# Cross Validation 
cv_results = cross_val_score(review_and_summary_clfsvm, X, 
                            targets,
                            scoring='f1_weighted',
                            cv=10)
print("For SVM model on review and summary: ", cv_results)
print("SVM average : ", np.average(cv_results))
print("Test accuracy using review sentiment and summary sentiment: {:.2f}".format(100*review_and_summary_clfsvm.score(X_test, y_test)))
print("Weighted F1 using review sentiment and summary sentiment: {:.2f}".format(100*f1_score(y_test, 
             review_and_summary_clfsvm.predict(X_test), 
             average='weighted')))


# In[99]:


param_grid = {'C': [0.1, 1, 10],  
              'gamma': [1, 0.1, 0.01], 
              'kernel': ['linear']}  
  
grid = GridSearchCV(SVC(), param_grid, refit = True, verbose = 3) 
  
# fitting the model for grid search 
grid.fit(X_train, y_train)


# In[100]:


print(grid.best_params_) 


# In[101]:


#XGBOOST 
from xgboost import XGBClassifier
review_summary_clf_XG = XGBClassifier() 
review_summary_clf_XG.fit(X_train, y_train)
print("Test accuracy using review sentiment, summary sentiment: {:.2f}".format(100*review_summary_clf_XG.score(X_test, y_test)))
print("Weighted F1 using review sentiment, summary sentiment: {:.2f}".format(100*f1_score(y_test, 
             review_summary_clf_XG.predict(X_test), 
             average='weighted')))
print("--------------------------------------------------")
# KNN
classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, y_train)
print("Test accuracy using review sentiment, summary sentiment: {:.2f}".format(100*classifier.score(X_test, y_test)))
print("Weighted F1 using review sentiment, summary sentiment: {:.2f}".format(100*f1_score(y_test, 
             classifier.predict(X_test), 
             average='weighted')))
print("--------------------------------------------------")
# decision tree
classifier = DecisionTreeClassifier(class_weight ={0: 1.417, 1: 1})
classifier.fit(X_train, y_train)
print("Test accuracy using review sentiment, summary sentiment: {:.2f}".format(100*review_summary_clf_XG.score(X_test, y_test)))
print("Weighted F1 using review sentiment, summary sentiment {:.2f}".format(100*f1_score(y_test, 
             classifier.predict(X_test), 
             average='weighted')))


# In[102]:


# Trying AdaBoost with XGBoost base classifier
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50, base_estimator=review_summary_clf_XG,learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)
print("Test accuracy using review sentiment, summary sentiment: {:.2f}".format(100*abc.score(X_test, y_test)))
print("Weighted F1 using review sentiment, summary sentiment {:.2f}".format(100*f1_score(y_test, 
             abc.predict(X_test), 
             average='weighted')))


# In[117]:


#5. TESTING MODEL 
#*****************************************************



#test data 
test = pd.read_csv("Test.csv")
products = df["amazon-id"]
#use sentiment model to get summary & review test data 
summary_data = vect2.transform(test['summary'].values.astype('U'))
review_data = vect1.transform(test['reviewText'].values.astype('U'))

#generate predictions
test_summary_preds = summary_clf.predict(summary_data)
test_review_preds = review_clf.predict(review_data)

test['sum_sentiment'] = test_summary_preds
test['rev_sentiment'] = test_review_preds
        
sum_sentiment_by_prod = test.groupby('amazon-id', sort=False).mean()['sum_sentiment']
rev_sentiment_by_prod = test.groupby('amazon-id', sort=False).mean()['rev_sentiment']

X_test = np.zeros((len(sum_sentiment_by_prod), 2))
for i in range(X_test.shape[0]):
    X_test[i] = (list(rev_sentiment_by_prod)[i], list(sum_sentiment_by_prod)[i])
    
# train classifier on whole training data 
classifier_lr = LogisticRegression(class_weight={0: 1.417, 1: 1}, C=100, penalty='l2', solver='newton-cg')


# In[118]:


classifier_lr.fit(X, targets)
test_preds = classifier_lr.predict(X_test)
test_preds.shape


# In[121]:


test_products = []
for t in test['amazon-id']:
    if t not in test_products:
        test_products.append(t)
pd.DataFrame({'amazon-id': test_products, 'prediction': test_preds}).to_csv('predictions_del4.csv')


# In[74]:


#PARADIGM 4: Just review & summary  & sales rank 
sales_ranks = {}
for i in range(len(products)):
    if products[i] not in sales_ranks:
        sales_ranks[products[i]] = [df['salesRank'][i]]
    else:
        sales_ranks[products[i]].append(df['salesRank'][i])

avg_sales_ranks = [[np.average(sales_ranks[prod])] for prod in sales_ranks]

scaler = StandardScaler() # 73.6 weighted F1
# scaler = MinMaxScaler() # 73.6 weighted F1
# scaler = MaxAbsScaler() # 73.6 weighted F1
# scaler = RobustScaler() # 73.6 weighted F1
# scaler = Normalizer() # 73.48 weighted F1
# scaler = QuantileTransformer() # 73.48 weighted F1
# scaler = PowerTransformer() # 73.48 weighted F1

scaler.fit(avg_sales_ranks)
avg_sales_ranks = scaler.transform(avg_sales_ranks)

X = np.zeros((len(avg_review_sentiments), 3))

for i in range(X.shape[0]):
    X[i] = (avg_review_sentiments[i][0], avg_summary_sentiments[i][0], avg_sales_ranks[i][0])

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]

for train_idx, test_idx in kf.split(targets):
    y_train, y_test = targets[train_idx], targets[test_idx]

review_summary_sales_clf = LogisticRegression(C=0.1, class_weight={0:1.417, 1:1}, 
                                             tol=2)
review_summary_sales_clf.fit(X_train, y_train)

print("Test accuracy using review sentiment, summary sentiment, and sales rank: {:.2f}".format(100*review_summary_sales_clf.score(X_test, y_test)))

print("Weighted F1 using review sentiment, summary sentiment, and sales rank: {:.2f}".format(100*f1_score(y_test, 
             review_summary_sales_clf.predict(X_test), 
             average='weighted')))

# precision and recall 
from sklearn.metrics import precision_score
precision = precision_score(y_test, review_summary_sales_clf.predict(X_test), average='binary')
print("Precision:",precision)
from sklearn.metrics import recall_score
recall = recall_score(y_test, review_summary_sales_clf.predict(X_test), average='binary')
print("Recall:", recall)


# In[75]:


#PARADIGM 5: Just review & summary  & sales rank & prices

#construct our price feature
prices = {}
for i in range(len(products)):
    if products[i] not in prices:
        prices[products[i]] = [df['price'][i]]
    else:
        prices[products[i]].append(df['price'][i])

avg_prices = [[np.average(prices[prod])] for prod in prices]

scaler.fit(avg_prices)
avg_prices = scaler.transform(avg_prices)

X = np.zeros((len(avg_review_sentiments), 4))
for i in range(X.shape[0]):
    X[i] = (avg_review_sentiments[i][0], avg_summary_sentiments[i][0], avg_sales_ranks[i][0], 
           avg_prices[i][0])

for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]

for train_idx, test_idx in kf.split(targets):
    y_train, y_test = targets[train_idx], targets[test_idx]
    
# LogReg
review_summary_sales_price_clf = LogisticRegression(C=100, class_weight={0:1.417, 1:1}, solver='newton-cg')
review_summary_sales_price_clf.fit(X_train, y_train)
print("Weighted F1 using review sentiment, summary sentiment, sales rank, and price: {:.2f}".format(100*f1_score(y_test, 
             review_summary_sales_price_clf.predict(X_test), 
             average='weighted')))


# In[76]:


review_summary_sales_price_clfsvm = svm.SVC(class_weight={0:1.417, 1:1}, kernel = 'linear')
review_summary_sales_price_clfsvm.fit(X_train, y_train)
print("Weighted F1 using review sentiment, summary sentiment, sales rank, and price: {:.2f}".format(100*f1_score(y_test, 
             review_summary_sales_price_clfsvm.predict(X_test), 
             average='weighted')))

