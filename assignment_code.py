#hm regression q2,q4(except classification did it ona regression problem)
#predict song popularity
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error as mse
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pdb

df = pd.read_csv('song_data.csv')

#print(df.head())
#print(df.info()) # no  missing values
#print(df.nunique())  # audio_mode is dummy, time signature has 5 categories(0-4), 0-11 for key

#print(df.describe())
#print(df.shape)

tr,tq = train_test_split(df,test_size= 0.2, random_state=122)
tr,te = train_test_split(tr,test_size=0.2,random_state =122)
#print(tr.head())
#print(te.head())
#print(tr.shape, te.shape)

tr['song_popularity'].plot.kde()  # distribution  - more with medium level popularity score than low and high, hwoever bimodal (one mode at 0)
#plt.show()

sns.heatmap(tr.corr(), annot=True)#correlated vars:energy and loudness(0.76),energy and accousticness(-0.66)
 
#plt.show()
#pdb.set_trace()
cat = ['audio_mode', 'time_signature', 'key','song_name']
c = tr.drop(cat, axis=1)
#print(c.columns)
#print(c.head())

                           #with pairplot dancebility and key looks kind of non linearly correlated

for i in tr[cat]:
 sns.boxplot(x=i,y ='song_popularity', data = tr)  #time signature =0 associated with low popularity
 #plt.show()

#distribution of categorical variables in test and train - are they same?(should also test for numerical variable distributions)
for i in range(len(cat[1:])):
 print('for', cat[i])
 fig,(ax1,ax2) = plt.subplots(nrows = 1,ncols= 2)
 tr[cat[i]].value_counts().plot(ax=ax1,kind = 'bar')
 te[cat[i]].value_counts().plot(ax=ax2, kind = 'bar')
 #plt.show()                                                     #time signature 0 is absent in test, validation set
 
#dummy encode key and time signature
 
rd = pd.get_dummies(df,columns = ['time_signature', 'key'])

#print(rd.head())
y = rd['song_popularity'] 
x = rd.drop(['song_name', 'song_popularity'], axis =1)


#UNDERFITTING 
# as none of the features show good correlation with Y=song_popularity - the dataset doesn't show linearity in relationship. Hence
# linear regression, simple SVM- which are meant for linear dataset,are likely to underfit

#linear_regression

#standardising data
z = StandardScaler()
z.fit(x.to_numpy())
x_lr = z.transform(x.to_numpy())
#print(x_lr)
y_lr = y.to_numpy()

lr = LinearRegression()
mod = lr.fit(x_lr, y_lr)
print(mod.score(x_lr,y_lr), 'is R^2')
print(mod.coef_, 'is coefficients',mod.intercept_, 'is intercept')
x_tr_pred = lr.predict(x_lr)
print(mse(y_lr,x_tr_pred),'is train MSE')

#preping validation set
te_new = pd.get_dummies(te,columns = ['time_signature', 'key'])
y_te = te_new['song_popularity'] 
x_te = te_new.drop(['song_name', 'song_popularity'], axis =1)
x_te['time_signature_0']= np.zeros((x_te.shape[0],1))
#print(x_te.shape, x.shape, te_new.shape)
#print(x_te.columns, x.columns)
#pdb.set_trace()
x_te_z = z.transform(x_te.to_numpy())
y_te_new = y_te.to_numpy()


pred_y = mod.predict(x_te_z)
print(mse(y_te_new,pred_y), 'is validation MSE')

#R^2 = 0.0483 -> the model is performing very poorly
#456.6560 is train MSE
#6.880e+29 is test MSE --> the train and test MSE are close but both are very high-->there is underfitting

#to address underfitting --> 1)change the model/algorithm (have a high variance, low bias model/ increase model complexity) 2)get more training data

#Let's change the algorithm to random forest regressor 


# randomforest regressor without much hyperparameter tuning
print('***************RF 1**************') 
rf_reg = RandomForestRegressor(n_estimators = 1000,max_features='sqrt',random_state=12 )
rf_reg.fit(x_lr,y_lr)
#print(score(x_lr,y_lr), 'is R^2')
y_pred_rf = rf_reg.predict(x_lr)
print(mse(y_lr, y_pred_rf), 'is train MSE')

pred_y_te = rf_reg.predict(x_te_z)
print(mse(y_te_new,pred_y_te),'is validation MSE')

#41.169 is train MSE
#202.224 is test MSE
# we see train and test MSE have hugely declined than in Linear regression case. However,
#train MSE is much greater than test MSE --> which suggests that the model is overfitting

#to prevent overfitting we opt for 1) regularisation 2) removing some features(multicollinear ones) 3)introducing more data

#1) regularised randomforest 
print('********RF 2***********')
#grid search to find good hyperparameter values


# Using Grid Search for Hyper paramter tuning
#train_d = train_data.drop('Survived',axis=1)
#train_target_d = train_data['Survived']
rf2_reg = RandomForestRegressor(n_estimators = 1000,max_depth = 17,max_features='auto',random_state=12 )
rf2_reg.fit(x_lr,y_lr)
#print(score(x_lr,y_lr), 'is R^2')
y_pred_rf = rf2_reg.predict(x_lr)
print(mse(y_lr, y_pred_rf), 'is train MSE')

pred_y_te = rf2_reg.predict(x_te_z)
print(mse(y_te_new,pred_y_te),'is validation MSE')
plt.barh(x.columns, rf_reg.feature_importances_)
#plt.show()

#Different models tried out:
#n_estimators = 1000,max_depth = 17,max_features='auto'
#92.33689067520052 is train MSE
#138.3675095434459 is validation MSE
#there is not only decrease in test MSE from model trained above but also the gap bw the two has declined.
#This is a better model than the previous one. It can be further optimised. 

#n_estimators = 1000,max_depth = 17,max_features='sqrt',random_state=12
#130.50289170102366 is train MSE
#261.15689934008674 is test MSE


#n_estimators = 1000,max_depth = 400, min_samples_leaf = 5,max_features='sqrt'()
#179.63343257206571 is train MSE
#260.54398806386484 is test MSE
#------> reduced the scores but not better than max_depth =None

#n_estimators = 1500, min_samples_leaf = 3,max_features='sqrt'
#124.09944037121438 is train MSE
#241.2655831420715 is test MSE
#with n_estimator>2000 there is no improvement over scores


#other models using Grid Search:

#n_estimators = 1000, max_depth = 400, min_samples_leaf = 10, max_features= 'sqrt()'
#254.25187865104073 is train MSE
#300.4340264672221 is test MSE

#n_estimators = 1000,max_depth = 400, min_samples_leaf = 20, max_features = 'sqrt()'
#318.5711073058445 is train MSE
#338.1087891237165 is test MSE

#n_estimators = 1000, min_samples_leaf = 20,max_features='sqrt()'
#318.5711073058445 is train MSE
#338.1087891237165 is test MSE

#n_estimators = 1000, min_samples_leaf = 10,max_features='sqrt()'
#318.5711073058445 is train MSE
#338.1087891237165 is test MSE

#n_estimators = 1000, min_samples_leaf = 5,max_features='sqrt()'
#254.25187865104073 is train MSE
#300.4340264672221 is test MSE
'''

#GRID SEARCH
clf = RandomForestRegressor()
param_grid =[
{'n_estimators':[1000],'max_depth':[400,None],'max_features':['sqrt'],
'min_samples_leaf':[5,10,20]}
 ]

grid_search = GridSearchCV(clf,param_grid,scoring= 'neg_mean_squared_error',cv=5)
grid_search.fit(x_lr,y_lr)
print (grid_search.best_params_)
print (grid_search.best_estimator_)

results = grid_search.cv_results_
for mean_scores, params in zip(results['mean_test_score'], results['params']):
 print (mean_scores, params)

new_clf =grid_search.best_estimator_

score = mse(x_lr,y_lr)
print ('train mse is ', score)
'''


#TEST SET SCORES

#preping test set
tq_new = pd.get_dummies(tq,columns = ['time_signature', 'key'])
y_tq = tq_new['song_popularity'] 
x_tq = tq_new.drop(['song_name', 'song_popularity'], axis =1)
x_tq['time_signature_0']= np.zeros((x_tq.shape[0],1))
#print(x_tq.shape, x.shape, tq_new.shape)
#print(x_tq.columns, x.columns)
#pdb.set_trace()
x_tq_z = z.transform(x_tq.to_numpy())
y_tq_new = y_tq.to_numpy()

f_pred = mod.predict(x_tq_z)
print(mse(y_tq_new,f_pred), 'linear regression test error')

f1_pred = rf_reg.predict(x_tq_z)
print(mse(y_tq_new,f1_pred), 'random forest initial model test error')

f2_pred = rf2_reg.predict(x_tq_z)
print(mse(y_tq_new,f2_pred), 'random forest final model test error')

#The best among 3 is rf2_reg model - which was the model built after hyperparameter tuning
#6.740322245360195e+29 linear regression test error
#196.408824756789 random forest initial model test error
#135.71013086724187 random forest final model test error











  




















