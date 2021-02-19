import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

### Sources:
### https://towardsdatascience.com/random-forest-in-python-24d0893d51c0
### https://medium.com/@hjhuney/implementing-a-random-forest-classification-model-in-python-583891c99652
### https://medium.com/@saeedAR/smote-and-near-miss-in-python-machine-learning-in-imbalanced-datasets-b7976d9a7a79
### http://rikunert.com/SMOTE_explained

fileout='classification_4xmm_RF.csv'
testsample=0

#prepare the data
features=pd.read_csv('4xmmdr9_toclassify_221120_RF.csv')
#features=pd.read_csv('2SXPS_toclassify_041120_RF.csv')
#features=pd.read_csv('catalogs/2SXPS_toclassify_RefSample_RF.csv')
#features=pd.read_csv('catalogs/1SXPSslim_GS_toclassify_RF.csv')
#print(features.head())

y=features["class"] #Oclass
# Make only 3 classes...
#y=np.minimum(y,2)


X=features.drop("class",axis=1) #Oclass
todrop=['2SXPS_ID','SRCID']#,'WAPO_GOODN', 'SC_EXT_ML', 'SC_DET_ML']


# X=X.drop("SRCID",axis=1)
# X=X.drop("WAPO_GAMMA_HI",axis=1)
# X=X.drop("WAPO_NH",axis=1)
# X=X.drop("WAPO_GOODN",axis=1)
# X=X.drop("logFxRatio",axis=1)
# X=X.drop("logStdRatio",axis=1)
# X=X.drop("logChiDOFcst",axis=1)
for f in list(X.columns):
    try:
        #print('\n'.join(['%s: %d'%(f,sum(np.isnan(X[f])))]))
        X.loc[np.isnan(X[f]),f] = -99
    except:
        print(f)



if not(testsample):
    itest=np.where(np.isnan(y)|(y==99))[0]
    X, y = X.drop(X.index[itest]),y.drop(y.index[itest])


if not(testsample):
    fileout=fileout[:-4]+'_test.csv'
    seed=2#np.random.randint(20)
    X_train, X_Test, y_train, y_Test = train_test_split(X, y, 
                                                        test_size=0.3, random_state=seed)

    for feature in todrop:
        if feature in list(X_train.columns):
            #print('dropped column "%s"'%feature)
            colSRCID,SRCID=feature,X_Test[feature]
            X_train=X_train.drop(feature,axis=1)
            X_Test=X_Test.drop(feature,axis=1)
    feature_list=list(X_Test.columns)
            
    equipart=[0.66,0.25,0.07,.02]
    limiting_class=np.argmin([sum(y_Test==i)/equipart[i] for i in range(4)])
    #print('limiting class:', limiting_class)
    nClasses=[int(sum(y_Test==limiting_class)*equipart[i]/equipart[limiting_class]) for i in range(4)]
    #print(nClasses)
    ind=[np.where(y_Test==i)[0] for i in range(4)]
    [np.random.shuffle(ind[i]) for i in range(4)]
    idrop=np.concatenate([np.where(y_Test==i)[0][nClasses[i]:] for i in range(4)])
    #print(sum(nClasses),len(idrop))
    X_test=X_Test.drop(X_Test.index[idrop])
    y_test=y_Test.drop(y_Test.index[idrop])

    #make equal sampling for each class
    #smt=SMOTE()
    #X_train, y_train = smt.fit_sample(X_train, y_train)
    np.bincount(y_train)
    rfc=RandomForestClassifier(max_depth=12)

    # for f in list(X.columns):
    #     if not(f in ['SRCID','SC_DET_ML','SC_EXT_ML']):
    #         X_train[f][abs(X_train[f])>1000]=np.nan
    #         X_test[f][abs(X_test[f])>1000]=np.nan
    
    # imp=SimpleImputer(missing_values=np.nan,strategy='mean')
    # imp=imp.fit(X_train)
    # X_train=imp.transform(X_train)
    # X_test=imp.transform(X_test)

    #build the forest
    rfc.fit(X_train,y_train,sample_weight=0.01*np.ones(len(y_train))+0.99*(y_train==2))
    
    #predictions
    rfc_predict = rfc.predict(X_test)
    
    #cross-validation
    #rfc_cv_score = cross_val_score(rfc, X, y, cv=10, scoring='roc_auc')
    
    #print the results
    
    #print("=== Confusion Matrix ===")
    #print(confusion_matrix(y_test, rfc_predict))
    #print('\n')
    print("=== Classification Report (seed=%d) ==="%seed)
    print(classification_report(y_test, rfc_predict))
    print('\n')
    
    rfc_predict = rfc.predict(X_Test)
    probas = rfc.predict_proba(X_Test)
    X_Test.insert(0,colSRCID,SRCID,True)
    X_Test['class']=y_Test
    X_Test['prediction']=rfc_predict
    Classes=sorted(np.unique(y_train))
    for iC in range(len(Classes)):
        X_Test['PC%s'%str(Classes[iC])]=probas[:,iC]
    X_Test.to_csv(fileout,index=False)
    
        
else:
    for feature in todrop:
        if feature in list(X.columns):
            #print('dropped column "%s"'%feature)
            colSRCID,SRCID=feature,X[feature]
            X=X.drop(feature,axis=1)
    feature_list=list(X.columns)
    
    itest=np.where(np.isnan(y)|(y==99))[0]
    X_train, y_train = X.drop(X.index[itest]),y.drop(y.index[itest])

    #build the forest
    rfc=RandomForestClassifier(max_depth=12)
    rfc.fit(X_train,y_train,sample_weight=0.01*np.ones(len(y_train))+0.99*(y_train==2))
    
    #predictions
    rfc_predict = rfc.predict(X)
    probas = rfc.predict_proba(X)
    Classes=sorted(np.unique(y_train))
    #print(Classes,probas[np.random.randint(len(probas)):][:10])
    
    X.insert(0,colSRCID,SRCID,True)
    X['class']=y
    X['prediction']=rfc_predict
    for iC in range(len(Classes)):
        X['PC%s'%str(Classes[iC])]=probas[:,iC]
        
    X.to_csv(fileout,index=False)
    
#get feature importance
feature_imp = pd.Series(rfc.feature_importances_,index=feature_list).sort_values(ascending=False)
#print(feature_imp)


### TUNING HYPERPARAMETERS... ###

# from sklearn.model_selection import RandomizedSearchCV
# # number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# # number of features at every split
# max_features = ['auto', 'sqrt']

# # max depth
# max_depth = [int(x) for x in np.linspace(100, 500, num = 11)]
# max_depth.append(None)
# # create random grid
# random_grid = {
#  'n_estimators': n_estimators,
#  'max_features': max_features,
#  'max_depth': max_depth
#  }
# # Random search of parameters
# rfc_random = RandomizedSearchCV(estimator = rfc, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=2, random_state=42, n_jobs = -1)
# # Fit the model
# rfc_random.fit(X_train, y_train)
# # print results
# print(rfc_random.best_params_)

exit()

# Import tools needed for visualization
from sklearn.tree import export_graphviz
import pydot
# Pull out one tree from the forest
tree = rfc.estimators_[5]
# Export the image to a dot file
export_graphviz(tree, out_file = 'tree.dot', feature_names = feature_list, rounded = True, precision = 1)
# Use dot file to create a graph
(graph, ) = pydot.graph_from_dot_file('tree.dot')
# Write graph to a png file
graph.write_png('tree.png')


# Limit depth of tree to 3 levels
rfc_small = RandomForestClassifier(max_depth = 3)
rfc_small.fit(X_train, y_train)
# Extract the small tree
tree_small = rfc_small.estimators_[5]
# Save the tree as a png image
export_graphviz(tree_small, out_file = 'small_tree.dot', feature_names = feature_list, rounded = True, precision = 1)
(graph, ) = pydot.graph_from_dot_file('small_tree.dot')
graph.write_png('small_tree.png');



# See Arnason+20 for details on possible improvements...
# => notably, make a binary version SMCO / non-SMCO ? Pb AGN...
# 2 steps: 1/ Star/non-star 2/ CO/non-CO
# New class: SNR (~ Star)

