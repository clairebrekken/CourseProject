'''
dev masterscript 
'''

#%% import packages

import pandas                  as     pd
import numpy                   as     np
import seaborn                 as     sns
import os 
import matplotlib.pyplot       as     plt
from   utils_text_clf          import utils_text_clf as utils
from   sklearn.model_selection import StratifiedKFold, \
                                      cross_validate, \
                                      cross_val_predict, \
                                      GridSearchCV
from   sklearn.pipeline        import Pipeline                            
from   sklearn.preprocessing   import StandardScaler
from   sklearn.tree            import DecisionTreeClassifier
from   sklearn.linear_model    import LogisticRegression
from   sklearn.svm             import LinearSVC, SVC
from   sklearn.neighbors       import KNeighborsClassifier
from   sklearn.naive_bayes     import GaussianNB
from   sklearn.ensemble        import RandomForestClassifier
import xgboost                 as     xgb
from   sklearn.metrics         import roc_curve
from   scipy                   import interp

# Turn interactive plotting off
plt.ion()  
import warnings
warnings.filterwarnings("ignore")

#%% Enter mutable info

data_dir    = os.getcwd() + '\data'''
results_dir = os.getcwd() + '\results'''

# training data 
#file_train = 'train.jsonl'

# feat data
file_feat   = 'train_feature_engineering.csv';

#file_train = os.path.join(data_dir, file_train) 
file_feat   = os.path.join(data_dir, file_feat) 

#%% load in data 

#df_train = utils.parse_json(file_train)
df_feat  = pd.read_csv(file_feat)

# feats
x_train  = df_feat.iloc[:, 2:]

# labels 
y_train  = df_feat.label

# convert labels to binary (1 - sarcasm)
y_train  = [1 if i == 'SARCASM' else 0 for i in y_train]

#%% check label proportions 

# print count
print('The count of sarcastic tweets is:', y_train.count(1))
print('The count of non-sarcastic tweets is:', y_train.count(0))

#%%
#===================================================
#                  _          _               _    
#  ___ _ __   ___ | |_    ___| |__   ___  ___| | __
# / __| '_ \ / _ \| __|  / __| '_ \ / _ \/ __| |/ /
# \__ \ |_) | (_) | |_  | (__| | | |  __/ (__|   < 
# |___/ .__/ \___/ \__|  \___|_| |_|\___|\___|_|\_\
#     |_|                                          
#===================================================

# create base models 

# decision tree
dtree_clf     = DecisionTreeClassifier(class_weight = 'balanced', 
                                       random_state = 42)
dtree         = Pipeline([('scaler',    StandardScaler()), 
                          ('dtree_clf', dtree_clf)]) 

# logistic regression 
logreg_clf    = LogisticRegression(n_jobs       = -1, 
                                   class_weight = 'balanced', 
                                   random_state = 42)
logreg        = Pipeline([('scaler',     StandardScaler()),
                          ('logreg_clf', logreg_clf)]) 

# linear SVM
svc_lin_clf   = LinearSVC(max_iter     = 20000, 
                          class_weight = 'balanced', 
                          random_state = 42)
svc_lin       = Pipeline([('scaler',      StandardScaler()),
                          ('svc_lin_clf', svc_lin_clf)]) 

# rbf kernel SVM
svc_rbf_clf   = SVC(kernel       = 'rbf', 
                    C            = 1, 
                    gamma        = 'auto', 
                    probability  = True, 
                    max_iter     = 20000, 
                    random_state = 42)
svc_rbf       = Pipeline([('scaler',      StandardScaler()),
                          ('svc_rbf_clf', svc_rbf_clf)]) 

# naive Bayes 
NB_clf        = GaussianNB()
NB            = Pipeline([('scaler', StandardScaler()),
                          ('NB_clf', NB_clf)]) 

# KNN
knn_clf       = KNeighborsClassifier(n_jobs       = -1)
knn           = Pipeline([('scaler',  StandardScaler()),
                          ('knn_clf', knn_clf)]) 

# random forest classifier
rndf_clf      = RandomForestClassifier(n_estimators   = 250,  
                                       max_leaf_nodes = 16, 
                                       n_jobs         = -1, 
                                       class_weight   = 'balanced', 
                                       random_state   = 42)
rndf          = Pipeline([('scaler',   StandardScaler()),
                          ('rndf_clf', rndf_clf)]) 

# xgboost
xgb_clf       = xgb.XGBClassifier(seed = 42)
xgb           = Pipeline([('scaler',  StandardScaler()),
                          ('xgb_clf', xgb_clf)]) 

#%% create model pipeline 

# Append models 
models        = [] 
models.append(['DTREE' , dtree])
models.append(['LOGREG', logreg])
models.append(['SVCLIN', svc_lin])
models.append(['SVCRBF', svc_rbf])
models.append(['KNN'   , knn])
models.append(['NB'    , NB])
models.append(['RNDF'  , rndf])
models.append(['XGB'   , xgb])


#%% set cross validation metrics 

n_splits      = 10 # folds 
cv            = StratifiedKFold(n_splits = n_splits, random_state = 42)
scores        = ['accuracy', 'recall', 'precision', 'roc_auc', 'f1']

#%% prepare cross-validation storage

# define the columns of the cross-validation results dataframe 
xval_cols     = ['classifier', 
                 'recall', 
                 'precision',
                 'f1',
                 'accuracy',
                 'roc_auc']

roc_cols      = ['classifier', 'tpr', 'fpr']

# dataframe to store cross-validation results 
df_xval       = pd.DataFrame(index = range(n_splits), columns = xval_cols)
df_xval_all   = pd.DataFrame()
df_xval_roc   = pd.DataFrame()

#%% spot check models on training data 

# loop through the classifiers 

for i, (name, model) in enumerate(models): 

    # cross-validate and compute scores 
    score_results              = cross_validate(model, 
                                                x_train, 
                                                y_train, 
                                                scoring            = scores, 
                                                cv                 = cv, 
                                                return_train_score = False, 
                                                n_jobs             = -1)
    
    print(name + ' cross-validation completed')

    # clean up the df 
    score_results              = pd.DataFrame(score_results).loc[:,['test_accuracy', 'test_f1', 'test_recall', 'test_precision', 'test_roc_auc']]
    score_results.columns      = score_results.columns.str.replace('test_','')
    
    # store the metric results 
    for metric in score_results.columns:
        df_xval.loc[:, metric] = score_results[metric]
        
    # Fill in the 'classifier' column 
    df_xval['classifier']      = np.repeat(name, n_splits, axis = 0)

    # compute class prediction probabilities 
    if hasattr(model, 'predict_proba'):
        y_pred                 = cross_val_predict(model, 
                                                   x_train, 
                                                   y_train, 
                                                   cv     = cv, 
                                                   n_jobs = -1, 
                                                   method = 'predict_proba')
        print(name + ' cross_val_predict completed')
                
    # compute the fpr, tpr
    fpr_reg                    = np.linspace(0, 1, 501) # at regular ticks 
            
    fpr, tpr, _                = roc_curve(y_train, y_pred[:,1], pos_label = 1)
    tpr                        = interp(fpr_reg, fpr, tpr)
        
    # Store fpr, tpr
    run_dict                   = dict.fromkeys(roc_cols) 
    run_dict['fpr']            = list(fpr_reg)
    run_dict['tpr']            = list(tpr)
    run_dict['classifier']     = name

    # append to the roc results dataframe 
    df_xval_roc                = df_xval_roc.append(pd.Series(run_dict).to_frame().T, ignore_index = True) 

    # append to the overall cross-validation results dataframe 
    df_xval_all                = df_xval_all.append(df_xval, ignore_index = True)

#%% review x-val results 

df_xval_all.groupby('classifier').mean()

#%% boxplot of results 

fig       = plt.figure(figsize = (20,15))
fig.tight_layout()

for i, metric in enumerate(df_xval_all.drop(['classifier'], axis = 1)):
    ax    = fig.add_subplot(3, 2, i+1)
    h     = sns.boxplot(y      = metric, 
                        x      = 'classifier', 
                        data   = df_xval_all,  
                        orient = 'v', 
                        ax     = ax)
    h.set_ylabel(metric,fontsize = 20)
    h.set_xticklabels(h.get_xticklabels(), rotation = 20)
    h.set_xlabel('classifier',fontsize = 10)
    plt.tick_params(labelsize = 10)

fig.suptitle('Cross-validation results grouped by classifier', fontsize = 20)

# save fig
fig_name  = 'x_train_xval_results.jpg'
fig_file  = os.path.join(results_dir, fig_name)

manager   = plt.get_current_fig_manager()
manager.window.showMaximized()

plt.show()
plt.pause(0.1) # needed for the image to be saved at full size
plt.savefig(fig_file)

#%% plot cross-validated ROC 

# Plot roc
plt.figure(figsize = [20,15])

for i in range(len(df_xval_roc.index)):
    
    # id classifier
    clf = df_xval_roc.loc[i, 'classifier']
    
    plt.plot(df_xval_roc.loc[i, 'fpr'], df_xval_roc.loc[i, 'tpr'], lw = 4, 
             label = df_xval_roc.loc[i, 'classifier'] + ' (AUC = %0.2f)' % df_xval_all[df_xval_all['classifier'] == clf]['roc_auc'].mean())

plt.plot([0, 1], [0, 1], color='navy', lw = 4, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize = 20)
plt.ylabel('True Positive Rate', fontsize = 20)
plt.title('Mean cross-validated ROCs', fontsize = 20)
plt.legend(loc="lower right", prop={'size': 20})
plt.tick_params(labelsize = 20)
plt.show()

# save fig
fig_name  = 'x_train_xval_roc.jpg'
fig_file  = os.path.join(results_dir, fig_name)

manager   = plt.get_current_fig_manager()
manager.window.showMaximized()
plt.show()
plt.pause(0.1) # needed for the image to be saved at full size
plt.savefig(fig_file)