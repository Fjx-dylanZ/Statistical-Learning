'''
search space
params = {
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 500, 800],
    'gamma': [0, 0.5, 1], # regularization
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9]
}
'''

xg_boost_param = {'colsample_bytree': 0.5,
 'gamma': 0.5,
 'learning_rate': 0.01,
 'max_depth': 3,
 'n_estimators': 500,
 'subsample': 0.8} # 2.5h havling random search


'''
params = {
    'max_depth': [2,3,4,5],
    'learning_rate': [0.0001, 0.001, 0.01, 0.02, 0.1],
    'n_estimators': [150, 200, 300, 500, 800, 1200],
    'gamma': [0, 0.2, 0.5, 1, 2], # regularization
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.4, 0.5, 0.7, 0.9, 1.0]
}
'''


xg_boost_param_2 = {'colsample_bytree': 0.5,
 'gamma': 0,
 'learning_rate': 0.01,
 'max_depth': 2,
 'n_estimators': 200,
 'subsample': 0.6}

xg_boost_param_3 = {'alpha': 0.012340293043913065,
 'colsample_bytree': 0.2014784460122454,
 'gamma': 0.0014879091585161574,
 'lambda': 0.0002453710647169391,
 'learning_rate': 0.0054201804563815585,
 'max_depth': 10,
 'min_child_weight': 10,
 'n_estimators': 596,
 'subsample': 0.5287916198749903}