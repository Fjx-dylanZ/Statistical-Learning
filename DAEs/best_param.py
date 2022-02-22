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