#!/usr/bin/env python
# coding: utf-8

# In[1]:


class CustomRandomForest:
    
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        
        self.regressor = RandomForestRegressor()

    def tune_parameters(self, X_train, y_train):
        from sklearn.model_selection import GridSearchCV
        
        parameters = {'n_estimators': [75, 100, 200, 250], 'max_depth': [5, 10, 15, 20]}
        regressor_ = GridSearchCV(self.regressor, parameters, cv=5, n_jobs=-1)
        regressor_.fit(X_train, y_train)
        
        self.best_parameters = regressor_.best_params_
        self.best_score = regressor_.best_score_
        
        print('Best parameters : {}'.format(self.best_parameters))
        print('Best score : {}'.format(self.best_score))
    
    def fit(self, X_train, y_train):
        from sklearn.ensemble import RandomForestRegressor
        
        self.regressor = RandomForestRegressor(n_estimators = self.best_parameters['n_estimators'], 
                                               max_depth = self.best_parameters['max_depth'])
        self.model = self.regressor.fit(X_train, y_train)
        
    def predict(self, X_test):
        
        y_pred = self.model.predict(X_test)
        return y_pred
    
    def evaluate(self, y_test, y_pred):
        import pandas as pd
        from sklearn.metrics import r2_score, mean_squared_error
        
        r2_rf = r2_score(y_test, y_pred)*100
        mse_rf = mean_squared_error(y_test, y_pred)
        
        scores = [['R2 Score', r2_rf],
                  ['Mean Squared Error', mse_rf]]
        df = pd.DataFrame(scores, columns=['Metrics', 'Score'])
        return df

