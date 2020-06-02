#!/usr/bin/env python
# coding: utf-8

# In[1]:


class CustomRandomForest:
    
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor
        
        self.regressor = RandomForestRegressor()
    
    def fit(self, X_train, y_train):
        
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


