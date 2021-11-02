from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn.model_selection import train_test_split

import pandas as pd
import pickle

df = pd.read_csv('final_test.csv')
df = df.fillna(df.median()) #Replace null values with median

'''
'size' column in dataset has string values.
Create a dicitonary s to map the various sizes to an integer value 
'''

s = {'XXS': 1,
     'XS': 2,
     'S': 3,
     'M': 4,
     'L': 5,
     'XL': 6,
     'XXL': 7,
     'XXXL': 8}

df = df.applymap(lambda x: s.get(x) if x in s else x) #Replace all string values with corresponding integer

X = df.iloc[:, :3]
y = df.iloc[:, -1]

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2)
print(X_train)

regressor = DecisionTreeRegressor(
    criterion='squared_error', random_state=100, max_depth=9, min_samples_leaf=1)
regressor.fit(X_train, Y_train)
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_pred, Y_test)
rmse = np.sqrt(mse)
print(rmse)
pickle.dump(regressor, open('model.pkl', 'wb'))
