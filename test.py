print("hello")
import numpy as np
print(np.__version__)


import pandas as pd
data = pd.read_csv('bp.csv')
data.head()
y = data['Systolic BP (mm Hg)']
x = data.drop(['Systolic BP (mm Hg)','Subject_ID'], axis=1)
print(x.shape, y.shape)
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)