import pandas as pd
import numpy as np
df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",header=None)
from sklearn.preprocessing import LabelEncoder
x = df.iloc[:,2:]
y = df.iloc[:,1]
le = LabelEncoder()
y = le.fit_transform(y)
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=0)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
pipe_lr = Pipeline([("scl",StandardScaler()),
                    ("pca",PCA(n_components=2)),
                    ("clf",LogisticRegression(random_state=1))])
pipe_lr.fit(x_train,y_train)