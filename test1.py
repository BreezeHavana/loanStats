import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("./LoanStats_2017Q2.csv",skiprows=1)
#观察数据构成
df.info()
df.head()
df.iloc[:4,:8]
#调整int_rate字段格式
df.int_rate.replace(r'\%','',True,regex=True)
df.int_rate.astype(float)
#去除所有nan
df.dropna(0, 'all',True)
df.dropna(1, 'all',True)

df.info()
df.iloc[:4,8:15]
#对emp_length字段进行处理