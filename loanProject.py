#coding:utf-8
import pandas as pd
import numpy as np
import missingno as msno
from sklearn.preprocessing import Imputer
import matplotlib.pyplot as plt
import seaborn as sns
import compute_ks as ks
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import ensemble
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.under_sampling import NearMiss
from sklearn.metrics import accuracy_score
df = pd.read_csv("./LoanStats_2017Q1.csv",skiprows=1)
#观察数据构成
df.info()
df.head()
#去除空行
df.dropna(thresh = 0.6 * len(df.columns), axis = 0, inplace = True)
#新增衍生变量installment_per，借贷人与共同借贷人每年还款额占收入的比例
np.where(np.isnan(df.dti))
imp = Imputer(strategy = 'median')
imp.fit(df.dti.values.reshape(-1, 1))
df.dti = imp.transform(df.dti.values.reshape(-1, 1))
df.annual_inc_joint.fillna(0, axis = 0, inplace = True)
df.dti_joint.fillna(0, axis = 0, inplace = True)
df.installment_per = (df.dti * df.annual_inc + df.dti_joint * df.annual_inc_joint + df.installment * 12) / (df.annual_inc_joint + df.annual_inc)
#查看各特征缺失的数据量并排序
df.columns
df.describe().T.assign(missing_pct=df.apply(lambda x : (len(x)-x.count())/float(len(x)))).sort(columns='missing_pct',ascending=False)
#去除缺失较多的特征
df.dropna(thresh = 0.6 * len(df), axis = 1, inplace = True)
#查看类别较多的离散变量
df.select_dtypes(include=['object']).describe().T.sort(columns='unique',ascending=False)
#调整int_rate字段格式
df.int_rate.replace(r'\%','',True,regex=True)
df.int_rate.astype(float)
df.int_rate = df.int_rate.astype(float)
df.info()
#对emp_length字段进行处理
df.emp_length.value_counts()
df.emp_length.replace(to_replace='[^0-9]+', value='', inplace=True, regex=True)
df.emp_length.replace(to_replace='', value=0, inplace=True)
df.emp_length = df.emp_length.astype(int)
df.revol_util.replace(r'\%','',True,regex=True)
df.revol_util = df.revol_util.astype(float)
#去除只有一类的特征
df = df.loc[:,df.apply(pd.Series.nunique) != 1]
#缺失值可视化
msno.matrix(df.select_dtypes(include=["object"]))
#填充离散变量缺失值
objColumns = df.select_dtypes(include=["object"]).columns
df[objColumns] = df[objColumns].fillna("Unknown")
msno.matrix(df.select_dtypes(include=["object"]))
#直接用中位数填充连续变量缺失值
msno.matrix(df.select_dtypes(include=[np.number]))
imp = Imputer(strategy = 'median')
imp.fit(df.select_dtypes(include=[np.number]))
numColumns = df.select_dtypes(include=[np.number]).columns
df[numColumns] = imp.transform(df[numColumns])
#######
#去除类别较多/无意义的离散变量
df.drop(['emp_title','zip_code','earliest_cr_line','addr_state','sub_grade','issue_d'],1,inplace=True)
df.drop(['title', 'pymnt_plan'],1,inplace=True)
#去除贷后相关字段
df.drop(['out_prncp','out_prncp_inv','total_pymnt','total_pymnt_inv','total_rec_prncp','hardship_flag',\
         'grade','total_rec_int','total_rec_late_fee','recoveries','collection_recovery_fee',\
         'last_pymnt_d','last_pymnt_amnt','next_pymnt_d','last_credit_pull_d','initial_list_status'],1, inplace=True)
#查看借款状态的种类
df.loan_status.unique()
df.loan_status.replace('Fully Paid',1,inplace=True)
df.loan_status.replace('Current',1,inplace=True)
df.loan_status.replace('In Grace Period',np.nan,inplace=True)
df.loan_status.replace('Late (31-120 days)',0,inplace=True)
df.loan_status.replace('Late (16-30 days)',0,inplace=True)
df.loan_status.replace('Charged Off',0,inplace=True)
df.loan_status.replace('Default',0,inplace=True)
df.dropna(subset=['loan_status'],inplace=True)

y = df.loan_status
dummie_df = df.drop(['loan_status'],1,inplace = False)
dummie_df = pd.get_dummies(dummie_df)
#标准化(后面用LR，若非线性模型则不需标准化)
sc = StandardScaler()
std_df = pd.DataFrame(sc.fit_transform(dummie_df), columns=dummie_df.columns)
#检查变量相关性
cor = std_df.corr()
cor.iloc[:,:] = np.tril(cor,-1)
cor = cor.stack()
cor[(cor > 0.70)|(cor < -0.70)]
#热力图
sns.heatmap(std_df.corr()[(std_df.corr() > 0.7) | (std_df.corr() < -0.7)],linewidths=0.1)
#去除相关度很高的变量(filter)
std_df.drop(['funded_amnt','funded_amnt_inv','installment','total_acc','dti_joint','open_il_24m','open_rv_24m',
         'total_rev_hi_lim','acc_open_past_24mths','avg_cur_bal','bc_util','num_actv_rev_tl','num_bc_sats',
         'num_bc_tl','num_op_rev_tl','num_rev_accts','num_rev_tl_bal_gt_0','num_sats','num_tl_30dpd','num_tl_op_past_12m',
         'percent_bc_gt_75','tax_liens','tot_hi_cred_lim','total_bal_ex_mort','total_bc_limit',
         'total_il_high_credit_limit','term_ 60 months','home_ownership_RENT','application_type_INDIVIDUAL',
         'application_type_JOINT'],axis=1,inplace=True)
#recursive feature elimination(wrapper)
model = LogisticRegression()
rfe = RFE(model, 15)
y = pd.Series(df.loan_status)
rfe = rfe.fit(std_df, y)
std_df = std_df[std_df.columns[rfe.support_]]
#RF(embedded)
clf = RandomForestClassifier()
clf.fit(std_df, y)
std_df.columns
clf.feature_importances_
std_df.drop(['verification_status_Verified','purpose_medical'],axis=1,inplace=True)
nm = NearMiss()
x, y = nm.fit_sample(std_df, y)
#建立模型
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)
param_grid = {'C' : [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]}
grid_search = GridSearchCV(LogisticRegression(),  param_grid, cv=10)
grid_search.fit(x_train, y_train)
y_pred = grid_search.predict(x_test)