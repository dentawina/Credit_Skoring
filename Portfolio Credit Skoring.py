#!/usr/bin/env python
# coding: utf-8

# # IMPORT LIBRARY

# In[1]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
plt.style.use('fivethirtyeight')
import plotly
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, roc_auc_score
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, RobustScaler, StandardScaler, MinMaxScaler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import classification_report,confusion_matrix
from lazypredict.Supervised import LazyClassifier
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
import warnings
warnings.filterwarnings("ignore")


# # Read Data

# In[2]:


#bacadata
df = pd.read_csv('loan_data_2007_2014.csv',index_col = 0)
df.head()


# # Data Understanding

# In[3]:


cols_to_drop = [
    # unique id
    'id'
    , 'member_id'
    
    # free text
    , 'url'
    , 'desc'
    
    # all null / constant / others
    , 'zip_code' 
    , 'annual_inc_joint'
    , 'dti_joint'
    , 'verification_status_joint'
    , 'open_acc_6m'
    , 'open_il_6m'
    , 'open_il_12m'
    , 'open_il_24m'
    , 'mths_since_rcnt_il'
    , 'total_bal_il'
    , 'il_util'
    , 'open_rv_12m'
    , 'open_rv_24m'
    , 'max_bal_bc'
    , 'all_util'
    , 'inq_fi'
    , 'total_cu_tl'
    , 'inq_last_12m'
    
    # expert judgment
    , 'sub_grade'
]


# In[4]:


df = df.drop(cols_to_drop, axis=1)


# In[5]:


df.shape


# In[6]:


df.describe()


# In[7]:


df.isnull().sum().sort_values(ascending = False)


# In[8]:


# mendefinisikan nilai
good_loans = ['Current', 'Fully Paid', 'In Grace Period', 
              'Does not meet the credit policy. Status:Fully Paid']

# membuat kolom baru untuk klasifikasi akhir
df['good_bad_loan'] = np.where(df['loan_status'].isin(good_loans), 1, 0)


# In[9]:


#membuat perbandingan antara good loan VS bad loan
plt.title('Good (1) vs Bad (0) Loans Balance')
sns.barplot(x=df.good_bad_loan.value_counts().index,y=df.good_bad_loan.value_counts().values)


# # Feature Engineering

# In[10]:


#Transformasi tipe Data
df_transform = ['term', 'emp_length', 'earliest_cr_line','last_credit_pull_d']
df[df_transform]


# In[11]:


#1.Konversikan kolom TERM ke tipe data numerik dan menghapus kata (months)
df['term'] = pd.to_numeric(df['term'].str.replace(' months', ''))
df['term']


# In[12]:


#2.Konversikan kolom emp_length ke tipe data numerik dan menghapus kata (years)
emp_map = {
    '< 1 year' : '0',
    '1 year' : '1',
    '2 years' : '2',
    '3 years' : '3',
    '4 years' : '4',
    '5 years' : '5',
    '6 years' : '6',
    '7 years' : '7',
    '8 years' : '8',
    '9 years' : '9',
    '10+ years' : '10'
}

df['emp_length'] = df['emp_length'].map(emp_map).fillna('0').astype(int)
df['emp_length'].unique()


# In[13]:


#3 menyeragamkan kolom earliest_cr_line
df['earliest_cr_line_date'] = pd.to_datetime(df['earliest_cr_line'], format = '%b-%y')


# In[14]:


# Asumsi sekarang  akhir bulan December 2014
df['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2014-12-31') - df['earliest_cr_line_date']) / np.timedelta64(1, 'M')))
df['mths_since_earliest_cr_line'].describe()


# In[15]:


# tampilkan baris di mana variabel memiliki nilai negatif
df.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][df['mths_since_earliest_cr_line'] < 0]


# In[16]:


df['earliest_cr_line_date'] = df['earliest_cr_line_date'].astype(str)
df['earliest_cr_line_date'][df['mths_since_earliest_cr_line'] < 0] = df['earliest_cr_line_date'][df['mths_since_earliest_cr_line'] < 0].str.replace('20','19')


# In[17]:


df['earliest_cr_line_date'] = pd.to_datetime(df['earliest_cr_line_date'])
df['earliest_cr_line_date']


# In[18]:


#hapus kolom earliest_cr_line_date, mths_since_earliest_cr_line dan earliest_cr_line karena tidak digunakan lagi
df.drop(columns = ['earliest_cr_line_date' ,'mths_since_earliest_cr_line', 
                          'earliest_cr_line'], inplace = True)


# In[19]:


#4 menyeragamkan kolom last credit pull d untuk menjadi numerik 
df['last_credit_pull_d']


# In[20]:


# Asumsikan sekarang  akhir bulan desember / awal bulan januari
# Extracts the date and the time from a string variable that is in a given format. and fill NaN data with max date
df['last_credit_pull_d'] = pd.to_datetime(df['last_credit_pull_d'], format = '%b-%y').fillna(pd.to_datetime("2015-01-01"))

# hitung selisih antara dua tanggal dalam bulan, ubah menjadi tipe data numerik dan bulatkan
df['mths_since_last_credit_pull_d'] = round(pd.to_numeric((pd.to_datetime('2018-12-31') - df['last_credit_pull_d']) / np.timedelta64(1, 'M')))

# Menampilkan deskripsi statistik dari kolom tersebut
df['mths_since_last_credit_pull_d'].describe()


# In[21]:


# Menghapus kolom last_credit_pull_d karena tidak terpakai lagi
df.drop(columns = ['last_credit_pull_d'], inplace = True)


# In[22]:


df.info()


# In[23]:


df.select_dtypes(include='object').nunique()


# In[24]:


df.drop(['emp_title', 'title', 'application_type'], axis=1, inplace=True)


# In[25]:


df.select_dtypes(exclude='object').nunique()


# In[26]:


df.drop(['policy_code'], axis=1, inplace=True)


# In[27]:


for col in df.select_dtypes(include='object').columns.tolist():
    print(df[col].value_counts(normalize=True)*100)
    print('\n')


# In[28]:


df.drop('pymnt_plan', axis=1, inplace=True)


# In[29]:


df.info()


# In[30]:


df.drop(['loan_status','mths_since_last_record'],axis=1,inplace=True)


# # train test split

# In[31]:


# memisahkan variabel target (y) dan prediktor (X)
X = df.drop('good_bad_loan',axis =1)
y = df['good_bad_loan']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = 42)


# In[32]:


from sklearn.preprocessing import LabelEncoder

# List of columns to be transformed
columns_to_transform = ['grade', 'verification_status', 'issue_d', 'purpose', 'addr_state',
                        'initial_list_status', 'last_pymnt_d', 'next_pymnt_d','home_ownership']

# Create a LabelEncoder object
le = LabelEncoder()

# Loop through each column and apply label encoding for X_train
for col in columns_to_transform:
    le.fit(X_train[col])
    X_train[col] = le.transform(X_train[col])


# In[33]:


for col in columns_to_transform:
    le.fit(X_test[col])
    X_test[col] = le.transform(X_test[col])


# In[34]:


check_missing = X_train.isnull().sum() * 100 / X_train.shape[0]
check_missing[check_missing > 0].sort_values(ascending=False)


# In[35]:


X_train.drop(['mths_since_last_major_derog','mths_since_last_delinq'],axis=1,inplace=True)


# In[36]:


X_test.drop(['mths_since_last_major_derog','mths_since_last_delinq'],axis=1,inplace=True)


# In[37]:


X_train.isnull().sum()


# In[38]:


# Misalkan 'X_train' adalah DataFrame yang berisi data training Anda
# Misalkan data training memiliki kolom-kolom berikut
columns_to_fillna = ['annual_inc', 'acc_now_delinq', 'total_acc', 'pub_rec', 'open_acc',
                     'inq_last_6mths', 'delinq_2yrs', 'collections_12_mths_ex_med',
                     'revol_util', 'tot_cur_bal', 'tot_coll_amt','total_rev_hi_lim']

# Mengisi nilai yang hilang dengan nilai rata-rata (mean) dari masing-masing kolom
for col in columns_to_fillna:
    X_train[col].fillna(X_train[col].mean(), inplace=True)


# In[39]:


for col in columns_to_fillna:
    X_test[col].fillna(X_test[col].mean(), inplace=True)


# In[40]:


X_train.shape


# In[41]:


# Hapus nilai infinit
X_train = X_train.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Ganti nilai infinit dengan median
X_train.fillna(X_train.median(), inplace=True)
X_test.fillna(X_train.median(), inplace=True)


# In[42]:


oversample = SMOTE()
X_train, y_train = oversample.fit_resample(X_train, y_train)


# In[43]:


X_train.shape


# In[44]:


X_test.shape


# In[45]:


# Scaling dengan RobustScaler
rs = RobustScaler()
X_train_transformed = rs.fit_transform(X_train)
X_test_transformed = rs.transform(X_test)


# # Feature Selection

# In[46]:


X_train_transformed_df = pd.DataFrame(data=X_train_transformed, columns=X_train.columns)
corr = X_train_transformed_df.corr(method = 'spearman')
plt.figure(figsize=(25,25))
sns.heatmap(corr, annot=True)


# In[47]:


var = []
drop = []
for x in X_train_transformed_df.columns:
  for y in X_train_transformed_df.columns:
    if x != y:
      if [y,x] not in var:
        corr, p_value = spearmanr(X_train_transformed_df[x], X_train_transformed_df[y])
        var.append([x, y])
        if (corr <= -0.8) | (corr >= 0.8):
          if p_value < 0.05:
            drop.append(y)


# In[48]:


drop


# In[49]:


X_train_transformed_df = X_train_transformed_df.drop(drop, axis=1)


# In[50]:


X_train_transformed_df.shape


# In[51]:


mutual_info_classif(X_train_transformed_df,
                    y_train,
                    random_state = 123)


# In[52]:


mutual_table = pd.DataFrame(mutual_info_classif(X_train_transformed_df,y_train,random_state = 123),
                            index = X_train_transformed_df.columns,
                            columns = ['mutual_info']).sort_values('mutual_info', ascending = False)


# In[53]:


mutual_table


# In[54]:


X_train_mt = mutual_table.iloc[0:19].index


# In[55]:


X_train = X_train_transformed_df.loc[:, X_train_mt]


# In[56]:


X_train.shape


# In[57]:


X_test_transformed.shape


# In[58]:


X_test_transformed_df = pd.DataFrame(data=X_test_transformed, columns=X_test.columns)


# In[59]:


X_test = X_test_transformed_df.loc[:,list(X_train.columns)]


# In[60]:


X_test.shape


# In[61]:


y_train.value_counts()


# In[62]:


rfc = RandomForestClassifier(max_depth=4)
rfc.fit(X_train, y_train)
# Make predictions on the test set
y_pred = rfc.predict(X_test)


# In[63]:


report = classification_report(y_test, y_pred)
print('Classification report:\n', report)


# In[64]:


rfc.score(X_train,y_train),rfc.score(X_test,y_test)


# In[65]:


y_pred_proba = rfc.predict_proba(X_test)[:][:,1]

df_actual_predicted = pd.concat([pd.DataFrame(np.array(y_test), columns=['y_actual']), pd.DataFrame(y_pred_proba, columns=['y_pred_proba'])], axis=1)
df_actual_predicted.index = y_test.index


# In[66]:


fpr, tpr, tr = roc_curve(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])
auc = roc_auc_score(df_actual_predicted['y_actual'], df_actual_predicted['y_pred_proba'])

plt.plot(fpr, tpr, label='AUC = %0.4f' %auc)
plt.plot(fpr, fpr, linestyle = '--', color='k')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()


# In[67]:


df_actual_predicted = df_actual_predicted.sort_values('y_pred_proba')
df_actual_predicted = df_actual_predicted.reset_index()

df_actual_predicted['Cumulative N Population'] = df_actual_predicted.index + 1
df_actual_predicted['Cumulative N Bad'] = df_actual_predicted['y_actual'].cumsum()
df_actual_predicted['Cumulative N Good'] = df_actual_predicted['Cumulative N Population'] - df_actual_predicted['Cumulative N Bad']
df_actual_predicted['Cumulative Perc Population'] = df_actual_predicted['Cumulative N Population'] / df_actual_predicted.shape[0]
df_actual_predicted['Cumulative Perc Bad'] = df_actual_predicted['Cumulative N Bad'] / df_actual_predicted['y_actual'].sum()
df_actual_predicted['Cumulative Perc Good'] = df_actual_predicted['Cumulative N Good'] / (df_actual_predicted.shape[0] - df_actual_predicted['y_actual'].sum())


# In[68]:


df_actual_predicted.head()


# In[69]:


KS = max(df_actual_predicted['Cumulative Perc Good'] - df_actual_predicted['Cumulative Perc Bad'])

plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Bad'], color='r')
plt.plot(df_actual_predicted['y_pred_proba'], df_actual_predicted['Cumulative Perc Good'], color='b')
plt.xlabel('Estimated Probability for Being Bad')
plt.ylabel('Cumulative %')
plt.title('Kolmogorov-Smirnov:  %0.4f' %KS)


# In[72]:


import joblib
model_filename = 'rfc.joblib'
joblib.dump(rfc, model_filename)


# In[73]:


print(f"Model telah disimpan sebagai {model_filename}")


# In[ ]:




