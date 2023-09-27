#!/usr/bin/env python
# coding: utf-8

# In[980]:


import pandas as pd
import io

df = pd.read_csv('Final_Test_Dataset.csv')
de = pd.read_csv('Final_Train_Dataset.csv')
print(df)


# In[981]:


df.head()


# In[982]:


df.shape,de.shape


# In[983]:


de.nunique()


# In[984]:


de.isna().sum()


# In[985]:


de['job_type'].fillna('Unknown', inplace=True)
df['job_type'].fillna('Unknown', inplace=True)


# In[986]:


de.head()


# In[987]:


de.info()


# In[988]:


train = de.dropna(subset=['key_skills'])


# In[989]:


de_train=train[['key_skills','job_description','job_desig','job_type','location','salary','experience']]


# In[990]:


de_train


# In[991]:


test = df.dropna(subset=['key_skills'])


# In[992]:


df_test=test[['key_skills','job_description','job_desig','job_type','location','experience']]


# In[993]:


df_test


# In[994]:


import re


# In[995]:


def clean_skill(skl):
    skills=str(skl).lower()
    skills=re.sub('\...','',skills)
    skills=re.sub(',','',skills)
    skills=re.sub(r'\s+',' ',skills)
    return skills
de_train['skills_cleaned']=de_train['key_skills'].apply(clean_skill)
df_test['skills_cleaned']=df_test['key_skills'].apply(clean_skill)


# In[996]:


de_train.head()


# In[997]:


de.job_description.fillna('missing',inplace=True)
df['job_description'].fillna('missing',inplace=True)


# In[998]:


def clean_descp(desc):
    job=str(desc).lower()
    job=re.sub(r'[^a-z]',' ',job)
    job=re.sub(r'\s+',' ',job)
    return job
de_train['job_description_cleaned']=de_train['job_description'].apply(clean_descp)
df_test['job_description_cleaned']=df_test['job_description'].apply(clean_descp)


# In[999]:


de_train.head()


# In[1000]:


def loc(loca):
    locat=loca.lower()
    locat=re.sub(r'[^a-z]',' ',locat)
    locat=re.sub(r'\s+',' ',locat)
    return locat
de_train['location_cleaned']=de_train['location'].apply(loc)
df_test['location_cleaned']=df_test['location'].apply(loc)


# In[1001]:


de_train.head()


# In[1002]:


de['job_type'].replace('NaN', 'missingjobtype', inplace=True)

de['job_type'].replace('Analytics', 'analytics', inplace=True)
de['job_type'].replace('Analytic', 'analytics', inplace=True)
de['job_type'].replace('ANALYTICS', 'analytics', inplace=True)
de['job_type'].replace('analytic', 'analytics', inplace=True)

df['job_type'].fillna('missingjobtype', inplace=True)
df['job_type'].replace('Analytics', 'analytics', inplace=True)
df['job_type'].replace('Analytic', 'analytics', inplace=True)
df['job_type'].replace('ANALYTICS', 'analytics', inplace=True)
df['job_type'].replace('analytic', 'analytics', inplace=True)

de_train['job_type_cleaned'] = train['job_type'] 
df_test['job_type_cleaned'] = test['job_type']


# In[1003]:


de_train.head()


# In[1004]:


def min_exp(val):
    exp = re.sub('-',' ',val)
    exp = exp.split(" ")
    exp = int(exp[0])
    return exp
    
def max_exp(val):
    exp = re.sub('-',' ',val)
    exp = exp.split(' ')
    exp = int(exp[1])
    return exp
    
de_train['min_exp'] = de_train['experience'].apply(lambda x : min_exp(x))
de_train['max_exp'] = de_train['experience'].apply(lambda x : max_exp(x))

df_test['min_exp'] = df_test['experience'].apply(lambda x : min_exp(x))
df_test['max_exp'] = df_test['experience'].apply(lambda x : max_exp(x))
        
de_train.head()


# In[1005]:


def clean_desig(desig):
    job_desig = desig.lower()
    job_desig = re.sub(r'[^a-z]', ' ', job_desig)
    job_desig = re.sub(r'\s+', ' ', job_desig)
    return job_desig

de_train['desig_cleaned'] = de_train['job_desig'].apply(clean_desig)
df_test['desig_cleaned'] = df_test['job_desig'].apply(clean_desig)


# In[1006]:


de_train.head()


# In[1007]:


de_train['merged'] = de_train['job_description_cleaned']+ ' ' +(de_train['desig_cleaned']  + ' ' + de_train['skills_cleaned']
                      + ' ' + de_train['job_type_cleaned'])

df_test['merged'] = (df_test['desig_cleaned'] + ' ' + df_test['job_description_cleaned'] + ' ' + df_test['skills_cleaned']
                     + ' ' + df_test['job_type_cleaned'])


# In[1008]:


de_train.head()


# In[1009]:


data_train  = de_train[['merged', 'location_cleaned', 'min_exp', 'max_exp']] 
data_test = df_test[['merged', 'location_cleaned', 'min_exp', 'max_exp']]


# In[1010]:


data_train.head()


# In[1011]:


data_test.head()


# In[1012]:


data_train = data_train.rename(columns = {'merged':'emp_info'},inplace = False)
data_train.head()


# In[1013]:


data_test = data_test.rename(columns = {'merged':'emp_info'},inplace = False)
data_test.head()


# In[1014]:


def min_sal(sal):
    val = str(sal).split("to")
    return val[0]
def max_sal(sal):
    val = str(sal).split("to")
    return val[1]

target = pd.DataFrame()
target["min_sal"] = de_train["salary"].apply(lambda x: min_sal(x))
target["max_sal"] = de_train["salary"].apply(lambda x: max_sal(x))
target1 = target.min_sal
target2 = target.max_sal


# In[1015]:


def get_ax(rows = 1,cols = 2,size = 7):
    fig, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return fig,ax


# In[1016]:


fig,ax = get_ax()
sns.distplot(data_train["emp_info"].str.len(),ax = ax[0])
sns.distplot(data_test["emp_info"].str.len(),ax = ax[1])


# In[1017]:


data_train.nunique()


# In[1018]:


fig,ax = get_ax()

sns.distplot(data_train.min_exp,ax = ax[0])
sns.distplot(data_train.max_exp,ax = ax[0])


sns.distplot(data_test.min_exp,ax = ax[1])
sns.distplot(data_test.max_exp,ax = ax[1])


# In[1019]:


sns.distplot(data_train.max_exp-data_train.min_exp)


# In[1020]:


data_train.head()


# In[1021]:


target.head()


# In[1022]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train['salary'] = le.fit_transform(train['salary'])


# #### from sklearn.model_selection import train_test_split
# 
# X_train, X_cv, y_train, y_cv = train_test_split(
#     data_train,train['salary'], test_size=0.20, 
#     stratify=train['salary'], random_state=75)

# In[1024]:


print('No. of sample texts X_train: ', len(X_train))
print('No. of sample texts X_cv   : ', len(X_cv))


# In[1025]:


X_train_merged = X_train['emp_info']
X_train_loc = X_train['location_cleaned']

X_cv_merged = X_cv['emp_info']
X_cv_loc = X_cv['location_cleaned']


# In[1026]:


from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
tf1 = TfidfVectorizer(min_df=3, token_pattern=r'\w{3,}', ngram_range=(1,3), max_df=0.9)
tf2 = TfidfVectorizer(min_df=2, token_pattern=r'\w{3,}')

X_train_merged = tf1.fit_transform(X_train_merged)
X_train_loc = tf2.fit_transform(X_train_loc)

X_cv_merged = tf1.transform(X_cv_merged)
X_cv_loc = tf2.transform(X_cv_loc)


# In[1027]:


import numpy as np
from scipy import sparse
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
X_train_MinExp = sc1.fit_transform(np.array(X_train['min_exp']).reshape(-1,1))
X_cv_MinExp = sc1.transform(np.array(X_cv['min_exp']).reshape(-1,1))
X_train_MinExp = sparse.csr_matrix(X_train_MinExp)
X_cv_MinExp = sparse.csr_matrix(X_cv_MinExp)

sc2 = StandardScaler()
X_train_MaxExp = sc2.fit_transform(np.array(X_train['max_exp']).reshape(-1,1))
X_cv_MaxExp = sc2.transform(np.array(X_cv['max_exp']).reshape(-1,1))
X_train_MaxExp = sparse.csr_matrix(X_train_MaxExp)
X_cv_MaxExp = sparse.csr_matrix(X_cv_MaxExp)


# In[1028]:


from scipy.sparse import hstack, csr_matrix

merged_train = hstack((X_train_merged, X_train_loc, X_train_MinExp, X_train_MaxExp))
merged_cv  = hstack((X_cv_merged, X_cv_loc, X_cv_MinExp, X_cv_MaxExp))


# In[1029]:


merged_train.shape, merged_cv.shape


# In[1030]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[1031]:


import lightgbm as lgb
train_data = lgb.Dataset(merged_train, label=y_train)
test_data = lgb.Dataset(merged_cv, label=y_cv)


# In[1032]:


param = {'objective': 'multiclass',
         'num_iterations': 80,
         'learning_rate': 0.04,  
         'num_leaves': 23,
         'max_depth': 7, 
         'min_data_in_leaf': 28, 
         'max_bin': 10, 
         'min_data_in_bin': 3,   
         'num_class': 6,
         'metric': 'multi_logloss'
         }


# In[1033]:


lgbm = lgb.train(params=param,
                 train_set=train_data,
                 num_boost_round=100,
                 valid_sets=[test_data])

y_pred_class = lgbm.predict(merged_cv)


# In[1034]:


predictions = []
for x in y_pred_class:
    predictions.append(np.argmax(x))

print('accuracy:', accuracy_score(y_cv, predictions))


# In[1035]:


X_train_merged = data_train['emp_info']
X_train_loc = data_train['location_cleaned']

X_test_merged = data_test['emp_info']
X_test_loc = data_test['location_cleaned']

y_train = train['salary']
X_train_merged = data_train['emp_info']
X_train_loc = data_train['location_cleaned']

X_test_merged = data_test['emp_info']
X_test_loc = data_test['location_cleaned']

y_train = train['salary']


# In[1036]:


tf1 = TfidfVectorizer(min_df=3, token_pattern=r'\w{3,}', ngram_range=(1,3))
tf2 = TfidfVectorizer(min_df=2, token_pattern=r'\w{3,}')

X_train_merged = tf1.fit_transform(X_train_merged)
X_train_loc = tf2.fit_transform(X_train_loc)

X_test_merged = tf1.transform(X_test_merged)
X_test_loc = tf2.transform(X_test_loc)


# In[1037]:


from scipy import sparse
from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
X_train_MinExp = sc1.fit_transform(np.array(de_train['min_exp']).reshape(-1,1))
X_test_MinExp = sc1.transform(np.array(df_test['min_exp']).reshape(-1,1))
X_train_MinExp = sparse.csr_matrix(X_train_MinExp)
X_test_MinExp = sparse.csr_matrix(X_test_MinExp)

sc2 = StandardScaler()
X_train_MaxExp = sc2.fit_transform(np.array(de_train['max_exp']).reshape(-1,1))
X_test_MaxExp = sc2.transform(np.array(df_test['max_exp']).reshape(-1,1))
X_train_MaxExp = sparse.csr_matrix(X_train_MaxExp)
X_test_MaxExp = sparse.csr_matrix(X_test_MaxExp)


# In[1038]:


merged_train = hstack((X_train_merged, X_train_loc, X_train_MinExp, X_train_MaxExp))
merged_test  = hstack((X_test_merged, X_test_loc, X_test_MinExp, X_test_MaxExp))


# In[1039]:


merged_test  = hstack((X_test_merged, X_test_loc, X_test_MinExp, X_test_MaxExp))
import lightgbm as lgb
train_data = lgb.Dataset(merged_train, label=y_train)

param = {'objective': 'multiclass',
         'num_iterations': 80,
         'learning_rate': 0.04, 
         'num_leaves': 23,
         'max_depth': 7, 
         'min_data_in_leaf': 28, 
         'max_bin': 10, 
         'min_data_in_bin': 3,   
         'num_class': 6,
         'metric': 'multi_logloss'
         }

lgbm = lgb.train(params=param, 
                 train_set=train_data)

predictions = lgbm.predict(merged_test)

y_pred_class = []
for x in predictions:
    y_pred_class.append(np.argmax(x))

y_pred_class = le.inverse_transform(y_pred_class)


# In[1040]:


df_sub = pd.DataFrame(data=y_pred_class, columns=['salary'])


# In[1041]:


df_sub


# In[1042]:


X_train


# In[1043]:


X_train = X_train.drop('emp_info', axis=1)
X_train = X_train.drop('location_cleaned',axis=1)


# In[ ]:





# In[1044]:


X_train


# In[1045]:


X= X_train.iloc[:,:-1].values
y= X_train.iloc[:,-1].values


# In[1046]:


X


# In[1047]:


y


# In[1048]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)


# In[1049]:


from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression


# In[1050]:


trf1 = LinearRegression()


# In[1051]:


pipe = Pipeline([('trf1',trf1)])


# In[1052]:


pipe.fit(X_train,y_train)


# In[1053]:


y_pred = pipe.predict(X_test)


# In[1054]:


plt.scatter(X_test,y_test,color='r')
plt.plot(X_test,y_pred,color='g')
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()


# In[1055]:


pipe.score(X_test,y_test)*100


# In[1056]:


import pickle


# In[1057]:


model = LinearRegression()
model.fit(X_train, y_train)


# In[1065]:


pickle.dump(model,open('model.pkl','wb'))


# In[1093]:


model=pickle.load(open('model.pkl','rb'))
model


# In[1095]:


print(model.predict(X_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




