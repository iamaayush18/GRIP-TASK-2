#!/usr/bin/env python
# coding: utf-8

# <h1>Task 2</h1>

# <h2>Prediction using Unsupervised Machine Learning</h2>

# <h2>Author: Kumar Ayush</h2>

# Dataset: https://bit.ly/3kXTdox

# <h3>Importing libraries</h3>

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


# <h3>Reading Data</h3>

# In[2]:


df=pd.read_csv(r'C:\Users\Lenovo\Downloads\Iris.csv')


# <h3>Exploring Data</h3>

# In[3]:


df.drop('Id',axis=1,inplace=True)


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.describe()


# In[7]:


df.isnull().sum()


# In[8]:


df.Species.nunique()
df.Species.value_counts()


# <h3>Data Visualization</h3>

# In[9]:


sns.boxplot(x=df.Species,y=df.SepalLengthCm)
plt.title('Iris Dataset')


# In[10]:


sns.boxplot(x=df.Species,y=df.SepalWidthCm)
plt.title('Iris Dataset')


# In[11]:


sns.boxplot(x=df.Species,y=df.PetalLengthCm)
plt.title('Iris Dataset')


# In[12]:


sns.boxplot(x=df.Species,y=df.PetalWidthCm)
plt.title('Iris Dataset')


# In[13]:


sns.heatmap(df.corr(),annot=True)


# <h3>K-Means Clustering</h3>

# In[14]:


x=df.iloc[:,[0,1,2,3]].values
sse=[]
for i in range (1,11):
    kmeans=KMeans(n_clusters=i,random_state=0,max_iter=300,n_init=10)
    kmeans.fit(x)
    sse.append(kmeans.inertia_)
    print('K:',i,'SSE:',kmeans.inertia_)


# In[15]:


plt.plot(range(1,11),sse)
plt.xlabel('K')
plt.ylabel("Sum of Squared Error")
plt.title('The Elbow Method')


# In[16]:


kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans


# In[17]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='Blue',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='Green',label='Iris-versicolor')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='Yellow',label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,marker='*',color='red',label='Centroids')
plt.title('Iris Data Clusters')
plt.legend()


# <h3>Normalizing the Data to cluster more accurately</h3>

# In[18]:


from sklearn import preprocessing


# In[19]:


sc=preprocessing.Normalizer()
x=sc.fit_transform(x)
kmeans=KMeans(n_clusters=3,init='k-means++',max_iter=300,n_init=10,random_state=0)
y_kmeans=kmeans.fit_predict(x)
y_kmeans


# In[20]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='Blue',label='Iris-setosa')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='Green',label='Iris-versicolor')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='Yellow',label='Iris-virginica')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,marker='*',color='red',label='Centroids')
plt.title('Iris Data Clusters')
plt.legend()


# In[ ]:




