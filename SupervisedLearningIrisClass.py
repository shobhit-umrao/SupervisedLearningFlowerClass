
# coding: utf-8

# In[2]:


import sys
print(sys.version)


# In[6]:


import sys
import scipy
import numpy
import matplotlib
import pandas
import sklearn

print('Python: {}'.format(sys.version))
print('scipy: {}'.format(scipy.__version__))
print('numpy: {}'.format(numpy.__version__))
print('matplotlib: {}'.format(matplotlib.__version__))
print('pandas: {}'.format(pandas.__version__))
print('sklearn: {}'.format(sklearn.__version__))


# In[7]:


print('scipy: {}'.format(scipy.__version__))


# In[11]:


from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


# In[14]:


print('scipy: {}'.format(scipy.__version__))


# In[15]:


#loading dataset lets see ...
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
names = ['sepal-lenght' , 'sepal-width' , 'petal-length' , 'petal-width' , 'class']
dataset = pandas.read_csv(url, names=names)


# In[17]:


print(dataset.shape) #rows anfd columns


# In[18]:


print(dataset.head)


# In[20]:


print(dataset.describe())


# In[21]:


print(dataset.groupby('class').size())


# In[22]:


dataset.hist()


# In[23]:


dataset.hist()
plt.show()


# In[26]:


#scatter plotting matrix
scatter_matrix(dataset)
plt.show()


# In[27]:


#scatter plotting matrix
scatter_matrix(dataset)
plt.show()


# In[35]:


#splitting dataset in 80-20 ratio for training and cross validation
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation=model_selection.train_test_split(X,Y,test_size=validation_size,random_state=seed)


# In[36]:


#test option and evaluation matrix
scoring = 'accuracy'


# In[37]:


models = []
models.append(('LR', LogisticRegression()))


# In[38]:


models.append(('KNN',KNeighborsClassifier()))


# In[39]:


models.append((('SVM',SVC())))


# In[40]:


print(models)


# In[42]:


#evaluating each model...
results = [] #results list...
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = seed)
    cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv = kfold, scoring=scoring)
    results.append(cv_results)
    names.append(name)
    message = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(message)


# In[44]:


#make predictions on validation data set

for name, model in models:
    model.fit(X_train, Y_train)
    predictions = model.predict(X_validation)
    print(name)
    print(accuracy_score(Y_validation, predictions))
    print(classification_report(Y_validation, predictions))

