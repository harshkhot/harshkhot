#!/usr/bin/env python
# coding: utf-8

# In[8]:


import cv2
import numpy as np


# In[9]:


img=cv2.imread("C:\\Users\\abc\\Desktop\\img1.jpg")


# In[10]:


img.shape


# In[11]:


img[0]


# In[12]:


import matplotlib.pyplot as plt


# In[13]:


plt.imshow(img)


# In[14]:


while True:
    cv2.imshow('result',img)
    #27 is ASCII for escape
    if cv2.waitKey(2)==27:
        break
cv2.destroyAllWindows()


# In[15]:


haar_data=cv2.CascadeClassifier("C:\\Users\\abc\\Desktop\\data.xml")


# In[16]:


haar_data.detectMultiScale(img)


# In[17]:


while True:
    faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        #for making rectangle around the faces (img,x,y,w,h,(b,g,r),width of rectangle)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
    cv2.imshow('result',img)
    #27 is ASCII for escape
    if cv2.waitKey(2)==27:
        break
cv2.destroyAllWindows()


# In[20]:


capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        #for making rectangle around the faces (img,x,y,w,h,(b,g,r),width of rectangle)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
        face=img[y:y+h,x:x+w,:]
        face=cv2.resize(face,(50,50))
        print(len(data))
        if len(data)<400:
            data.append(face)
    cv2.imshow('result',img)
    #27 is ASCII for escape
    if cv2.waitKey(2)==27 or len(data)>=200:
        break
capture.release()
cv2.destroyAllWindows()


# In[19]:


np.save('without_mask.npy',data)


# In[21]:


np.save('with_mask.npy',data)


# In[22]:


plt.imshow(data[0])


# In[23]:


with_mask=np.load('with_mask.npy')


# In[24]:


without_mask=np.load('without_mask.npy')


# In[25]:


with_mask.shape


# In[26]:


without_mask.shape


# In[27]:


with_mask=with_mask.reshape(200,50*50*3)
without_mask=without_mask.reshape(200,50*50*3)


# In[28]:


without_mask.shape


# In[29]:


with_mask.shape


# In[30]:


x=np.r_[with_mask,without_mask]


# In[31]:


x.shape


# In[32]:


labels=np.zeros(x.shape[0])   #images with mask labled as zeros


# In[33]:


labels[200:]=1.0   #images without mask labled as ones


# In[34]:


names={0:'mask',1:'no mask'}


# In[35]:


#svm-support vector machine
#svc-support vector classifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# In[36]:


from sklearn.model_selection import train_test_split #this package is used to train and test the data set


# In[37]:


x_train,x_test,y_train,y_test=train_test_split(x,labels,test_size=0.25)


# In[38]:


x_train.shape


# In[39]:


#PCA-Principal component analysis
#from sklearn.decomposition import PCA


# In[40]:


#pca=PCA(n_components=3)
#x_train=pca.fit_transform(x_train)#converting data from one colunm of 750 to 3 colunms of 250 each


# In[41]:


x_train[0]


# In[42]:


x_train.shape


# In[43]:


svm=SVC()
svm.fit(x_train,y_train)


# In[44]:


y_pred=svm.predict(x_test)


# In[45]:


accuracy_score(y_test,y_pred)


# In[48]:


haar_data=cv2.CascadeClassifier("C:\\Users\\abc\\Desktop\\data.xml")
capture=cv2.VideoCapture(0)
data=[]
while True:
    flag,img=capture.read()
    if flag:
        faces=haar_data.detectMultiScale(img)
    for x,y,w,h in faces:
        #for making rectangle around the faces (img,x,y,w,h,(b,g,r),width of rectangle)
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,255), 4)
        face=img[y:y+h,x:x+w,:]
        face=cv2.resize(face,(50,50))
        face=face.reshape(1,-1)
        #face=pca.transform(face)
        pred=svm.predict(face)
        n=names[int(pred)]
        print(n)
    cv2.imshow('result',img)
    #27 is ASCII for escape
    if cv2.waitKey(2)==27:
        break
capture.release()
cv2.destroyAllWindows()


# In[ ]:




