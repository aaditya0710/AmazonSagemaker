#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sagemaker
import boto3 #to read dataset from s3 bucket
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.session import s3_input,Session


# In[4]:


bucket_name = 'bankapplicationagain'#change this variable to a unique name for your bucket
my_region = boto3.session.Session().region_name#set the region of the instance
print(my_region)


# In[5]:


s3 = boto3.resource('s3') #get access of s3 bucket
try:
    if my_region == 'us-east-1':
        s3.create_bucket(Bucket=bucket_name)
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


# In[6]:


prefix = 'xgboost-as-a-built-in-algo'
output_path = 's3://{}/{}/output'.format(bucket_name,prefix)
print(output_path)


# In[9]:


import pandas as pd
import urllib
try:
    urllib.request.urlretrieve("https://d1.awsstatic.com/tmt/build-train-deploy-machine-learning-model-sagemaker/bank_clean.27f01fbbdf43271788427f3682996ae29ceca05d.csv", "bank_clean.csv")
    print("success: download bank_clean.csv")
except Exception as e:
    print("data load error",e)
try:
    model_data = pd.read_csv("./bank_clean.csv",index_col=0)
    print("success: data loaded into dataframe")
except Exception as e:
    print('Data load error',e)


# In[11]:


import numpy as np
train_data,test_data = np.split(model_data.sample(frac=1,random_state=1729),[int(0.7*len(model_data))])
print(train_data.shape,test_data.shape)


# In[13]:


import os
pd.concat([train_data['y_yes'], train_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('train.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,'train/train.csv')).upload_file('train.csv')
s3_input_train = sagemaker.s3_input(s3_data='s3://{}/{}/train'.format(bucket_name,prefix),content_type='csv')


# In[14]:


pd.concat([test_data['y_yes'], test_data.drop(['y_no', 'y_yes'], axis=1)], axis=1).to_csv('test.csv', index=False, header=False)
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix,'test/test.csv')).upload_file('test.csv')
s3_input_test = sagemaker.s3_input(s3_data='s3://{}/{}/test'.format(bucket_name,prefix),content_type='csv')


# In[16]:


container = get_image_uri(boto3.Session().region_name,'xgboost',repo_version='1.0-1')


# In[17]:


hyperparameters = {
        "max_depth":"5",
        "eta":"0.2",
        "gamma":"4",
        "min_child_weight":"6",
        "subsample":"0.7",
        "objective":"binary:logistic",
        "num_round":50
        }


# In[18]:


estimator = sagemaker.estimator.Estimator(image_name=container, 
                                          hyperparameters=hyperparameters,
                                          role=sagemaker.get_execution_role(),
                                          train_instance_count=1, 
                                          train_instance_type='ml.m5.2xlarge', 
                                          train_volume_size=5, # 5 GB 
                                          output_path=output_path,
                                          train_use_spot_instances=True,
                                          train_max_run=300,
                                          train_max_wait=600)


# In[19]:


estimator.fit({'train': s3_input_train,'validation': s3_input_test})


# In[20]:


xgb_predictor = estimator.deploy(initial_instance_count=1,instance_type='ml.m4.xlarge')


# In[21]:


from sagemaker.predictor import csv_serializer
test_data_array = test_data.drop(['y_no', 'y_yes'], axis=1).values #load the data into an array
xgb_predictor.content_type = 'text/csv' # set the data type for an inference
xgb_predictor.serializer = csv_serializer # set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8') # predict!
predictions_array = np.fromstring(predictions[1:], sep=',') # and turn the prediction into an array
print(predictions_array.shape)


# In[22]:


predictions_array


# In[23]:



cm = pd.crosstab(index=test_data['y_yes'], columns=np.round(predictions_array), rownames=['Observed'], colnames=['Predicted'])
tn = cm.iloc[0,0]; fn = cm.iloc[1,0]; tp = cm.iloc[1,1]; fp = cm.iloc[0,1]; p = (tp+tn)/(tp+tn+fp+fn)*100
print("\n{0:<20}{1:<4.1f}%\n".format("Overall Classification Rate: ", p))
print("{0:<15}{1:<15}{2:>8}".format("Predicted", "No Purchase", "Purchase"))
print("Observed")
print("{0:<15}{1:<2.0f}% ({2:<}){3:>6.0f}% ({4:<})".format("No Purchase", tn/(tn+fn)*100,tn, fp/(tp+fp)*100, fp))
print("{0:<16}{1:<1.0f}% ({2:<}){3:>7.0f}% ({4:<}) \n".format("Purchase", fn/(tn+fn)*100,fn, tp/(tp+fp)*100, tp))


# In[ ]:




