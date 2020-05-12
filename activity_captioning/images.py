#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pickle
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.models import Model, load_model
from keras.preprocessing.sequence import pad_sequences


# In[2]:


model = load_model("./models/model_14.h5")


# In[3]:


model_temp = ResNet50(weights="imagenet", input_shape=(224,224,3))


# In[4]:


model_resnet = Model(model_temp.input,model_temp.layers[-2].output)


# In[5]:


def preprocess_image(img):
    img = image.load_img(img, target_size=(224,224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img


# In[6]:


def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape((1,feature_vector.shape[1]))
    return feature_vector


# In[7]:





# In[8]:





# In[10]:


with open("word_to_idx.pkl","rb") as f:
    word_to_idx = pickle.load(f)
    
with open("idx_to_word.pkl","rb") as f:
    idx_to_word = pickle.load(f)


# In[11]:


def predict_caption(photo):
    in_text = "startseq"
    max_len = 35
    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len, padding='post')

        ypred =  model.predict([photo,sequence])
        ypred = ypred.argmax()
        word = idx_to_word[ypred]
        in_text+= ' ' +word
        
        if word =='endseq':
            break
        
        
    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)
    
    return final_caption


# In[12]:





# In[16]:


def caption_this_image(image):
    encode = encode_image(image)
    caption = predict_caption(encode)
    return caption


# In[17]:





# In[18]:





# In[ ]:




