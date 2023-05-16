#!/usr/bin/env python
# coding: utf-8

# In[1]:


import unet
import importlib
import ipyplot
import numpy as np
from keras.utils import array_to_img


# In[2]:


sample, mask = unet.import_data_sample(unet.TRAINING_DATA_DIRECTORY)


# In[3]:


array_to_img(sample)


# In[4]:


array_to_img(mask)


# In[5]:


unet.display_bounding_boxes(sample)


# In[ ]:


samples, masks = unet.load_and_crop_training_data_samples()

ipyplot.plot_images(samples)
ipyplot.plot_images([array_to_img(mask) for mask in masks])


# In[ ]:


unet.main()


# In[ ]:


get_ipython().system('jupyter nbconvert --to script unet_notebook.ipynb')


# In[ ]:




