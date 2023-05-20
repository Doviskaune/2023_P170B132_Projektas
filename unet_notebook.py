#!/usr/bin/env python
# coding: utf-8

# In[2]:


import unet
import importlib
# !pip install ipyplot
import ipyplot
import numpy as np
from keras.utils import array_to_img
import tensorflow as tf
import matplotlib.pyplot as plt


# In[31]:


importlib.reload(unet)

sample, mask = unet.load_data_samples(unet.VALIDATION_DATA_DIRECTORY)


# In[32]:


ipyplot.plot_images(sample)


# In[ ]:


array_to_img(mask)


# In[13]:


unet.display_bounding_boxes(sample)


# In[ ]:


samples, masks = unet.load_and_crop_training_data_samples()

ipyplot.plot_images(samples)
ipyplot.plot_images([array_to_img(mask) for mask in masks])


# In[10]:


# importlib.reload(unet)

samples, masks = unet.load_and_crop_training_data_samples()
validation_samples, validation_masks = unet.load_and_crop_validation_data_samples()
samples, masks = unet.create_rotated_data_samples(samples, masks)
validation_samples, validation_masks = unet.create_rotated_data_samples(validation_samples, validation_masks)


# In[ ]:


print(validation_samples, validation_samples)


# In[11]:


importlib.reload(unet)

# tf.keras.backend.clear_session()
unet_model = unet.prepare_unet_model(weight=3)

model_history = unet.train(unet_model, samples, masks, validation_samples, validation_masks, batch_size=16, epochs=10)


# In[ ]:


# importlib.reload(unet)
#  1:3 weights

training_samples, training_masks = unet.load_data_samples(unet.TRAINING_DATA_DIRECTORY)

unet.show_full_image_predictions(unet_model, training_samples, training_masks)


# In[ ]:


# importlib.reload(unet)
#  1:3 weights

validation_samples, validation_masks = unet.load_data_samples(unet.VALIDATION_DATA_DIRECTORY)

unet.show_full_image_predictions(unet_model, validation_samples, validation_masks)


# In[6]:


# importlib.reload(unet)
#  1:5 weights

training_samples, training_masks = unet.load_data_samples(unet.TRAINING_DATA_DIRECTORY)

unet.show_full_image_predictions(unet_model, training_samples, training_masks)


# In[7]:


# importlib.reload(unet)
#  1:5 weights

validation_samples, validation_masks = unet.load_data_samples(unet.VALIDATION_DATA_DIRECTORY)

unet.show_full_image_predictions(unet_model, validation_samples, validation_masks)


# In[106]:


importlib.reload(unet)

training_samples, training_masks = unet.load_data_samples(unet.TRAINING_DATA_DIRECTORY)

unet.show_full_image_predictions(unet_model, training_samples, training_masks)


# In[107]:


importlib.reload(unet)

validation_samples, validation_masks = unet.load_data_samples(unet.VALIDATION_DATA_DIRECTORY)

unet.show_full_image_predictions(unet_model, validation_samples, validation_masks)


# In[ ]:


print(len(validation_samples))
# index = 870

for index in (150, 350, 870, 880, 910):
    unet.show_prediction(unet_model, validation_samples[index], validation_masks[index])


# In[ ]:


unet.display_bounding_boxes(sample)


# In[56]:


box_top_left_corners = unet.get_bounding_boxes_start_point(sample)

print(len(np.unique(box_top_left_corners[0])), len(np.unique(box_top_left_corners[1])))


# In[100]:


importlib.reload(unet)

sample, mask = unet.load_data_sample(unet.VALIDATION_DATA_DIRECTORY, 1)

unet.show_full_image_prediction(unet_model, sample, mask)


# In[67]:


importlib.reload(unet)

sample, mask = unet.load_data_sample(unet.VALIDATION_DATA_DIRECTORY, 1)

columns, rows = unet.calculate_number_of_columns_and_rows(sample)

samples_cropped, masks_cropped = unet.pad_and_crop_image(sample), unet.pad_and_crop_image(mask)

masks_predicted = []
for sample in samples_cropped:
    pred_mask = unet_model.predict(sample.numpy().reshape((1, unet.INPUT_SIZE, unet.INPUT_SIZE, 3)))
    pred_mask = tf.image.crop_to_bounding_box(pred_mask, unet.INPUT_MARGIN_SIZE, unet.INPUT_MARGIN_SIZE, unet.INPUT_CONTENT_SIZE, unet.INPUT_CONTENT_SIZE)
    masks_predicted.append(pred_mask.numpy())



# In[96]:


print(columns, rows)

print(np.array(masks_predicted).shape)

c_list = []

for i in range(rows):
    c = np.concatenate(masks_predicted[i * columns:(i + 1) * columns], axis=2)
    c_list.append(c)

c = np.concatenate(c_list, axis=1)
print(c[0].shape)

array_to_img(unet.create_mask(c))


# In[97]:


array_to_img(mask)


# In[85]:


plt.imshow(array_to_img(unet.create_mask(masks_predicted[3])))
plt.show()

array_to_img(unet.create_mask(masks_predicted[3]))


# In[ ]:


importlib.reload(unet)

# import tensorflow as tf

# tf.keras.backend.clear_session()

samples, masks = unet.load_and_crop_training_data_samples()
validation_samples, validation_masks = unet.load_and_crop_validation_data_samples()
samples, masks = unet.create_rotated_data_samples(samples, masks)
validation_samples, validation_masks = unet.create_rotated_data_samples(validation_samples, validation_masks)

unet_model = unet.prepare_unet_model()

model_history = unet.train(unet_model, samples, masks, validation_samples, validation_masks, batch_size=32, epochs=20)

unet.save_model(unet_model)

unet.plot_training_progress(model_history)


# In[ ]:


get_ipython().system('jupyter nbconvert --to script unet_notebook.ipynb')


# In[ ]:




