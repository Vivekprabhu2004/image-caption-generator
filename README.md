# image-caption-generator
A model which can generate captions to uploaded images
Image Caption Generator using InceptionV3 and LSTM

This project demonstrates an image caption generator using deep learning techniques. It utilizes InceptionV3 for image feature extraction and LSTM networks for caption generation. The model is trained on the Flickr8k dataset, and it generates captions for new images using a greedy search algorithm.  

---

 Project Structure


├── data/  
│   ├── archive.zip              # Dataset containing images and captions  
│   ├── Flickr_8k.trainImages.txt  # Training image list  
│   ├── Flickr_8k.testImages.txt   # Test image list  
│   ├── glove.6B.200d.txt        # Pre-trained GloVe embeddings  
│   └── model_checkpoint.h5      # Checkpoint for the best model during training  
├── image_caption_generator.py     # Main code (in Python)  
└── README.md                    # Documentation (this file)  


---

 Requirements

Make sure you have the following installed:  
- Python 3.x  
- TensorFlow  
- Keras  
- NumPy  
- pandas  
- OpenCV (for image handling)  
- Matplotlib (for visualizations)  

---

 How to Use

# 1. Dataset Setup
1. Upload the Dataset: Place the Flickr8k dataset (images + captions) in your Google Drive.
2. Modify Paths: Update paths in the code to point to your dataset and other required files.

# 2. Import Necessary Libraries
Ensure the following libraries are imported:
python
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import os
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, LSTM, Embedding, Dense, Dropout, add
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.callbacks import ModelCheckpoint
from google.colab import drive
from google.colab.patches import cv2_imshow


# 3. Mount Google Drive
python
drive.mount('/content/drive')

This will allow you to access files stored on your Drive.  

---

 Model Workflow

1. Image Preprocessing:  
   - Uses InceptionV3 to extract features from images.  
   - Preprocesses images to 299x299 pixels for compatibility with InceptionV3.  

2. Text Preprocessing:  
   - Loads and cleans captions by removing punctuation and converting to lowercase.  
   - Appends 'startseq' and 'endseq' tokens to each caption for LSTM training.

3. Splitting Data:  
   - Randomly splits the dataset (98% for training, 2% for testing).  

4. Tokenization and Word Embeddings:  
   - Tokenizes the captions to convert text into sequences of integers.  
   - Uses GloVe embeddings for better semantic representation.  

5. Model Architecture:  
   - Combines image features (from InceptionV3) with LSTM-generated captions.  
   - Uses Dense layers for better predictions and outputs vocabulary-sized softmax.

6. Training the Model:  
   - Uses categorical cross-entropy as the loss function.  
   - Saves the best model based on loss using 'ModelCheckpoint'.  

---

 Model Training

1. Compile the Model:
    python
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    

2. Train the Model:
    python
    model.fit([X1, X2], y, epochs=50, batch_size=256, callbacks=[checkpoint])
    

---

 Inference (Generating Captions)

1. Encode the Image:
    python
    encoded_img = encode('/path/to/image.jpg', encoding_model)
    encoded_img = np.reshape(encoded_img, (1, 2048))
    

2. Generate Caption:
    python
    caption = greedy_search(encoded_img)
    print("Generated Caption:", caption)
    

3. Display the Image with Caption:
    python
    image = cv2.imread('/path/to/image.jpg')
    cv2_imshow(image)
    

 Notes

1. Links to Dataset and Models:  
   - Add your own links to datasets or model files wherever required.
   - Example: Replace '/path/to/your/image.jpg' with appropriate paths.

2. Error Handling:  
   - Ensure that GloVe embeddings are correctly loaded to avoid shape mismatches.
   - If using custom images, make sure they are resized to '(299, 299)'.

---

 References

- [Keras Documentation](https://keras.io)
- [InceptionV3 Model](https://keras.io/api/applications/inceptionv3/)
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)

---



---
