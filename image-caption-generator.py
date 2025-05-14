

# Import necessary modules for Google Drive integration and mounting the drive
from google.colab import drive
drive.mount('/content/drive')

# Unzipping the dataset into the specified directory
!unzip '#PASTE YOUR DATA SET PATH HERE#' -d '/content/drive/My Drive/data'

# Import necessary libraries for data handling, neural networks, and visualization
import numpy as np
import pandas as pd
import os
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.models import Model
from keras.layers import Flatten, Dense, LSTM, Dropout, Embedding, Activation
from keras.layers import concatenate, BatchNormalization, Input, add
from keras.utils import to_categorical, plot_model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import matplotlib.pyplot as plt
import cv2

# Function to load image descriptions from text file and map them by image ID
def load_description(text):
    mapping = dict()
    for line in text.split("\n"):
        token = line.split(",")
        if len(line) < 2:  # Skip empty or incomplete lines
            continue
        img_id = token[0].split('.')[0]  # Extract image ID without extension
        img_des = token[1]  # Extract description
        if img_id not in mapping:
            mapping[img_id] = list()
        mapping[img_id].append(img_des)
    return mapping

# Load the captions text file and process it into descriptions dictionary
token_path = '/content/drive/MyDrive/data/captions.txt'
text = open(token_path, 'r', encoding='utf-8').read()
descriptions = load_description(text)
print(descriptions['1000268201_693b08cb0e'])  # Display a sample description

# Clean the text by removing punctuation, lowering case, and filtering out small words
import string

def clean_description(desc):
    for key, des_list in desc.items():
        for i in range(len(des_list)):
            caption = des_list[i]
            caption = [ch for ch in caption if ch not in string.punctuation]
            caption = ''.join(caption)
            caption = caption.split(' ')
            caption = [word.lower() for word in caption if len(word) > 1 and word.isalpha()]
            caption = ' '.join(caption)
            des_list[i] = caption

clean_description(descriptions)  # Apply cleaning function
descriptions['1000268201_693b08cb0e']  # Verify cleaned text

# Generate a vocabulary of unique words from descriptions
def to_vocab(desc):
    words = set()
    for key in desc.keys():
        for line in desc[key]:
            words.update(line.split())
    return words

vocab = to_vocab(descriptions)  # Store all unique words as vocabulary

# Shuffle image IDs and split them into train and test datasets
import random
all_img_ids = list(descriptions.keys())
random.shuffle(all_img_ids)

split_ratio = 0.98  # Use 98% data for training, 2% for testing
split_index = int(len(all_img_ids) * split_ratio)

train_img_ids = all_img_ids[:split_index]
test_img_ids = all_img_ids[split_index:]

# Save train and test image IDs to text files
train_images_file = '/content/drive/MyDrive/data/Flickr_8k.trainImages.txt'
test_images_file = '/content/drive/MyDrive/data/Flickr_8k.testImages.txt'

with open(train_images_file, 'w') as file:
    for img_id in train_img_ids:
        file.write(f"{img_id}.jpg\n")

with open(test_images_file, 'w') as file:
    for img_id in test_img_ids:
        file.write(f"{img_id}.jpg\n")

# Load all images and filter out those belonging to the training set
import glob
images = '/content/drive/MyDrive/data/Images/'
img = glob.glob(images + '*.jpg')

train_path = '/content/drive/MyDrive/data/Flickr_8k.trainImages.txt'
train_images = open(train_path, 'r', encoding='utf-8').read().split("\n")
train_img = [im for im in img if im[len(images):] in train_images]

# Function to add 'startseq' and 'endseq' markers to captions
def load_clean_descriptions(des, dataset):
    dataset_des = dict()
    for key, des_list in des.items():
        if key + '.jpg' in dataset:
            if key not in dataset_des:
                dataset_des[key] = list()
            for line in des_list:
                desc = 'startseq ' + line + ' endseq'
                dataset_des[key].append(desc)
    return dataset_des

train_descriptions = load_clean_descriptions(descriptions, train_images)
print(train_descriptions['1000268201_693b08cb0e'])  # Display processed captions

# Image preprocessing using InceptionV3
from keras.preprocessing.image import load_img, img_to_array

def preprocess_img(img_path):
    img = load_img(img_path, target_size=(299, 299))  # Resize to 299x299
    x = img_to_array(img)  # Convert to array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = preprocess_input(x)  # Normalize input
    return x

# Encode image features using InceptionV3
base_model = InceptionV3(weights='imagenet')
model = Model(base_model.input, base_model.layers[-2].output)

def encode(image):
    image = preprocess_img(image)
    vec = model.predict(image)
    vec = np.reshape(vec, (vec.shape[1]))
    return vec

# Encode all training images
encoding_train = {img[len(images):]: encode(img) for img in train_img}

# Process captions into input-output sequences for the model
all_train_captions = [cap for val in train_descriptions.values() for cap in val]

# Create a word-to-index and index-to-word mapping for the vocabulary
ixtoword, wordtoix = {}, {}
ix = 1  # Start indexing from 1
for word in vocab:
    wordtoix[word] = ix
    ixtoword[ix] = word
    ix += 1

# Model building: Combining image and text features
from keras.callbacks import ModelCheckpoint

ip1 = Input(shape=(2048,))
fe1 = Dropout(0.2)(ip1)
fe2 = Dense(256, activation='relu')(fe1)

ip2 = Input(shape=(max_length,))
se1 = Embedding(vocab_size, emb_dim, mask_zero=True)(ip2)
se2 = Dropout(0.2)(se1)
se3 = LSTM(256)(se2)

decoder1 = add([fe2, se3])
decoder2 = Dense(256, activation='relu')(decoder1)
outputs = Dense(vocab_size, activation='softmax')(decoder2)

model = Model(inputs=[ip1, ip2], outputs=outputs)

# Compile and train the model
checkpoint = ModelCheckpoint('/content/drive/MyDrive/data/model_checkpoint.h5', monitor='loss', save_best_only=True)
model.compile(loss='categorical_crossentropy', optimizer='adam')

model.fit([X1, X2], y, epochs=50, batch_size=256, callbacks=[checkpoint])

# Save the trained model using pickle
import pickle
with open('/content/drive/MyDrive/data/model.pkl', 'wb') as file:
    pickle.dump(model, file)

# Function for generating captions using greedy search
def greedy_search(pic):
    start = 'startseq'
    for _ in range(max_length):
        seq = [wordtoix[word] for word in start.split() if word in wordtoix]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = model.predict([pic, seq], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        start += ' ' + word
        if word == 'endseq':
            break
    return ' '.join(start.split()[1:-1])

# Displaying an image and generating its caption
from google.colab.patches import cv2_imshow
new_image_path = '#PASTE YOUR IMAGE PATH HERE#'
encoded_img = np.reshape(encode(new_image_path), (1, 2048))
caption = greedy_search(encoded_img)
print("Generated Caption:", caption)

image = cv2.imread(new_image_path)
cv2_imshow(image)
cv2.waitKey(0)
cv2.destroyAllWindows()
