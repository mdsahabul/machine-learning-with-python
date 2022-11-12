# machine-learning-with-python
# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
# get data files
!wget https://cdn.freecodecamp.org/project-data/books/book-crossings.zip

!unzip book-crossings.zip

books_filename = 'BX-Books.csv'
ratings_filename = 'BX-Book-Ratings.csv'
# import csv data into dataframes
df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
    ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})
# add your code here - consider creating a new cell for each section of code
# Calculate user and book rating counts
user_RatingCount = df_ratings.groupby('user')['rating'].count().reset_index().rename(columns = {'rating':'userTotalRatingCount'})
book_RatingCount = df_ratings.groupby('isbn')['rating'].count().reset_index().rename(columns = {'rating':'bookTotalRatingCount'})

# Add to df_ratings
df_ratings = df_ratings.merge(user_RatingCount,how='left', left_on='user', right_on='user')
df_ratings = df_ratings.merge(book_RatingCount, how='left', left_on='isbn', right_on='isbn')

# Filter data for statistical significance
df_ratings_2 =df_ratings.loc[(df_ratings['userTotalRatingCount']>=200) & (df_ratings['bookTotalRatingCount']>=100)]
# function to return recommended books - this will be tested
def get_recommends(book = ""):

  X = books_with_ratings_pivot[books_with_ratings_pivot.index == book]
  X = X.to_numpy().reshape(1,-1)
  distances, indices = model_knn.kneighbors(X,n_neighbors=8)
  recommended_books = []
  for x in reversed(range(1,6)):
      bookrecommended = [books_with_ratings_pivot.index[indices.flatten()[x]], distances.flatten()[x]]
      recommended_books.append(bookrecommended)
  recommended_books = [book, recommended_books]
  
  return recommended_books
books = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
print(books)
def test_book_recommendation():
  test_pass = True
  recommends = get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))")
  if recommends[0] != "Where the Heart Is (Oprah's Book Club (Paperback))":
    test_pass = False
  recommended_books = ["I'll Be Seeing You", 'The Weight of Water', 'The Surgeon', 'I Know This Much Is True']
  recommended_books_dist = [0.8, 0.77, 0.77, 0.77]
  for i in range(2): 
    if recommends[1][i][0] not in recommended_books:
      test_pass = False
    if abs(recommends[1][i][1] - recommended_books_dist[i]) >= 0.05:
      test_pass = False
  if test_pass:
    print("You passed the challenge! 🎉🎉🎉🎉🎉")
  else:
    print("You havn't passed yet. Keep trying!")

test_book_recommendation()
try:
  # This command only in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
# Get project files
!wget https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip

!unzip cats_and_dogs.zip

PATH = 'cats_and_dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150
import cv2
image_check = cv2.imread('cats_and_dogs/test/1.jpg')
print(image_check.shape)
#3
train_image_generator = ImageDataGenerator(rescale = 1.0/255.0)
validation_image_generator = ImageDataGenerator(rescale = 1.0/255.0)
test_image_generator = ImageDataGenerator(rescale = 1.0/255.0)

train_data_gen = train_image_generator.flow_from_directory('cats_and_dogs/train',
                                                           target_size=(IMG_WIDTH,IMG_HEIGHT),
                                                           batch_size=batch_size,
                                                           class_mode='binary')
val_data_gen = validation_image_generator.flow_from_directory('cats_and_dogs/validation',
                                                           target_size=(IMG_WIDTH,IMG_HEIGHT),
                                                           batch_size=batch_size,
                                                           class_mode='binary')
# For test images: without subdirectories flow_from_directory fails to recognize classes => there is a workaround for test
# See: https://kylewbanks.com/blog/loading-unlabeled-images-with-imagedatagenerator-flowfromdirectory-keras
test_data_gen = test_image_generator.flow_from_directory('cats_and_dogs',
                                                           classes=['test'],
                                                           target_size=(IMG_WIDTH,IMG_HEIGHT),
                                                           batch_size=batch_size,
                                                           class_mode='binary',
                                                           shuffle=False)
# 4
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])
# 5
train_image_generator = ImageDataGenerator(
                              rotation_range=50,
                              width_shift_range=0.2,
                              height_shift_range=0.2,
                              shear_range=0.25,
                              zoom_range=0.3,
                              horizontal_flip=True,
                              fill_mode='nearest',
                              rescale = 1.0/255.0)
# 6
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

# 8
history = None
# steps_per_epoch = TotalTrainingSamples / TrainingBatchSize
# validation_steps = TotalvalidationSamples / ValidationBatchSize
import math
history = model.fit(train_data_gen, 
                    epochs=epochs,
                    steps_per_epoch=math.ceil(2000/batch_size), 
                    validation_data=val_data_gen,
                    validation_steps=math.ceil(1000/batch_size))
# 9
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
prediction = model.predict(test_data_gen)
probabilities = [1 if pred > 0.5 else 0 for pred in prediction]

probabilities
test_images, _ = next(test_data_gen)
plotImages(test_images, probabilities=prediction.flatten())
# 11
answers =  [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 
            0, 0, 0, 0, 0, 0]

correct = 0

for probability, answer in zip(probabilities, answers):
  if round(probability) == answer:
    correct +=1

percentage_identified = (correct / len(answers))

passed_challenge = percentage_identified > 0.63

print(f"Your model correctly identified {round(percentage_identified, 2)*100}% of the images of cats and dogs.")

if passed_challenge:
  print("You passed the challenge!")
else:
  print("You haven't passed yet. Your model should identify at least 63% of the images. Keep trying. You will get it!")
 
