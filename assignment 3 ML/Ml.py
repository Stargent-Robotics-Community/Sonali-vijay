import tensorflow.keras
from PIL import Image, ImageOps
import numpy as np

#Disable scientific notation for clarity
np.set_printoptions(supress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5')

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1.
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image
image1 = Image.open('ideas\Chhota Bheem\9.jpg')
image2 = Image.open('ideas\Doraemon\9.jpg')
image3 = Image.open('ideas\Jerry\9.png')


#resize the image to a 224x224 with the same strategy as in TM2:
#resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image1 = ImageOps.fit(image1, size, Image.ANTIALIAS)
image2 = ImageOps.fit(image2, size, Image.ANTIALIAS)
image3 = ImageOps.fit(image3, size, Image.ANTIALIAS)


#turn the image into a numpy array
image_array1 = np.asarray(image1)
image_array2 = np.asarray(image2)
image_array3 = np.asarray(image3)


#display the resized image
image1.show()
image2.show()
image3.show()


# Normalize the image
normalized_image_array1 = (image_array1.astype(np.float32) / 127.0) - 1
normalized_image_array2 = (image_array2.astype(np.float32) / 127.0) - 1
normalized_image_array3 = (image_array3.astype(np.float32) / 127.0) - 1


# Load the image into the array
data[0] = normalized_image_array1
data[1] = normalized_image_array2
data[2] = normalized_image_array3

# run the inference
prediction = model.predict(data)

#print prediction
print("*"*100)
print("\nThe Result of prediction is:\n")
i=np.argmax(prediction[0],axis=0)
#print(i)
print("The first image is of "+labels[i])
j=np.argmax(prediction[1],axis=0)
#print(j)
print("The first image is of "+labels[j])
k=np.argmax(prediction[2],axis=0)
#print(k)
print("The first image is of "+labels[k])


