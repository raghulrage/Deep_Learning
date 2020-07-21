import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Dropout, Conv2D, MaxPooling2D
import pickle
import time 
from tensorflow.keras.callbacks import TensorBoard

Name = 'Cats-vd-dogs-cnn-64x2-{}'.format(time.time())

tensorboard = TensorBoard(log_dir = 'logs/{}'.format(Name))

gpu_option = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction = 0.333)

sess = tf.compat.v1.Session(config = tf.compat.v1.ConfigProto(gpu_options=gpu_option))

X = pickle.load(open('X.pickle','rb'))

y = pickle.load(open('y.pickle','rb'))

X = X/255.0

model = Sequential()

model.add(Conv2D(64,(3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

model.fit(X, y, batch_size = 32, epochs = 20, validation_split = 0.3, callbacks = [tensorboard])