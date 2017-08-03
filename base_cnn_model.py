#coding:utf-8
# 从图片中直接训练一个小网络（作为基准方法）
# base model: 0.81
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Convolution2D,MaxPooling2D
from keras.layers import Activation,Dropout,Flatten,Dense
from keras.utils.visualize_util import plot

#  搭建基本模型
model=Sequential()
model.add(Convolution2D(32,3,3,input_shape=(3,150,150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64,3,3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

plot(model,to_file='cnn_model.png',show_shapes=True)

# 生成器 data augmentation
train_datagen=ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
	)

test_datagen=ImageDataGenerator(rescale=1./255)

train_generator=train_datagen.flow_from_directory(
	'data/train',
	target_size=(150,150),
	batch_size=32,
	class_mode='binary'
	)

validation_generator=test_datagen.flow_from_directory(
	'data/validation',
	target_size=(150,150),
	batch_size=32,
	class_mode='binary'
	)

model.fit_generator(
	train_generator,
	samples_per_epoch=2000,
	nb_epoch=50,
	validation_data=validation_generator,
	nb_val_samples=800
	)

model.save_weights('base_cnn_model.h5')

