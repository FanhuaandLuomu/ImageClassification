#coding:utf-8
# fine-tune预训练网络(vgg+全连接sigmoid)的高层
# val_acc:0.95+
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
import cPickle
from keras.utils.visualize_util import plot
from keras.layers import Input

# 搭建vgg16模型
def VGG_16(weights_path=None):
	model = Sequential()
	model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(64, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(128, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(256, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(ZeroPadding2D((1,1)))
	model.add(Convolution2D(512, 3, 3, activation='relu'))
	model.add(MaxPooling2D((2,2), strides=(2,2)))

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

# 顶层分类层
# top_model=Sequential()
# top_model.add(Dense(256,input_shape=(512*7*7,)))
# top_model.add(Dropout(0.5))
# top_model.add(Dense(1,activation='sigmoid'))

# 载入预训练权重
model=VGG_16('G://keras_weights/vgg16_weights.h5')

# 取vgg16的卷积部分  （去除全连接层）
model2=Model(model.layers[0].input,model.layers[31].output)

# 顶层分类层
input1=Input((512*7*7,),name='input1')
dense1=Dense(256)(input1)
dropout1=Dropout(0.5)(dense1)
output1=Dense(1,activation='sigmoid',name='output')(dropout1)

top_model=Model(input1,output1)

# load weights
top_model.load_weights('G://kaggle/bottleneck_fc_model.h5')

# 将两个模型合并
input=model2.input   # 输入
output=top_model(model2.output)  # 输出

model = Model(input, output)

plot(model,to_file='vgg16_fine_tune.png',show_shapes=True)

# 冻结前几层  最后一个卷积模块不冻结
for layer in model.layers[:25]:
	layer.trainable=False

model.compile(loss='binary_crossentropy',
				optimizer=SGD(lr=1e-4,momentum=0.9),
				metrics=['accuracy']
				)

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
	target_size=(224,224),
	batch_size=8,
	class_mode='binary'
	)

validation_generator=test_datagen.flow_from_directory(
	'data/validation',
	target_size=(224,224),
	batch_size=8,
	class_mode='binary'
	)

model.fit_generator(
	train_generator,
	samples_per_epoch=2000,
	nb_epoch=50,
	validation_data=validation_generator,
	nb_val_samples=800
	)










