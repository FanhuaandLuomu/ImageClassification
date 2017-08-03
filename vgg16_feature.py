#coding:utf-8
# 利用预训练网络vgg16的bottleneck（瓶颈）特征
# 0.89
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import Model
import numpy as np
import cPickle
from keras.utils.visualize_util import plot

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
	# (512,7,7)

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(1000, activation='softmax'))

	if weights_path:
		model.load_weights(weights_path)

	return model

# 载入预训练权重
model=VGG_16('G://keras_weights/vgg16_weights.h5')
# print dir(model)
print model.summary()
plot(model,to_file='vgg16.png',show_shapes=True)

# 取vgg16的卷积部分  （去除全连接层）
model2=Model(model.layers[0].input,model.layers[31].output)
print model2.summary()
plot(model2,to_file='vgg16_cnn.png',show_shapes=True)

# data augmentation
datagen=ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True
	)

# 生成器  每次yield （1,3,224,224） 若batch_size=32 会内存溢出
generator=datagen.flow_from_directory(
	'data/train',
	target_size=(224,224),
	batch_size=1,
	class_mode=None,
	shuffle=False
	)

# im=generator.next()
# print im.shape

# 特征提取模块
# train features
bottleneck_features_train=model2.predict_generator(generator,2000)
print bottleneck_features_train.shape
# np.save(open('bottleneck_features_train.npy','w'),bottleneck_features_train)
cPickle.dump(bottleneck_features_train,open('bottleneck_features_train.pkl','w'))

generator=datagen.flow_from_directory(
	'data/validation',
	target_size=(224,224),
	batch_size=1,
	class_mode=None,
	shuffle=False
	)

# validation features
bottleneck_features_validation=model2.predict_generator(generator,800)
# np.save(open('bottleneck_features_validation.npy','w'),bottleneck_features_validation)
cPickle.dump(bottleneck_features_validation,open('bottleneck_features_validation.pkl','w'))


# 训练全连接神经网络分类
# train_data=np.load(open('bottleneck_features_train.npy'))
train_data=cPickle.load(open('bottleneck_features_train.pkl'))
print train_data.shape
train_labels=np.array([0]*1000+[1]*1000)

# validation_data=np.load(open('bottleneck_features_validation.npy'))
validation_data=cPickle.load(open('bottleneck_features_validation.pkl'))
print validation_data.shape
validation_labels=np.array([0]*400+[1]*400)

model=Sequential()
# 与官网有点区别，Flatten层我加载特征提取那儿了
model.add(Dense(256,input_shape=train_data.shape[1:],activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1,activation='sigmoid'))

model.compile(optimizer='adam',
			loss='binary_crossentropy',
			metrics=['accuracy']
			)
model.fit(train_data,train_labels,
		nb_epoch=50,
		batch_size=32,
		validation_data=(validation_data,validation_labels)
		)

# save weights
model.save_weights('bottleneck_fc_model.h5')