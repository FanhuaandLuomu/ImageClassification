#coding:utf-8
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

img=load_img('cat.8.jpg')
print img

x=img_to_array(img)
print x.shape
x=x.reshape((1,)+x.shape)
print x.shape
print x

datagen=ImageDataGenerator(
	rotation_range=40,   # 0~180 随机旋转
	width_shift_range=0.2,  # 水平 竖直方向 移动程度
	height_shift_range=0.2,
	shear_range=0.2,   # 剪切变换程度
	zoom_range=0.2,     # 随机放大
	horizontal_flip=True,   # 水平翻转
	fill_mode='nearest'    # 像素填充
	)

i=0
for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',\
					save_prefix='cat',save_format='png'):
	i+=1
	if i>20:
		break