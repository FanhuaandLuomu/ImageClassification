# ImageClassification
原文：http://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
参考上述面向小数据集的图像分类方法（kaggle的猫狗分类），重新实现了可运行的py程序，对一些细节做了改动。
# base model
从图片中直接训练一个小网络（作为基准方法）
va_acc：0.81
# vgg16提取特征+训练全连接分类
利用预训练网络vgg16的bottleneck（瓶颈）特征,输入至全连接网络，sigmoid分类
val_acc:0.90
# fine_tune vgg16
fine-tune预训练网络(vgg+全连接sigmoid)的高层,冻结网络的前几层（卷积特征提取层）
val_acc:0.95+

# 语料下载
为方便操作，语料我已整理出实验所需格式，
其中train:1000+1000, validation:400+400
百度网盘下载：http://pan.baidu.com/s/1qY6tc4G
