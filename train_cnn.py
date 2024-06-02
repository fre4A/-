import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
from time import *

#创建函数加载图像数据集并创建训练集和验证集
def data_load(data_dir, img_height, img_width, batch_size):#定义函数data_load及四个参数分别为数据集所在目录路径/图像高/宽/批量
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(#从指定目录中加载训练数据集  tf....函数从目录中自动加载图像数据，并根据子目录的名称为每个类别分配标签
        data_dir,#从指定的 data_dir 目录中加载图像数据集，并将其作为 TensorFlow 数据集返回
        label_mode='categorical',#编码方式为独热编码
        validation_split=0.1,#将训练集的10%作为验证集
        subset="training",#训练集的子集
        seed=123,#确保每次运行时划分的一致性
        image_size=(img_height, img_width),#指定图像大小
        batch_size=batch_size)#指定批处理大小
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(#创建验证数据集
        data_dir,
        label_mode='categorical',
        validation_split=0.2,#将数据集的20%作为验证集
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    class_names = train_ds.class_names#获取训练数据集中的类别名称
    return train_ds, val_ds, class_names#返回训练数据集、验证数据集和类别名称

#定义卷积神经网络模型 
def model_load(IMG_SHAPE=(224, 224, 3), class_num=6):#函数定义，它接受两个参数：IMG_SHAPE（图像的形状，默认为 (224, 224, 3) 即高度为 224、宽度为 224、通道数为 3）和 class_num（类别数量，默认为 6）
    model = tf.keras.models.Sequential([#创建序列模型 按顺序堆叠层
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=IMG_SHAPE),#预处理层，用于将像素值缩放到 [0, 1] 范围内，以便更好地训练模型。1. / 255 是缩放的因子，input_shape=IMG_SHAPE 指定了输入图像的形状。
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),#卷积层，使用 32 个 3x3 的卷积核，激活函数为 ReLU
        tf.keras.layers.MaxPooling2D(2, 2),#最大池化层，用于对特征图进行降采样，减少参数数量并保留重要特征
        # Add another convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),#卷积层，使用 64 个 3x3 的卷积核，激活函数为 ReLU
        tf.keras.layers.MaxPooling2D(2, 2),#最大池化层，用于对特征图进行降采样
        tf.keras.layers.Flatten(),#扁平化层，用于将多维的输入数据转换成一维的数据，以便连接到全连接层
        # The same 128 dense layers, and 10 output layers as in the pre-convolution example:
        tf.keras.layers.Dense(128, activation='relu'),#全连接层，包含 128 个神经元，激活函数为 ReLU
        tf.keras.layers.Dense(class_num, activation='softmax')#输出层，包含 class_num 个神经元，使用 softmax 激活函数将输出转换成概率分布
    ])
    model.summary()#打印模型的摘要信息，包括每一层的名称、输出形状和参数数量等
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])#编译模型 指定了优化器、损失函数和评估指标 优化器选用 Adam，损失函数选用交叉熵损失函数（用于多分类问题），评估指标选用准确率
    return model#函数返回编译好的模型


# 展示训练过程的曲线 使用 Matplotlib 库来绘制两个子图，分别表示准确率和损失值的变化情况

#在第一个子图中，绘制了训练准确率和验证准确率随着时期的变化曲线，并加上了图例、坐标轴标签和标题

#在第二个子图中，绘制了训练损失值和验证损失值随着时期的变化曲线，并同样加上了图例、坐标轴标签和标题

def show_loss_acc(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()), 1])
    plt.title('Training and Validation Accuracy')

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.savefig('results_cnn.png', dpi=100)

#训练函数 加载数据集、创建模型、进行训练并保存模型，然后打印训练时间，并调用之前定义的 show_loss_acc(history) 函数来展示训练过程中的损失和准确率变化
def train(epochs):
    begin_time = time()
    train_ds, val_ds, class_names = data_load("D:\Trash\Trash_jpg", 224, 224, 16)#调用 data_load 函数加载训练和验证数据集，以及类别名称
    print(class_names)
    model = model_load(class_num=len(class_names))#调用 model_load 函数创建模型，其中 class_num 参数是类别的数量
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)#使用加载的数据集对模型进行训练，训练的时期数由 epochs 参数指定，并将训练历史保存在 history 变量中
    model.save("models/cnn_245_epoch30.h5")#将训练好的模型保存在指定路径下，文件名为 "cnn_245_epoch30.h5"
    end_time = time()#训练结束 停止计时
    run_time = end_time - begin_time#计算训练时间
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    show_loss_acc(history)#调用 show_loss_acc 函数展示训练过程中的损失和准确率变化



if __name__ == '__main__':
    train(epochs=30)
