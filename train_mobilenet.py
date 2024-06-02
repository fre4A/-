import tensorflow as tf
import matplotlib.pyplot as plt
from time import *


# 加载数据集
def data_load(data_dir, img_height, img_width, batch_size):
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.1,
        subset="training",
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        label_mode='categorical',
        validation_split=0.2,
        subset="validation",
        seed=123,
        color_mode="rgb",
        image_size=(img_height, img_width),
        batch_size=batch_size)

    class_names = train_ds.class_names

    return train_ds, val_ds, class_names


# 模型加载，指定图片处理的大小和是否进行迁移学习
#这个模型使用 MobileNetV2 作为基础模型，通过添加全局平均池化层和全连接层来进行分类任务 在微调过程中，基础模型的权重被固定，只有顶部的全连接层会被训练
def model_load(IMG_SHAPE=(224, 224, 3), class_num=245):
    # 微调的过程中不需要进行归一化的处理
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                   include_top=False,
                                                   weights='imagenet')#使用 MobileNetV2 架构创建一个基础模型。参数 input_shape 指定输入图像的形状，include_top=False 表示不包含顶部的全连接层，weights='imagenet' 表示使用 ImageNet 数据集预训练的权重
    base_model.trainable = False#将基础模型的权重设为不可训练，即在微调过程中保持固定
    model = tf.keras.models.Sequential([
        tf.keras.layers.experimental.preprocessing.Rescaling(1. / 127.5, offset=-1, input_shape=IMG_SHAPE),#添加一个归一化层，将输入图像的像素值从 [0,255] 缩放到 [-1,1] 的范围
        base_model,#将 MobileNetV2 架构作为模型的一部分添加到顺序模型中
        tf.keras.layers.GlobalAveragePooling2D(),#全局平均池化层，用于将卷积特征图转换为一维向量
        tf.keras.layers.Dense(class_num, activation='softmax')#全连接层，输出大小为 class_num，使用 softmax 激活函数进行分类
    ])
    model.summary()#打印模型的结构信息，包括每一层的名称、输出形状和参数数量
    # 模型训练
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model#译模型，指定优化器为 Adam，损失函数为分类交叉熵，评估指标为准确率


# 展示训练过程的曲线
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
    plt.savefig('results_mobilenet_epoch30.png', dpi=100)


def train(epochs):
    begin_time = time()
    train_ds, val_ds, class_names = data_load("F:/datas/tmp/data/tttt/trash_jpg", 224, 224, 16)
    print(class_names)
    model = model_load(class_num=len(class_names))
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs)
    model.save("models/mobilenet_245_epoch30.h5")
    end_time = time()
    run_time = end_time - begin_time
    print('该循环程序运行时间：', run_time, "s")  # 该循环程序运行时间： 1.4201874732
    show_loss_acc(history)


if __name__ == '__main__':
    train(epochs=30)
