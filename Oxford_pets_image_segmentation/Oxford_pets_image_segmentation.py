# -*- coding:utf-8 -*-
"""
 @Author   : Kai
 @Time     : 2021/6/4 12:32
 @Email    : 1240498481@qq.com
 @FileName : Oxford_pets_image_segmentation.py
 @Software : PyCharm

 在牛津宠物数据集上从头开始训练的图像分割模型。
"""

# 导包
import os
from IPython.display import Image, display
from tensorflow.keras.preprocessing.image import load_img
import PIL
from PIL import ImageOps
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
import random

# 下载数据
# os.system('curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz')
# os.system('curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz')
# os.system('tar -xf images.tar.gz')
# os.system('tar -xf annotations.tar.gz')

# 准备输入图像和目标分割掩码的路径
input_dir = 'images/'
target_dir = 'annotations/trimaps/'
img_size = (160, 160)
num_classes = 3
batch_size = 32

input_img_paths = sorted(
    [
        os.path.join(input_dir, fname)
        for fname in os.listdir(input_dir)
        if fname.endswith('.jpg')
    ]
)

target_img_paths = sorted(
    [
        os.path.join(target_dir, fname)
        for fname in os.listdir(target_dir)
        if fname.endswith('.png') and not fname.startswith('.')
    ]
)

print('Number of samples: ', len(input_img_paths))

for input_path, target_path in zip(input_img_paths[:10], target_img_paths[:10]):
    print(input_path, '|', target_path)


# 一张输入图像和相应的分割掩码是什么样的?
display(Image(filename=input_img_paths[9]))

img = PIL.ImageOps.autocontrast(load_img(target_img_paths[9]))
display(img)


# 准备Sequence类以加载和矢量化批量数据
class OxfordPets(keras.utils.Sequence):

    def __init__(self, batch_size, img_size, input_img_paths, target_img_paths):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.target_img_paths = target_img_paths

    def __len__(self):
        return len(self.target_img_paths) // self.batch_size

    def __getitem__(self, idx):
        i = idx * self.batch_size
        batch_input_img_paths = self.input_img_paths[i: i + self.batch_size]
        batch_target_img_paths = self.target_img_paths[i: i + self.batch_size]

        x = np.zeros((self.batch_size, ) + self.img_size + (3, ), dtype='float32')
        for j, path in enumerate(batch_input_img_paths):
            img = load_img(path, target_size=self.img_size)
            x[j] = img

        y = np.zeros((self.batch_size, ) + self.img_size +  (1, ), dtype='uint8')
        for j, path in enumerate(batch_target_img_paths):
            img = load_img(path, target_size=self.img_size, color_mode='grayscale')
            y[j] = np.expand_dims(img, 2)
            y[j] -= 1
        return x, y


# 准备U-Net Exception风格的模型
def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3, ))

    x = layers.Conv2D(32, 3, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    previous_block_activation = x

    for filters in [64, 128, 256]:
        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.SeparableConv2D(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

        residual = layers.Conv2D(filters, 1, strides=2, padding='same')(previous_block_activation)
        x = layers.add([x, residual])
        previous_block_activation = x

    for filters in [256, 128, 64, 32]:
        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation('relu')(x)
        x = layers.Conv2DTranspose(filters, 3, padding='same')(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding='same')(residual)
        x = layers.add([x, residual])
        previous_block_activation = x

    outputs = layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(x)

    model = keras.Model(inputs, outputs)
    # 保存模型架构
    keras.utils.plot_model(model, show_shapes=True)
    return model

keras.backend.clear_session()

model = get_model(img_size, num_classes)
model.summary()


# 留出验证拆分
val_samples = 1000
random.Random(1337).shuffle(input_img_paths)
random.Random(1337).shuffle(target_img_paths)
train_input_img_paths = input_img_paths[: -val_samples]
train_target_img_paths = input_img_paths[-val_samples: ]
val_input_img_paths = input_img_paths[-val_samples: ]
val_target_img_paths = target_img_paths[-val_samples: ]

train_gen = OxfordPets(
    batch_size, img_size, train_input_img_paths, train_target_img_paths
)
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)


# 训练模型
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')

callbacks = [
    keras.callbacks.ModelCheckpoint('oxford_segnentation.h5', save_best_only=True)
]

epochs = 15
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)


# 可视化预测
val_gen = OxfordPets(batch_size, img_size, val_input_img_paths, val_target_img_paths)
val_preds = model.predict(val_gen)


def display_mask(i):
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = PIL.ImageOps.autocontrast(keras.preprocessing.image.array_to_img(mask))
    display(img)


i = 10

display(Image(filename=val_input_img_paths[i]))

img = PIL.ImageOps.autocontrast(load_img(val_target_img_paths[i]))
display(img)

display_mask(i)

