import kagglehub
import os
import tensorflow as tf
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.models import Sequential
import time
import matplotlib.pyplot as plt

filepath = kagglehub.dataset_download("alessiocorrado99/animals10")
print(filepath)
data='/kaggle/input/animals10/raw-img'
print("Classes:", os.listdir(data))
images_path = data
isize=(224,224)
bsize=32
train= tf.keras.utils.image_dataset_from_directory(data,validation_split=0.2,subset="training",seed=123,image_size=isize,batch_size=bsize)
val= tf.keras.utils.image_dataset_from_directory(data,validation_split=0.2,subset="validation",seed=123,image_size=isize,batch_size=bsize)
class_names = train.class_names
print("Class names:", class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_d = train.prefetch(buffer_size=AUTOTUNE)
val_d = val.prefetch(buffer_size=AUTOTUNE)
def build_zfnet(input_shape=(224, 224, 3), num_classes=10):
    model = Sequential()
    model.add(layers.Conv2D(96, (7, 7), strides=2, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(256, (5, 5), strides=2, activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))
    model.add(layers.Conv2D(384, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((3, 3), strides=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def build_vgg16(input_shape=(224, 224, 3), num_classes=10):
    model = Sequential()
    for filters in [64, 128, 256, 512, 512]:
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same', input_shape=input_shape if filters==64 else None))
        model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        if filters >= 256:
            model.add(layers.Conv2D(filters, (3, 3), activation='relu', padding='same'))
        model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(4096, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(num_classes, activation='softmax'))
    return model

def inception_module(x, f1, f3_in, f3_out, f5_in, f5_out, pool_proj):
    path1 = layers.Conv2D(f1, (1,1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(f3_in, (1,1), padding='same', activation='relu')(x)
    path2 = layers.Conv2D(f3_out, (3,3), padding='same', activation='relu')(path2)
    path3 = layers.Conv2D(f5_in, (1,1), padding='same', activation='relu')(x)
    path3 = layers.Conv2D(f5_out, (5,5), padding='same', activation='relu')(path3)
    path4 = layers.MaxPooling2D((3,3), strides=1, padding='same')(x)
    path4 = layers.Conv2D(pool_proj, (1,1), padding='same', activation='relu')(path4)
    return layers.concatenate([path1, path2, path3, path4], axis=-1)

def build_googlenet(input_shape=(224,224,3), num_classes=10):
    input_layer = Input(shape=input_shape)
    x = layers.Conv2D(64, (7,7), strides=2, padding='same', activation='relu')(input_layer)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    x = layers.Conv2D(64, (1,1), padding='same', activation='relu')(x)
    x = layers.Conv2D(192, (3,3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    x = inception_module(x, 64, 96, 128, 16, 32, 32)
    x = inception_module(x, 128, 128, 192, 32, 96, 64)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    x = inception_module(x, 192, 96, 208, 16, 48, 64)
    x = inception_module(x, 160, 112, 224, 24, 64, 64)
    x = inception_module(x, 128, 128, 256, 24, 64, 64)
    x = inception_module(x, 112, 144, 288, 32, 64, 64)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = layers.MaxPooling2D((3,3), strides=2, padding='same')(x)
    x = inception_module(x, 256, 160, 320, 32, 128, 128)
    x = inception_module(x, 384, 192, 384, 48, 128, 128)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    output_layer = layers.Dense(num_classes, activation='softmax')(x)
    return Model(input_layer, output_layer)

def train_and_evaluate(model, name):
    model.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    start=time.time()
    h=model.fit(train_d,validation_data=val_d,epochs=10) #history
    d=time.time()-start #duration
    loss,acc=model.evaluate(val_d)
    print(f"{name} Accuracy: {acc:.4f} , Inference Time: {d:.2f}s")
    return h,acc,d

print("\n Training ZFNet...")
zfnet_history, zfnet_acc, zfnet_time = train_and_evaluate(build_zfnet(input_shape=(224,224,3), num_classes=len(class_names)), "ZFNet")
print("\n Training VGG16...")
vgg16_history, vgg16_acc, vgg16_time = train_and_evaluate(build_vgg16(input_shape=(224,224,3), num_classes=len(class_names)), "VGG16")
print("\n Training GoogLeNet...")
googlenet_history, googlenet_acc, googlenet_time = train_and_evaluate(build_googlenet(input_shape=(224,224,3), num_classes=len(class_names)), "GoogLeNet")

  def plot_history(histories, labels, metric):
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        plt.plot(hist.history[metric], label=f'{label} Train')
        plt.plot(hist.history[f'val_{metric}'], label=f'{label} Val')
    plt.title(f'Model {metric.capitalize()}')
    plt.xlabel('Epochs')
    plt.ylabel(metric.capitalize())
    plt.legend()
    plt.grid(True)
    plt.show()
plot_history([zfnet_history, vgg16_history, googlenet_history], ['ZFNet', 'VGG16', 'GoogLeNet'], 'accuracy')
plot_history([zfnet_history, vgg16_history, googlenet_history], ['ZFNet', 'VGG16', 'GoogLeNet'], 'loss')

print("\n Final Accuracy and Inference Time Comparison:")
print(f"ZFNet--- Accuracy: {zfnet_acc:.4f}, Time: {zfnet_time:.2f}s")
print(f"VGG16--- Accuracy: {vgg16_acc:.4f}, Time: {vgg16_time:.2f}s")
print(f"GoogLeNet--- Accuracy: {googlenet_acc:.4f}, Time: {googlenet_time:.2f}s")
