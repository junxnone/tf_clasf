import pathlib
import random
import tensorflow as tf
import argparse

from tensorflow.keras.callbacks import ReduceLROnPlateau
AUTOTUNE = tf.data.experimental.AUTOTUNE

parser = argparse.ArgumentParser(description='CIFAR10 Training')
parser.add_argument('--data_path', '-dp', default='../cifar10_0', type=str, help='data path')
parser.add_argument('--epochs', '-e', default=20, type=int, help='training epochs')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--batch', '-b', default=32, type=int, help='training batch')

args = parser.parse_args()


def load_data(data_path):
    data_root = pathlib.Path(data_path)
    train_image_paths = list(data_root.glob('train/*/*'))
    train_image_paths = [str(path) for path in train_image_paths]
    train_image_count = len(train_image_paths)
    print(f'Train images: {train_image_count}')
    
    test_image_paths = list(data_root.glob('test/*/*'))
    test_image_paths = [str(path) for path in test_image_paths]
    test_image_count = len(test_image_paths)
    print(f'Test images: {test_image_count}')

    label_names = sorted(item.name for item in data_root.glob('train/*') if item.is_dir())
    print(f'Classes: {label_names}')

    label_to_index = dict((name, index) for index, name in enumerate(label_names))
    print(f'Labels 2 Index {label_to_index}')

    index_to_label = dict((index, name) for index, name in enumerate(label_names))
    print(f'Index 2 Labels{index_to_label}')

    train_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in train_image_paths]
    test_image_labels = [label_to_index[pathlib.Path(path).parent.name]
                        for path in test_image_paths]
    #for ipath, ilabel in zip(test_image_paths, test_image_labels):
    #    print(f'{ipath}-{index_to_label[ilabel]}')

    train_ds = tf.data.Dataset.from_tensor_slices((train_image_paths, train_image_labels))
    train_image_label_ds = train_ds.map(load_and_preprocess_from_path_label)
    train_ds = train_image_label_ds.shuffle(buffer_size=train_image_count)
    train_ds = train_ds.batch(args.batch).prefetch(buffer_size=AUTOTUNE)
 
    test_ds = tf.data.Dataset.from_tensor_slices((test_image_paths, test_image_labels))
    test_image_label_ds = test_ds.map(load_and_preprocess_from_path_label)
    test_ds = test_image_label_ds.shuffle(buffer_size=test_image_count)
    test_ds = test_ds.batch(args.batch).prefetch(buffer_size=AUTOTUNE)

    return train_ds, test_ds, label_to_index, index_to_label

def load_and_preprocess_from_path_label(path, label):
  return load_and_preprocess_image(path), label

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range
  image = image * 2 - 1
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)

if __name__ == '__main__':

    train_ds, test_ds, label_to_index, index_to_label = load_data(args.data_path)

    mobile_net = tf.keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False)
    mobile_net.trainable=False
    model = tf.keras.Sequential([
    mobile_net,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(len(label_to_index), activation = 'softmax')])

    print(model.summary())

    model.compile(optimizer=tf.keras.optimizers.SGD(args.lr),
                loss='sparse_categorical_crossentropy',
                metrics=["accuracy"])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                patience=5, min_lr=0.001)

    model.fit(train_ds, epochs=args.epochs, validation_data=test_ds, callbacks=[reduce_lr])
    model.save('output/model.h5')
