
import tensorflow as tf
import numpy as np
import pathlib
from cnvrg import Endpoint
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

e = Endpoint()

model = tf.keras.models.load_model('output/model.h5')
index2label = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}

def preprocess_image(image):
  image = tf.image.decode_jpeg(image, channels=3)
  image = tf.image.resize(image, [224, 224])
  image /= 255.0  # normalize to [0,1] range
  image = image * 2 - 1
  image = tf.reshape(image, (1, 224, 224, 3))
  return image

def load_and_preprocess_image(path):
  image = tf.io.read_file(path)
  return preprocess_image(image)


def predict(img_path):
    image = load_and_preprocess_image(img_path)
    result = model.predict(image)
    resinfo = index2label[np.argmax(result)]
    e.log_metric("result", np.argmax(result))
    return resinfo

def test_images():
    data_path= '../cifar10_0/test'
    data_root = pathlib.Path(data_path)
    test_image_paths = list(data_root.glob('*/*'))
    test_image_paths = [str(path) for path in test_image_paths]
    print(test_image_paths)

    for ipath in test_image_paths:
        predict(ipath)
#test_images()
