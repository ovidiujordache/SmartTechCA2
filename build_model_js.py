import os
import csv
import tensorflow as tf
from tensorflow.keras.layers import Cropping2D, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.data import Dataset
from tensorflow.image import decode_jpeg
from tqdm import tqdm
import linecount
import argparse


parser = argparse.ArgumentParser(description='Train model for autonomous driving')
parser.add_argument('--data', dest='data_dir', default='/home/ovi/PROJECTS_YEAR_4/SMART_TECH/SmartTechCA2Data/track_1/one_lap/',
                    help='Directory containing driving log CSV and images')
parser.add_argument('--model', dest='model_dir', default='model', help='Directory to save/load the model')
parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
args = parser.parse_args()

data_dir = args.data_dir
model_dir = args.model_dir
epochs = args.epochs

path_to_csv = os.path.join(data_dir, 'driving_log.csv')

def data_generator():
    while True:
        with open(path_to_csv) as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                center_image = tf.io.read_file(row['center'].strip())
                left_image = tf.io.read_file(row['left'].strip())
                right_image = tf.io.read_file(row['right'].strip())

                offset = 0.333

                yield center_image.numpy(), float(row['steering'])
                yield left_image.numpy(), float(row['steering']) + offset
                yield right_image.numpy(), float(row['steering']) - offset

def init_model():
    try:
        model = tf.keras.models.load_model(f'file://{model_dir}/model')
        print(f'Model loaded from: {model_dir}')
    except:
        model = Sequential([
            Cropping2D(cropping=((75, 25), (0, 0)), input_shape=(160, 320, 3)),
            Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(units=1024, activation='relu'),
            Dropout(rate=0.25),
            Dense(units=128, activation='relu'),
            Dense(units=1, activation='linear')
        ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
def preprocess_image(image_buffer, steering):
    # Ensure image_buffer is of type tf.Tensor with dtype=tf.string
    image_buffer = tf.convert_to_tensor(image_buffer, dtype=tf.string)

    # Decode JPEG and normalize pixel values
    image = tf.image.decode_jpeg(image_buffer, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # Normalize pixel values to the range [0, 1]
    image /= 255.0

    # Use the existing TensorFlow tensor for steering
    steering = tf.cast(steering, dtype=tf.float32)

    return  image, steering

batch_size = 64
data = Dataset.from_generator(data_generator, output_signature=(tf.TensorSpec(shape=(), dtype=tf.string), tf.TensorSpec(shape=(), dtype=tf.float32)))
dataset = data.map(preprocess_image).shuffle(buffer_size=batch_size).batch(batch_size)

model = init_model()
total_samples=0
with open(path_to_csv) as file:
    total_samples = sum(1 for line in file) * 3


model.fit(dataset, epochs=epochs, steps_per_epoch=total_samples // batch_size)

model.save(f'file://{model_dir}/model')

print(f'Model saved to: {model_dir}')
