import machine_learning_tool as ml
import cupy as np
from PIL import Image

model = ml.model([
    ml.conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=(1, 28, 28), learning_rate=0.1),
    ml.avg_pool2D(size=(3, 3), input_shape=(5, 24, 24)),
    ml.conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=(5, 22, 22), learning_rate=0.1),
    ml.avg_pool2D(size=(3, 3), input_shape=(5, 18, 18)),
    ml.flatten(input_shape=(5, 16, 16)),
    ml.dense(nods=10, activation='relu', input_shape=(5*16*16, ), learning_rate=0.1),
    ml.dense(nods=2, activation='sigmoid', input_shape=(10, ), no_activation_derivative=True, learning_rate=0.1)
])

def load(dir, shape):
    file = open(dir, 'rb').read()
    file = np.frombuffer(file)
    file = np.reshape(file, shape)
    return file

saved_data_dir = input('saved file directory:')

k1 = load('{}\\k1'.format(saved_data_dir), (5, 1, 5, 5))
b1 = load('{}\\b1'.format(saved_data_dir), (5, 24, 24))
k2 = load('{}\\k2'.format(saved_data_dir), (5, 5, 5, 5))
b2 = load('{}\\b2'.format(saved_data_dir), (5, 18, 18))
w3 = load('{}\\w3'.format(saved_data_dir), (5*16*16, 10))
b3 = load('{}\\b3'.format(saved_data_dir), (10, ))
w4 = load('{}\\w4'.format(saved_data_dir), (10, 2))
b4 = load('{}\\b4'.format(saved_data_dir), (2, ))

model.layers[0].kernel = k1
model.layers[0].bias = b1
model.layers[2].kernel = k2
model.layers[2].bias = b2
model.layers[5].weight = w3
model.layers[5].bias = b3
model.layers[6].weight = w4
model.layers[6].bias = b4

while True:
    image_dir = input('image directory:')
    image = Image.open(image_dir).convert('L')
    image = np.array(image)/255.0
    image = np.reshape(image, (1, 1, 28, 28))
    prediction = model.forward_propagation(image)[1][-1]
    print(np.argmax(prediction))
    print('=====================')