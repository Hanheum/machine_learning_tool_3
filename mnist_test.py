import machine_learning_tool as ml
import cupy as np
from PIL import Image
from os import listdir
from time import time

dataset_dir = 'C:\\Users\\chh36\\Desktop\\mini_mnist\\'

train_dir = listdir(dataset_dir)
train_images, train_labels = [], []

total_count = 0
for dir in train_dir:
    num = len(listdir(dataset_dir+dir))
    total_count += num

for count, category in enumerate(train_dir):
    for image_name in listdir(dataset_dir+category):
        image = Image.open(dataset_dir+category+'\\'+image_name).convert('L')
        image = image.resize((28, 28))
        image = np.array(image)/255.0
        image = np.reshape(image, (1, 28, 28))
        train_images.append(image)

        label = np.zeros((len(train_dir)))
        label[count] = 1
        train_labels.append(label)

print('image load completed, {} images found, {} classes'.format(total_count, len(train_dir)))

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

model = ml.model([
    ml.conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=(1, 28, 28), learning_rate=0.1),
    ml.avg_pool2D(size=(3, 3), input_shape=(5, 24, 24)),
    ml.conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=(5, 22, 22), learning_rate=0.1),
    ml.avg_pool2D(size=(3, 3), input_shape=(5, 18, 18)),
    ml.flatten(input_shape=(5, 16, 16)),
    ml.dense(nods=10, activation='relu', input_shape=(5*16*16, ), learning_rate=0.1),
    ml.dense(nods=2, activation='sigmoid', input_shape=(10, ), no_activation_derivative=True, learning_rate=0.1)
])

def saver(x, file_dir):
    file_x = x.tobytes()
    file = open(file_dir, 'wb')
    file.write(file_x)
    file.close()

epochs = 100
for epoch in range(epochs):
    start = time()
    cost = model.backward_propagation(train_images, train_labels)
    end = time()
    duration = round(end-start)
    print('epoch: {} | duration : {} seconds | cost: {}'.format(epoch, duration, cost))

    k1 = model.layers[0].kernel
    saver(k1, 'C:\\Users\\chh36\\Desktop\\save_file\\k1')
    b1 = model.layers[0].bias
    saver(b1, 'C:\\Users\\chh36\\Desktop\\save_file\\b1')
    k2 = model.layers[2].kernel
    saver(k2, 'C:\\Users\\chh36\\Desktop\\save_file\\k2')
    b2 = model.layers[2].bias
    saver(b2, 'C:\\Users\\chh36\\Desktop\\save_file\\b2')
    w3 = model.layers[5].weight
    saver(w3, 'C:\\Users\\chh36\\Desktop\\save_file\\w3')
    b3 = model.layers[5].bias
    saver(b3, 'C:\\Users\\chh36\\Desktop\\save_file\\b3')
    w4 = model.layers[6].weight
    saver(w4, 'C:\\Users\\chh36\\Desktop\\save_file\\w4')
    b4 = model.layers[6].bias
    saver(b4, 'C:\\Users\\chh36\\Desktop\\save_file\\b4')