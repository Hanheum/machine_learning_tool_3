import machine_learning_tool as ml
import cupy as np
from PIL import Image
from os import listdir
from time import time

dataset_dir = 'C:\\Users\\chh36\\Desktop\\target_trace\\'

train_dir = listdir(dataset_dir)
train_images, train_labels = [], []

total_count = 0
for dir in train_dir:
    num = len(listdir(dataset_dir+dir))
    total_count += num

for count, category in enumerate(train_dir):
    for image_name in listdir(dataset_dir+category):
        image = Image.open(dataset_dir+category+'\\'+image_name).convert('RGB')
        image = image.resize((100, 100))
        image = np.array(image)/255.0
        image = np.reshape(image, (10000, 3))
        image = image.T
        image = np.reshape(image, (3, 100, 100))
        train_images.append(image)

        label = np.zeros((len(train_dir)))
        label[count] = 1
        train_labels.append(label)

print('image load completed, {} images found, {} classes'.format(total_count, len(train_dir)))

train_images = np.asarray(train_images)
train_labels = np.asarray(train_labels)

L1 = ml.conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=(3, 100, 100), learning_rate=0.1)
L2 = ml.avg_pool2D(size=(3, 3), input_shape=(5, 96, 96))
L3 = ml.conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=(5, 94, 94), learning_rate=0.1)
L4 = ml.avg_pool2D(size=(3, 3), input_shape=(5, 90, 90))
L5 = ml.conv2D(filters=5, kernel_size=(5, 5), activation='relu', input_shape=(5, 88, 88), learning_rate=0.1)
L6 = ml.avg_pool2D(size=(3, 3), input_shape=(5, 84, 84))
L7 = ml.flatten(input_shape=(5, 82, 82))
L8 = ml.dense(nods=20, activation='relu', input_shape=(5*82*82, ), learning_rate=0.1)
L9 = ml.dense(nods=10, activation='relu', input_shape=(20, ), learning_rate=0.1)
L10 = ml.dense(nods=2, activation='sigmoid', input_shape=(10, ), learning_rate=0.1)

model = ml.model([L1, L2, L3, L4, L5, L6, L7, L8, L9, L10])

epochs = 10
for epoch in range(epochs):
    start = time()
    cost = model.backward_propagation(train_images, train_labels)
    end = time()
    duration = round(end-start)
    print('epoch: {} | duration: {} seconds | cost: {}'.format(epoch, duration, cost))