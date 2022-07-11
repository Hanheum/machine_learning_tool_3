import numpy as np

image_data = open('C:\\Users\\HLK\\Desktop\\mnist_train_images', 'rb')
image_data.read(16)
train_images = np.zeros((60000, 28, 28))
for i in range(60000):
    image = image_data.read(784)
    image = np.frombuffer(image, dtype=np.uint8)
    image = np.reshape(image, (28, 28))
    train_images[i] = image

label_data = open('C:\\Users\\HLK\\Desktop\\mnist_train_labels', 'rb')
label_data.read(8)
label_data = label_data.read()
label_data = np.frombuffer(label_data, dtype=np.uint8)

def one_hot(index, target_num):
    arr = np.zeros((target_num, ))
    arr[index] = 1
    return arr

train_labels = []
for i in label_data:
    label = one_hot(i, 10)
    train_labels.append(label)
train_labels = np.asarray(train_labels)

