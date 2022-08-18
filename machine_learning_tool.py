import cupy as np
from math import e
from cupyx.scipy.signal import correlate, convolve
from os import listdir
from random import choice
from time import time

def ReLU_deriv(X, second_param=0, third_param=0):
    return X>0

def ReLU(X):
    return np.maximum(X, 0.0001*X)

def sigmoid_deriv(X, second_param, third_param=0):
    a = sigmoid(X)
    da = a*(1-a)
    return da

def sigmoid(x):
    unit_one = np.full(x.shape, 1)
    unit_e = np.full(x.shape, e)
    y = unit_one/(unit_one+unit_e**(-x))
    return y

def softmax(a):
    Y = np.zeros_like(a.copy())
    for i, b in enumerate(a.copy()):
        min = np.min(b)
        if min < 0:
            b -= min * 2
        total = np.sum(b)
        y = b/total
        Y[i] = y
    return Y

def softmax_single(a):
    min = np.min(a)
    if min<0:
        a -= min*2
    total = np.sum(a)
    y = a/total
    return y

def softmax_deriv(f, second_param, third_param):
    which_one = third_param
    first = softmax_single(second_param.copy())[which_one]
    second_thing = second_param.copy()
    second_thing[which_one] += 0.000000001
    second = softmax_single(second_thing.copy())[which_one]
    deriv = (second-first)/0.000000001
    return deriv

def MSE(X, Y):
    cost = np.sum((Y-X)**2)/(np.prod(np.array(Y.shape)))
    return cost

def categorical_crossentropy(Y, Y_pred):
    Y = np.reshape(Y, (int(np.prod(np.array(Y.shape))), ))
    logged = np.log10(Y_pred)
    logged = np.reshape(logged, (int(np.prod(np.array(logged.shape))), ))
    cost_value = 0
    for i, a in enumerate(Y):
        if a == 0:
            continue
        try:
            int(logged[i])
            cost_value += a*logged[i]
        except:
            continue
    return -cost_value

def no_effect(x):
    return x

def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred)+(1-y_true)*np.log(1-y_pred))

def random_seed_generator(min2max, size):
    seed = []
    size_nums = []
    for i in range(*min2max):
        size_nums.append(i)
    for i in range(size):
        seed_choice = choice(size_nums)
        size_nums.remove(seed_choice)
        seed.append(seed_choice)
    return seed

def cost_deriv(Z, A, Y, function, A_deriv=True, A_activation_deriv=ReLU_deriv):
    very_small_unit = 0.00000001
    first_cost = function(Y, A)
    deriv = np.zeros_like(A)
    factor_size = len(deriv[0])
    for a in range(len(A)):
        for b in range(factor_size):
            A[a][b] += very_small_unit
            if A_deriv:
                deriv[a][b] = ((function(Y, A)-first_cost)/very_small_unit)*A_activation_deriv(A[a][b], second_param=Z[a], third_param=b)
            else:
                deriv[a][b] = ((function(Y, A) - first_cost) / very_small_unit)
            A[a][b] -= very_small_unit
    return deriv

class dense:
    def __init__(self, nods, activation, input_shape=None, learning_rate=0.001, no_activation_derivative=False):
        self.nods = nods
        self.input_shape = input_shape
        self.weight = 0
        try:
            self.weight = np.random.randn(input_shape[0], nods)
        except:
            pass
        self.bias = np.random.randn((nods))
        self.learning_rate = learning_rate
        self.Z = 0
        self.A = 0
        self.type = 'dense'
        self.activation_derivative = 1
        self.activation = ReLU
        self.no_activation_derivative = no_activation_derivative
        self.output_shape = (nods, )
        if activation == 'relu':
            self.activation_derivative = ReLU_deriv
            pass
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_deriv
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_derivative = softmax_deriv

    def init2(self, input_shape):
        self.input_shape = input_shape
        self.weight = np.random.randn(input_shape[0], self.nods)
        self.bias = np.random.rand((self.nods))

    def forward(self, x):
        self.Z = x.dot(self.weight)+self.bias
        self.A = self.activation(self.Z)
        return self.Z, self.A

    def backward(self, x, dZ, previous_activation_derivative=1):
        m = len(dZ)
        dw = (1/m)*x.T.dot(dZ)
        db = (1/m)*sum(dZ)
        dx = (1/m)*dZ.dot(self.weight.T)*previous_activation_derivative
        self.weight = self.weight - dw * self.learning_rate
        self.bias = self.bias - db * self.learning_rate
        return dx

class conv2D:
    def __init__(self, filters, kernel_size, activation, input_shape=None, learning_rate=0.001):
        self.filters = filters
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        try:
            self.kernel = np.random.randn(self.filters, self.input_shape[0], *kernel_size)
            self.expected_shape = int(self.input_shape[-1]-kernel_size[0]+1)
            self.bias = np.zeros((self.filters, self.expected_shape, self.expected_shape))
            self.output_shape = (filters, self.expected_shape, self.expected_shape)
        except:
            pass
        self.kernel_size = kernel_size
        self.Z = 0
        self.A = 0
        self.type = 'conv2D'
        self.activation_derivative = 1
        self.activation = ReLU
        if activation == 'relu':
            self.activation_derivative = ReLU_deriv
            pass
        elif activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_derivative = sigmoid_deriv
        elif activation == 'softmax':
            self.activation = softmax
            self.activation_derivative = softmax_deriv

    def init2(self, input_shape):
        self.expected_shape = int(input_shape[-1] - self.kernel_size[0] + 1)
        self.kernel = np.random.randn(self.filters, input_shape[0], *self.kernel_size)
        self.bias = np.zeros((self.filters, self.expected_shape, self.expected_shape))
        self.output_shape = (self.filters, self.expected_shape, self.expected_shape)
        self.input_shape = input_shape

    def forward(self, x):
        self.Z = np.zeros((len(x), *self.bias.shape))
        for a, one_x in enumerate(x):
            for f in range(self.filters):
                for i in range(self.input_shape[0]):
                    y = correlate(one_x[i], self.kernel[f][i], mode='valid', method='direct')
                    self.Z[a][f] += y
                self.Z[a][f] += self.bias[f]
        self.A = self.activation(self.Z)
        return self.Z, self.A

    def backward(self, x, dZ):
        m = len(x)
        db = sum(dZ)/m
        dk = np.zeros_like(self.kernel, dtype='float64')
        dx = np.zeros_like(x, dtype='float64')
        for a in range(m):
            for d in range(len(self.kernel)):
                for j in range(len(self.kernel[d])):
                    dk[d][j] += correlate(x[a][j], dZ[a][d], mode='valid', method='direct')
                    dx[a][j] += convolve(dZ[a][d], self.kernel[d][j], mode='full', method='direct')
        dk = dk/m
        dx = dx/m
        self.kernel = self.kernel - dk * self.learning_rate
        self.bias = self.bias - db * self.learning_rate
        return dx

class avg_pool2D:
    def __init__(self, size, input_shape=None):
        self.size = size
        self.kernel = np.ones(self.size)
        try:
            self.expected_shape = int(input_shape[-1]-self.size[0]+1)
            self.output_shape = (input_shape[0], self.expected_shape, self.expected_shape)
        except:
            pass
        self.Z = 0
        self.input_shape = input_shape
        self.filter_size = self.size[0]*self.size[1]
        self.type = 'avg_pool2D'

    def init2(self, input_shape):
        self.input_shape = input_shape
        self.expected_shape = int(input_shape[-1] - self.size[0] + 1)
        self.output_shape = (input_shape[0], self.expected_shape, self.expected_shape)

    def forward(self, x):
        m = len(x)
        self.Z = np.zeros((m, self.input_shape[0], self.expected_shape, self.expected_shape))
        for a in range(m):
            for b in range(self.input_shape[0]):
                self.Z[a][b] = correlate(x[a][b], self.kernel, mode='valid', method='direct')
        self.Z = self.Z / self.filter_size
        return self.Z

    def backward(self, dZ):
        m = len(dZ)
        dX = np.zeros((m, *self.input_shape))
        for a in range(m):
            for b in range(self.input_shape[0]):
                dX[a][b] = correlate(dZ[a][b], self.kernel, mode='full', method='direct')
        return dX

class flatten:
    def __init__(self, input_shape=None):
        self.input_shape = input_shape
        try:
            self.output_shape_1 = int(np.prod(np.asarray(self.input_shape)))
            self.output_shape = (int(np.prod(np.asarray(self.input_shape))), )
        except:
            pass
        self.type = 'flatten'
        self.activation_derivative = no_effect

    def init2(self, input_shape):
        self.input_shape = input_shape
        self.output_shape_1 = int(np.prod(np.asarray(input_shape)))
        self.output_shape = (int(np.prod(np.asarray(input_shape))), )

    def forward(self, x):
        return np.reshape(x, (len(x), self.output_shape_1))

    def backward(self, dz):
        return np.reshape(dz, (len(dz), *self.input_shape))

class model:
    def __init__(self, layers, name='model', cost='mse'):
        self.name = name
        self.layers = layers
        self.types = []
        for i in self.layers:
            self.types.append(i.type)
        self.Zs = []
        self.As = []
        if cost == 'mse':
            self.cost_function = MSE
        elif cost == 'categorical_crossentropy':
            self.cost_function = binary_cross_entropy

        shapes = []
        for layer in layers:
            if layer.input_shape == None:
                layer.init2(shapes[-1])
                shapes.append(layer.output_shape)
            else:
                shapes.append(layer.output_shape)

    def forward_propagation(self, x):
        target = x
        for layer in self.layers:
            if layer.type == 'conv2D' or layer.type == 'dense':
                Z, A = layer.forward(target)
                target = A
                self.Zs.append(Z)
                self.As.append(A)
            elif layer.type == 'avg_pool2D' or layer.type == 'flatten':
                Z = layer.forward(target)
                target = Z
                self.Zs.append(Z)
                self.As.append(Z)

        return self.Zs, self.As

    def backward_propagation(self, x, y):
        Zs, As = self.forward_propagation(x)
        target_dZ = cost_deriv(Z=Zs[-1], A=As[-1], Y=y, function=self.cost_function, A_activation_deriv=self.layers[-1].activation_derivative, A_deriv=True)
        for count in range(len(self.layers)):
            layer_index = int(len(self.layers)-count-1)
            if self.types[layer_index] == 'dense':
                if layer_index != 0:
                    if self.layers[layer_index].no_activation_derivative == False:
                        target_dZ = self.layers[layer_index].backward(As[layer_index-1], target_dZ, previous_activation_derivative=self.layers[layer_index-1].activation_derivative(Zs[layer_index-1]))
                    else:
                        target_dZ = self.layers[layer_index].backward(As[layer_index - 1], target_dZ,previous_activation_derivative=1)
                else:
                    target_dZ = self.layers[layer_index].backward(x, target_dZ)

            elif self.types[layer_index] == 'conv2D':
                if layer_index != 0:
                    target_dZ = self.layers[layer_index].backward(As[layer_index-1], target_dZ)
                else:
                    target_dZ = self.layers[layer_index].backward(x, target_dZ)

            elif self.types[layer_index] == 'avg_pool2D' or self.types[layer_index] == 'flatten':
                target_dZ = self.layers[layer_index].backward(target_dZ)

        cost = self.cost_function(y, As[-1])
        return cost

    def sgd(self, x, y, batch_size=32):
        random_seed = random_seed_generator((0, len(y)), batch_size)
        x_shape = x[0].shape
        y_shape = y[0].shape
        batch_x = np.zeros((batch_size, *x_shape))
        batch_y = np.zeros((batch_size, *y_shape))
        for a, i in enumerate(random_seed):
            batch_x[a] = x[i]
            batch_y[a] = y[i]
        cost = self.backward_propagation(x=batch_x, y=batch_y)
        return cost

    def save(self, save_dir):
        shapes = open(save_dir+'\\shapes.txt', 'wb')
        shapes_txt = ''
        for count, layer in enumerate(self.layers):
            if layer.type == 'dense':
                file1 = open(save_dir + '\\{}w'.format(count), 'wb')
                file1.write(layer.weight.tobytes())
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                file2.write(layer.bias.tobytes())
                file2.close()

                shapes_txt += '{}\n'.format(layer.weight.shape)
                shapes_txt += '{}\n'.format(layer.bias.shape)
            elif layer.type == 'conv2D':
                file1 = open(save_dir + '\\{}k'.format(count), 'wb')
                file1.write(layer.kernel.tobytes())
                file1.close()

                file2 = open(save_dir + '\\{}b'.format(count), 'wb')
                file2.write(layer.bias.tobytes())
                file2.close()

                shapes_txt += '{}\n'.format(layer.kernel.shape)
                shapes_txt += '{}\n'.format(layer.bias.shape)
            else:
                file1 = open(save_dir+'\\{}a'.format(count), 'wb')
                file1.close()
                file2 = open(save_dir+'\\{}b'.format(count), 'wb')
                file2.close()
                shapes_txt += 'a\n'
                shapes_txt += 'b\n'
        shapes.write(shapes_txt.encode())
        shapes.close()

    def load(self, save_dir):
        try:
            shapes = open(save_dir+'\\shapes.txt', 'rb').readlines()
            files = listdir(save_dir)
            files.remove('shapes.txt')
            files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
            for count, layer in enumerate(self.layers):
                if layer.type == 'conv2D':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb').read()
                    kernel = np.frombuffer(file1)
                    kernel = np.reshape(kernel, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb').read()
                    bias = np.frombuffer(file2)
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.kernel = kernel
                    layer.bias = bias
                elif layer.type == 'dense':
                    file1 = open(save_dir+'\\{}'.format(files[int(2*count+1)]), 'rb').read()
                    weight = np.frombuffer(file1)
                    weight = np.reshape(weight, eval(shapes[int(2*count)]))

                    file2 = open(save_dir+'\\{}'.format(files[int(2*count)]), 'rb').read()
                    bias = np.frombuffer(file2)
                    bias = np.reshape(bias, eval(shapes[int(2*count+1)]))

                    layer.weight = weight
                    layer.bias = bias
        except:
            pass

    def train(self, x, y, optimizer, epochs, batch_size=32):
        if optimizer == 'gd':
            opt = self.backward_propagation
            values = (x, y)
        elif optimizer == 'sgd':
            opt = self.sgd
            values = (x, y, batch_size)
        for epoch in range(epochs):
            start_time = time()
            cost_value = opt(*values)
            end_time = time()
            duration = end_time-start_time
            print('epoch: {} | duration: {} seconds | cost: {}'.format(epoch+1, round(duration), cost_value))