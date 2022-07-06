import cupy as np
from math import e
from cupyx.scipy.signal import correlate, convolve
from os import listdir
from random import choice

def ReLU_deriv(X):
    return X>0

def ReLU(X):
    return np.maximum(X, 0)

def sigmoid_deriv(X):
    a = sigmoid(X)
    da = a*(1-a)
    return da

def sigmoid(x):
    unit_one = np.full(x.shape, 1)
    unit_e = np.full(x.shape, e)
    y = unit_one/(unit_one+unit_e**(-x))
    return y

def softmax(a) :
    c = np.max(a)
    exp_a = np.exp(a-c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def MSE(X, Y):
    cost = np.sum((Y-X)**2)/(np.prod(np.array(Y.shape)))
    return cost

def categorical_crossentropy(Y, Y_pred):
    logged = np.log10(Y_pred)
    print(Y_pred)
    sumed = Y.dot(logged.T)
    CE = -np.sum(sumed)
    print(CE)
    return CE

def no_effect(x):
    return x

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

def cost_deriv(A, Y, function, A_deriv=True):
    very_small_unit = 0.00000001
    first_cost = function(A, Y)
    deriv = np.zeros_like(A)
    factor_size = len(deriv[0])
    for a in range(len(A)):
        for b in range(factor_size):
            A[a][b] += very_small_unit
            if A_deriv:
                deriv[a][b] = ((MSE(A, Y)-first_cost)/very_small_unit)*sigmoid_deriv(A[a][b])
            else:
                deriv[a][b] = ((MSE(A, Y) - first_cost) / very_small_unit)
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
        self.bias = np.zeros((nods))
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

    def init2(self, input_shape):
        self.input_shape = input_shape
        self.weight = np.random.randn(input_shape[0], self.nods)
        self.bias = np.zeros((self.nods))

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
    def __init__(self, filters, kernel_size, activation, input_shape, learning_rate=0.001):
        self.filters = filters
        self.input_shape = input_shape
        self.learning_rate = learning_rate
        self.kernel = np.random.randn(self.filters, self.input_shape[0], *kernel_size)
        self.expected_shape = int(self.input_shape[-1]-kernel_size[0]+1)
        self.bias = np.random.randn(self.filters, self.expected_shape, self.expected_shape)
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

    def forward(self, x):
        self.Z = np.zeros((len(x), self.filters, self.expected_shape, self.expected_shape))
        for a, one_x in enumerate(x):
            for f in range(self.filters):
                for i in range(self.input_shape[0]):
                    y = correlate(one_x[i], self.kernel[f][i], mode='valid', method='direct')+self.bias[f]
                    self.Z[a][f] = y
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
    def __init__(self, size, input_shape):
        self.size = size
        self.kernel = np.ones(self.size)
        self.expected_shape = int(input_shape[-1]-self.size[0]+1)
        self.Z = 0
        self.input_shape = input_shape
        self.filter_size = self.size[0]*self.size[1]
        self.type = 'avg_pool2D'

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
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = int(np.prod(np.asarray(self.input_shape)))
        self.type = 'flatten'
        self.activation_derivative = no_effect

    def forward(self, x):
        return np.reshape(x, (len(x), self.output_shape))

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
            self.cost_function = categorical_crossentropy

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
        target_dZ = cost_deriv(As[-1], y, self.cost_function)
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

        cost = self.cost_function(As[-1], y)
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