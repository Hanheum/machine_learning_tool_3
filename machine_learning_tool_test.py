import machine_learning_tool as ml
import cupy as np

x = [
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 0],
    [0, 0],
    [0, 1]
]

y = [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
]

x = np.asarray(x)
y = np.asarray(y)

model = ml.model([
    ml.dense(nods=10, activation='relu', input_shape=(2, ), learning_rate=0.1),
    ml.dense(nods=10, activation='relu', input_shape=(10, ), learning_rate=0.1),
    ml.dense(nods=10, activation='relu', input_shape=(10, ), learning_rate=0.1),
    ml.dense(nods=3, activation='relu', input_shape=(10, ), learning_rate=0.1)
])

for i in range(10000):
    print(i)
    model.backward_propagation(x, y)

prediction = model.forward_propagation(x)[1][-1]
for i in prediction:
    minilist = []
    for a in i:
        minilist.append(round(float(a), 3))
    print(minilist)