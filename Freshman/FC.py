# sigmoid函数、平方和误差、全连接网络、随机梯度下降优化算法
import numpy as np


class SigmoidActivator(object):
    
    def forward(self, weighted_input): # sigmoid函数
        return 1.0 / (1.0 + np.exp(-weighted_input))
    
    def backward(self, output): # sigmoid函数的导数(直接用到forward的结果)
        return output * (1 - output)


class FullConnectedLayer(object):
    
    def __init__(self, input_size, output_size, activator):
        self.input_size = input_size
        self.output_size = output_size # 下一层的维度
        self.activator = activator
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size)) # 传向下一层的权重
        self.b = np.zeros((output_size, 1)) # 传向下一层的偏置项
        self.output = np.zeros((output_size, 1))
        
    def forward(self, input_array): # 前向计算
        self.input = input_array
        self.output = self.activator.forward(
            np.matmul(self.W, input_array) + self.b
        )
        
    def backward(self, delta_array): # 接受下一层的delta，反向计算W,b梯度
        self.delta = self.activator.backward(self.input) * np.matmul(
            self.W.T, delta_array
        )
        self.W_grad = np.matmul(delta_array, self.input.T)
        self.b_grad = delta_array
        
    def update(self, learning_rate):
        self.W += learning_rate * self.W_grad
        self.b += learning_rate * self.b_grad
        

class Network(object):
    
    def __init__(self, layers_size):
        self.layers = [] # 不包括输出层
        for i in range(len(layers_size) - 1):
            self.layers.append(
                FullConnectedLayer(layers_size[i], layers_size[i + 1],
                                   SigmoidActivator())
            )
            
    def predict(self, sample):
        output = sample # 初始化输入向量
        for layer in self.layers: # 正向预测
            layer.forward(output)
            output = layer.output
        return output
    
    def train(self, labels, data_set, rate, epoch):
        for _ in range(epoch): # 训练轮数
            for d in range(len(data_set)):
                self.train_one_sample(
                    labels[d], data_set[d], rate
                )
    
    def train_one_sample(self, label, sample, rate):
        self.predict(sample)
        self.calc_gradient(label)
        self.update_weight(rate)
        
    def calc_gradient(self, label):
        # 初始化delta为输出层的delta
        delta = self.layers[-1].activator.backward(
            self.layers[-1].output
        ) * (label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta) # 计算delta和grad
            delta = layer.delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)


if __name__ == '__main__':
    layers_size = np.array([2, 3, 1])
    data_set = np.array([[[1],[1]], [[0],[0]]])
    labels = np.array([0, 1])
    n1 = Network(layers_size)
    n1.train(labels, data_set, 0.1, 1000)
    print(n1.predict([[1], [1]]))
    print(n1.predict([[0], [1]]))
    print(n1.predict([[1], [0]]))
    print(n1.predict([[0], [0]]))