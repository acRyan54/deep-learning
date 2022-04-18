from functools import reduce
import re


class Perceptron(object): # 感知器
    
    def __init__(self, input_num, activator):
        self.activator = activator
        self.weights = [0.0 for _ in range(input_num)]
        self.bias = 0.0
    
    def __str__(self):
        return 'weights:\t%s\nbias:\t%s' % (self.weights, self.bias)
    
    def predict(self, input_vec):
        return self.activator(
            reduce(lambda a, b: a + b,
                   map(lambda w, x: w * x,
                       self.weights, input_vec), 0.0) + self.bias
        )
    
    def train(self, input_vecs, labels, iteration, rate):
        for _ in range(iteration): # iteration为训练次数
            self.__one_iteration(input_vecs, labels, rate)
    
    def __one_iteration(self, input_vecs, labels, rate):
        for (input_vec, label) in zip(input_vecs, labels):
            output = self.predict(input_vec)
            self.__update_weights(input_vec, output, label, rate)
            
    def __update_weights(self, input_vec, output, label, rate):
        delta = label - output
        self.weights = list(map(lambda w, x: w + rate * delta * x,
                           self.weights, input_vec))
        self.bias += rate * delta
        

def train_and_perceptron():
    def sgn(x): # 阶跃函数
        return 1 if x > 0 else 0
    p = Perceptron(2, sgn)
    input_vecs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    labels = [1, 0, 0, 0]
    p.train(input_vecs, labels, 10, 0.1)
    return p

def train_or_perceptron():
    def sgn(x):
        return 1 if x > 0 else 0
    p = Perceptron(2, sgn)
    input_vecs = [[1, 1], [1, 0], [0, 1], [0, 0]]
    labels = [1, 1, 1, 0]
    p.train(input_vecs, labels, 10, 0.1)
    return p

if __name__ == '__main__':
    p1 = train_and_perceptron()
    print(p1)
    # while(True): # 测试
    #     input_vec = list(map(int, re.split(r'[\s,.]+', input().strip())))
    #     print(p.predict(input_vec))
        
    p2 = train_or_perceptron()
    print(p2)
    
    
'''
weights:        [0.1, 0.2]
bias:   -0.2
weights:        [0.1, 0.1]
bias:   0.0
'''