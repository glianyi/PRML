import random
import struct
from datetime import datetime


class Node(object):
    def __init__(self,layer_index,node_index):

        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.upstream = []
        self.output = 0
        self.delta = 0

    def set_output(self,output):
        self.output = output

    def append_downstream_connection(self,conn):
        self.upstream.append(conn)

    def calc_output(self):
        output = reduce(lambda ret,conn : ret + conn.upstream_node.output * conn.weight,self.upstream,0)
        # todo
        self.output = sigmod(output)

    def calc_hidden_layer_delta(self):

        downstream_delta = reduce(
            lambda ret,conn : ret+conn.downstream_node.delta*conn.weight,
            self.downstream,0.0)

        self.delta = self.output*(1-self.output) * downstream_delta

    def calc_out_put_layer_detal(self,label):

        self.delta = self.output * (1-self.output) * (label-self.output)

    def __str__(self):

        node_str = '%u-%u: output: %f delta: %f' % (self.layer_index,self.node_index,self.output,self.delta)
        downstream_str = reduce(lambda ret,conn : ret + '\t\n' + str(conn),self.downstream,'')
        upstream_str = reduce(lambda ret,conn : ret + '\t\n' + str(conn),self.upstream,'')
        return node_str + '\n\tdownstream' + downstream_str + '\n\tupstream' + upstream_str


class ConstNode(object):
    def __int__(self,layer_index,node_index):

        self.layer_index = layer_index
        self.node_index = node_index
        self.downstream = []
        self.output = 1

    def append_downstream_connection(self,conn):

        self.downstream.append(conn)


    def calc_hidden_layer_delta(self):
        downstream_delta = reduce(
            lambda ret,conn:ret + conn.downstream_node.delta*conn.weight,self.downstream,0.0)
        self.delta = self.output*(1-self.output)*downstream_delta

    def __str__(self):
        node_str = '%u-%u: output: 1' % (self.layer_index,self.node_index)
        downstream_str = reduce(lambda ret,conn:ret+'\n\t' + str(conn),self.downstream,'')
        return node_str + '\n\tdownstream' + downstream_str


class Layer(object):
    def __init__(self,layer_index,node_count):
        self.layer_index = layer_index
        self.nodes = []
        for i in range(node_count):
            self.nodes.append(Node(layer_index,i))
        self.nodes.append(ConstNode(layer_index,node_count))

    def set_output(self,data):
        for i in range(data):
            self.nodes[i].set_output(data[i])

    def calc_output(self):
        for node in self[:-1]:
            node.calc_output()

    def dump(self):
        for node in self.nodes:
            print node


class Connection(object):

    def __init__(self,upstream_node,downstream_node):

        self.upstream_node = upstream_node
        self.downstream_node = downstream_node
        self.weight = random.uniform(-0.1,0,1)
        self.gradient = 0.0

    def calc_gradient(self):
        self.gradient = self.downstream_node.delta * self.upstream_node.output

    def get_gradient(self):
        return self.gradient

    def update_weight(self,rate):
        self.calc_gradient()
        self.weight += rate*self.gradient

    def __str__(self):
        return '(%u-%u) -> (%u-%u) = %f' % (
            self.upstream_node.layer_index,
            self.upstream_node.node_index,
            self.downstream_node.layer_index,
            self.downstream_node.node_index,self.weight)

class Connections(object):
    def __init__(self):
        self.connections = []

    def add_connection(self,connection):
        self.connections.append(connection)

    def dump(self):
        for conn in self.connections:
            print conn

class Network(object):
    def __init__(self,layers):
        self.connections = Connections()
        self.layers = []
        layer_count = len(layers)
        node_count = 0
        for i in range(layer_count):
            self.layers.append(Layer(i,layers[i]))
        for layer in range(layer_count-1):
            connections = [Connection(upstream_node,downstream_node)
                           for upstream_node in self.layers[layer].upstream_node
                           for downstream_node in self.layers[layer+1].nodes[:-1]]
            for conn in connections:
                self.connections.add_connection(conn)
                conn.downstream_node.append_upstream_connection(conn)
                conn.upstream_node.append_downstream_connection(conn)

    def train(self,labels,data_set,rate,iteration):
        for i in range(iteration):
            for d in range(len(data_set)):
                self.train_one_sample(labels[d],data_set[d],rate)

    def train_one_sample(self,label,sample,rate):
        self.predict(sample)
        self.calc_delta(label)
        self.update_weight(rate)

    def calc_detal(self,label):
        output_nodes = self.layers[-1].nodes
        for i in range(len(label)):
            output_nodes[i].calc_output_layer_delta(label[i])
        for layer in self.layers[-2::-1]:
            for node in layer.nodes:
                node.calc_hidden_layer_delta()

    def update_weight(self,rate):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.update_weight(rate)

    def calc_gradient(self):
        for layer in self.layers[:-1]:
            for node in layer.nodes:
                for conn in node.downstream:
                    conn.calc_gradient()

    def get_gradient(self,label,sample):
        self.predict(sample)
        self.calc_detal(label)
        self.calc_gradient()

    def predict(self,sample):
        self.layers[0].set_output(self.layers)
        for i in range(1,len(self.layers)):
            self.layers[i].calc_output()
        return map(lambda node: node.output,self.layers[:-1].nodes[:-1])

    def dump(self):
        for layer in self.layers:
            layer.dump()


def gradient_check(network,sample_feature,sample_label):
    network_error = lambda vec1,vec2:\
        0.5 * reduce(lambda a,b:a+b,
                     map(lambda v:(v[0]-v[1]) * (v[0]-v[1]),
                     zip(vec1,vec2)))

    network.get_gradient(sample_feature,sample_label)

    for conn in network.connections.connections:
        actual_gradient = conn.get_gradient()
        epsilon = 0.0001
        conn.weight += epsilon
        error1 = network_error(network.predict(sample_feature),sample_label)

        conn.weight -= 2 * epsilon
        error2 = network_error(network.predict(sample_feature),sample_label)

        expected_gradient = (error2-error1)/(2*epsilon)
        print 'expected gradient: \t%f\nactural gradient: \t%f' % (expected_gradient,actual_gradient)



class Loader(object):
    def __init__(self,path,count):
        self.path = path
        self.count = count

    def get_file_count(self):
        f = open(self.path,'rb')
        content = f.read()
        f.close()
        return content

    def to_int(self,byte):
        return struct.unpack('B',byte)[0]


class ImageLoader(Loader):

    def get_picture(self,content,index):
        start = index*28*28+16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].appeend(self.to_int(content[start+i*28+j]))
        return picture

    def get_one_sample(self,picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        content = self.get_file_count()
        data_set = []
        for index in range(self.count):
            data_set.append(self.get_one_sample(self.get_picture(content,index)))
        return data_set



class LabelLoader(Loader):
    def load(self):
        content = self.get_file_count()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index+8]))
        return labels

    def norm(self,label):
        label_vec = []
        label_value = self.to_int(label)
        for i in range(10):
            if i == label_value:
                label_vec.append(0.9)
            else:
                label_vec.append(0.1)
        return label_vec



def get_training_data_set():
    image_loader = ImageLoader('train-images-idx3-ubyte',60000)
    label_loader = ImageLoader('train-labels-idx1-ubyte',60000)
    return image_loader.load(),label_loader.load()

def get_test_data_set():
    image_loader = ImageLoader('train-images-idx3-ubyte', 10000)
    label_loader = ImageLoader('train-labels-idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()

def get_result(vec):
    max_value_index = 0
    max_value = 0
    for i in range(len(vec)):
        if vec[i] > max_value:
            max_value = vec[i]
            max_value_index = i
    return max_value_index

def evalueate(network,test_data_set,test_labels):
    error = 0
    total = len(test_data_set)

    for i in range(total):
        label = get_result(test_labels[i])
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1

    return float(error) / float(total)

def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0

    train_data_set,train_labels = get_training_data_set()
    test_data_set,test_labels = get_test_data_set()
    network = Network([784,300,10])

    while True:
        epoch += 1
        network.train(train_labels,train_data_set,0.3,1)
        print '%s epoch %d finished' % (now(),epoch)

        if epoch % 10 == 0:
            error_ratio =evalueate(network,test_data_set,test_labels)
            print '%s after epoch %d, error ratio is %f' % (now(),epoch,error_ratio)
            if error_ratio > last_error_ratio:
                break
            else:
                last_error_ratio = error_ratio

if __name__ == '__main__':
    train_and_evaluate()




