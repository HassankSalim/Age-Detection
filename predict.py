import tensorflow as tf
from sys import argv
import numpy as np
import pandas as pd
from glob import glob
import cv2


filenames = glob('data/Pre_test/*.jpg')
map_to_class = {0:'YOUNG', 1:'MIDDLE', 2:'OLD'}

def read(file_name):
    img = cv2.imread(file_name, 1)
    return np.expand_dims(img, axis = 0)

def load_graph(file_name):
    with tf.gfile.GFile(file_name, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph

graph  = load_graph('models/basic_model.pb')
for i in graph.get_operations():
    print(i.name)

X = graph.get_tensor_by_name('prefix/Placeholder:0')
output = graph.get_tensor_by_name('prefix/final_output:0')


with tf.Session(graph = graph) as sess:

    out_frame = []
    for i in filenames:
        temp = {}
        temp['ID'] = i.split('/')[-1]
        out = sess.run(output, feed_dict={ X : read(i) })
        temp['Class'] = map_to_class[out[0]]
        out_frame.append(temp)

submission_frame = pd.DataFrame(out_frame)
submission_frame.to_csv('submission.csv', index = False)
