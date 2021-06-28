"""
SmartVision Facenet module
Author: Avanish Shrestha, 2019
"""

from __future__ import division, print_function, absolute_import

import os
import re
import numpy as np
import tensorflow as tf
  
def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1/std_adj)
    return y

def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files)==0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files)>1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file
    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups())>=2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

class FaceNet:
    def __init__(self, model, input_map=None):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            model_exp = os.path.expanduser(model)
            if (os.path.isfile(model_exp)):
                print('Model filename: %s' % model_exp)
                
                with tf.io.gfile.GFile(model_exp,'rb') as f:
                    graph_def = tf.compat.v1.GraphDef()
                    graph_def.ParseFromString(f.read())
                    tf.import_graph_def(graph_def, input_map=input_map, name='')
            else:
                print('Model directory: %s' % model_exp)
                meta_file, ckpt_file = get_model_filenames(model_exp)
                print('Metagraph file: %s' % meta_file)
                print('Checkpoint file: %s' % ckpt_file)
                saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file), input_map=input_map)
                saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))
        gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options), graph=self.detection_graph)
        self.images_placeholder = self.detection_graph.get_tensor_by_name("input:0")
        self.embeddings = self.detection_graph.get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = self.detection_graph.get_tensor_by_name("phase_train:0")

    def get_embedding(self, face):
        face = prewhiten(face)
        feed_dict = { self.images_placeholder:[face], self.phase_train_placeholder:False }
        emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)
        return emb_array
    
    
 