""" Sets of customfunctions applied on activations before pvalule calculation """
import numpy as np
from keras import objectives
import tensorflow as tf
import torch

class CustomFunction:
    """ User provided custom function on activations """
    def __init__(self, **kwargs):
        if 'data' in kwargs.keys():
            data = kwargs.get('data')
            if isinstance(data, str):
                self.data = np.load(data)
            elif isinstance(data, np.ndarray):
                self.data = data

    def avg_pool2d(self, out):
        import torch.nn.functional as F
        out = F.avg_pool2d(torch.from_numpy(out), 4)
        return out.detach().numpy()

    def reconstruction_error(self, y_pred):
        """ auto encoder reconstruction error """
        y_pred = tf.convert_to_tensor(y_pred)
        rec_err = objectives.binary_crossentropy(self.data, y_pred)
        eval_rec_err = None
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            eval_rec_err = rec_err.eval()

        eval_rec_err = np.reshape(eval_rec_err, (eval_rec_err.shape[0], eval_rec_err.shape[1], eval_rec_err.shape[2], 1))
        return eval_rec_err

