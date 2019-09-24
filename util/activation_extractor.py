""" Activation extractor """
import collections
from pydoc import locate
from functools import partial

import keras.backend as backend
from keras.models import Sequential
import numpy as np
import six

import torch
import torch.utils.data as tdata
from torch.utils.data import sampler

from util.model_loader import ModelLoader
from util.caching import get_cached_data, write_to_cache

class PytorchActivationExtractor:
    """ Activation values extractor specifc to Pytorch models (.pth format).
    """

    def __init__(self, model, layers, classpath, conditional=None):

        self.model = model
        self.layers = layers
        self.classpath = classpath
        self.conditional = conditional
        self.modelpath = None
        if isinstance(model, str):
            checkpoint = ModelLoader.loadfromfile(model)
            self.modelpath = model
        elif isinstance(model, collections.OrderedDict):
            checkpoint = model
        else:
            raise ValueError

        modelclass = locate(classpath)

        #TODO figure out how to pass the model arguments dynamically
        self.modelinstance = modelclass()

        self.modelinstance.load_state_dict(checkpoint)
        self.modelinstance.eval()
        self.activations = collections.defaultdict(list)

        def save_activation(name, mod, inp, out):
            self.activations[name].append(out.cpu())

        for idx, (name, m) in enumerate(self.modelinstance.named_modules()):
            if name in self.layers or str(idx) in self.layers:
                # partial to assign the layer name to each hook
                m.register_forward_hook(partial(save_activation, name))
 
    
    def extract_activation(self, data, labels=None, customfunction=None):

        datapath = None
        cachekey, cacheddata = get_cached_data(data, self.modelpath, self.layers,  \
            conditional=self.conditional, classpath=self.classpath, customfunction=customfunction)
        if cachekey is not None:
            if isinstance(cacheddata, tuple):
                return cacheddata[0], cacheddata[1]
            else:
                return cacheddata, None

        self.activations = collections.defaultdict(list)
        
        if isinstance(data, str):
            datapath = data
            data = np.load(data)
        elif isinstance(data, np.ndarray):
            data = data

        if isinstance(labels, str):
            labels = np.load(labels)
        elif isinstance(labels, np.ndarray):
            labels = labels
            
        channel = data.shape[3]
        # transpose data if data in nhwc format i.e channel is 3
        #TODO more elegant to assert that the shape of the data based on the input expected by the model

        if channel == 3:
            data = data.transpose(0, 3, 1, 2)

        torchdata = torch.stack([torch.Tensor(i) for i in data])
        
        dataset = tdata.TensorDataset(torchdata)
        dataloader = tdata.DataLoader(dataset, batch_size=100, shuffle=False, num_workers=2, \
            sampler=sampler.SequentialSampler(dataset)) 
        
        y_prob = None

        for idx, inputs in enumerate(dataloader):
            y_pred =  self.modelinstance(inputs[0])
            y_prob_ = np.argmax(y_pred.detach().numpy(), axis=1)
            
            if y_prob is None:
                y_prob = y_prob_
            else:
                y_prob = np.concatenate((y_prob, y_prob_), axis=0)
            # break;
        activations = {name: torch.cat(outputs, 0) for name, outputs in self.activations.items()}
        merged_acts = None

        if(not activations):
            raise ValueError('Could not extract from specified layer')

        for _, act in activations.items():
            act = act.detach().numpy()
            # apply transformation on activations as defined by custom function
            if customfunction is not None and callable(customfunction):
                _acts = customfunction(act)
                act = _acts

            if len(act.shape) == 4:
                act = np.reshape(act, (act.shape[0], act.shape[1] * act.shape[2] * act.shape[3]))
            else:
                act = np.reshape(act, (act.shape[0], act.shape[1]))                
            
            if merged_acts is None:
                merged_acts = act
            else:
                merged_acts = np.concatenate((merged_acts, act), axis=-1)
        
        
        if self.conditional:
            if(labels is not None):
                y_prob = labels
                y_prob = y_prob.reshape((-1,))
                
            unique_labels = set(y_prob)
            conditional_f_acts = {}
            for label in unique_labels:
                indices = [x == label for x in y_prob]
                selected_f_acts =  merged_acts[indices]
                conditional_f_acts[label] = selected_f_acts
            
            cachekey = write_to_cache(conditional_f_acts, datapath, self.modelpath, self.layers, \
                classpath=self.classpath, conditional=self.conditional)
            return conditional_f_acts, None

        else:
            cachekey = write_to_cache((merged_acts, y_prob), datapath, self.modelpath, self.layers, \
                classpath=self.classpath, conditional=self.conditional)
            return merged_acts, y_prob

class KerasActivationExtractor:
    """ Activation values extractor specifc to Keras models (.h5 format).
    """

    def __init__(self, model, layers, conditional):
        
        self.modelpath = None
        if isinstance(model, str):
            self.model = ModelLoader.loadfromfile(model)
            self.modelpath = model
        elif isinstance(model, Sequential):
            self.model = model
        else:
            raise ValueError

        self.layers = layers
        self.conditional = conditional

    def extract_activation(self, data, labels=None, customfunction=None):
        """ Extracts activation values from model defined in the instantiation of this class
        and in layers as specified by the layer name. Asserts that a layer with such name exists.
        :param data: data to pass through the activation layer
        :return ndarray of dim (datasize, total no of activations in layer) if not conditional and
        list of ndarray with the length of the list being the number of unique labels the model
        is trained to classify

        """

        datapath = None

        cachekey, cacheddata = get_cached_data(data, self.modelpath, self.layers, \
            conditional=self.conditional, customfunction=customfunction)
        
        if cachekey is not None:
            if isinstance(cacheddata, tuple):
                return cacheddata[0], cacheddata[1]
            else:
                return cacheddata, None

        layers = self.layers
        if isinstance(data, str):
            datapath = data
            data = np.load(data)
        elif isinstance(data, np.ndarray):
            data = data
            
        if isinstance(labels, str):
            labels = np.load(labels)
        elif isinstance(labels, np.ndarray):
            labels = labels

        a_funcs = KerasActivationExtractor.get_activations(self.model, layers)

        channel = data.shape[1]
        # transpose data if data in nhwc format i.e channel is 3
        #TODO more elegant to assert that the shape of the data based on the input expected by the model

        if channel == 3:
            data = data.transpose(0, 2, 3, 1)
            
        filler = data
        f_acts = None
        for layer in layers:
            f_act = a_funcs[layer]([filler])

            if f_acts is None:
                f_acts = f_act
            else:
                f_acts = np.concatenate((f_acts, f_act), axis=-1)

        
        # apply transformation on activations as defined by custom function
        f_acts = f_acts[0]
        if customfunction is not None and callable(customfunction):
            _f_acts = customfunction(f_acts)
            f_acts = _f_acts

        if len(f_acts.shape) == 4:
            f_acts = np.reshape(f_acts, (f_acts.shape[0], f_acts.shape[1] *
                                            f_acts.shape[2] * f_acts.shape[3]))
        else:  # flattened and denses
            f_acts = np.reshape(
                f_acts, (f_acts.shape[0], f_acts.shape[1]))
        
        
        y_prob = np.argmax(self.model.predict(filler), axis=1)

        if self.conditional:
            if(labels is not None):
                y_prob = labels
                y_prob = y_prob.reshape((-1,))
            else:
                y_prob = np.argmax(self.model.predict(filler), axis=1)

            unique_labels = set(y_prob) 
            conditional_f_acts = {}
            for label in unique_labels:
                indices = [x == label for x in y_prob]
                selected_f_acts =  f_acts[indices]
                conditional_f_acts[label] = selected_f_acts
            
            cachekey = write_to_cache(conditional_f_acts, datapath, self.modelpath, self.layers, conditional=self.conditional)
            return conditional_f_acts, None

        else:
            cachekey = write_to_cache((f_acts, y_prob), datapath, self.modelpath, self.layers, conditional=self.conditional)
            return f_acts, y_prob

    @staticmethod
    def get_layer_idx(model, name):
        """ Get the layer index from its name """
        for idx, layer in enumerate(model.layers):
            if layer.name == name:
                return idx

    @staticmethod
    def get_activations(model, layers):
        """ Get activations based on a bruteforce search of the layers using thier labels
        :param model: keras model to search
        :param layers: list of layers (string or int) to search for
        :return a dictionary of corresponding activations
            (keras backend functions that can be evaluated when passed with input)
        """

        model_activation_functions = {}
        idxes = []
        for layer in layers:
            #get a layer

            if isinstance(layer, int) or layer.isdigit():
                idx = int(layer)
            elif isinstance(layer, six.string_types):
                idx = KerasActivationExtractor.get_layer_idx(model, layer)

            if idx is not None:
                idxes.append(idx)
                model_activation_functions[layer] = backend.function(
                    [model.layers[0].input], [model.layers[idx].output])

        assert len(model_activation_functions) == len(layers)
        assert idxes == sorted(idxes)

        return model_activation_functions
