import os
import errno
import hashlib
import pickle

def get_cached_pvals(bgdcachekey, evalcachekey):

    cachekey = hashlib.md5(str.encode(bgdcachekey +  evalcachekey)).hexdigest()
    picklefile = cachekey + '.obj'

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = dir_path + '/inputdata/.cache/'

    if os.path.exists(dir_path + picklefile) != 1:
        return None

    data = None

    with open(dir_path + picklefile, 'rb') as fp:
        data = pickle.load(fp)

        if isinstance(data, tuple):
            data = data[0]

    return data

def write_pvals_to_cache(pvalues, bgdcachekey, evalcachekey):
    cachekey = hashlib.md5(str.encode(bgdcachekey +  evalcachekey)).hexdigest()
    picklefile = cachekey + '.obj'

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = dir_path + '/inputdata/.cache/'

    with open(dir_path + picklefile, 'wb') as pickle_file:
        pickle.dump(pvalues, pickle_file)


def write_to_cache(activations, data, model, layers, conditional=False, classpath=None):
    
    if not isinstance(data, str):
        return None

    print("writing to cache")
    confighash = get_confighash(data, model, layers, conditional, classpath)
    picklefile = confighash + '.obj'

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = dir_path + '/inputdata/.cache/'

    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            
        except OSError as exc: 
            if exc.errno != errno.EEXIST:
                raise
            
    with open(dir_path + picklefile, 'wb') as pickle_file:
        pickle.dump(activations, pickle_file)

    return confighash

def get_cached_data(data, model, layers, conditional=False, \
     classpath=None, customfunction=None):
    """ Check if the activation data exists already in the cache """

    if not isinstance(model, str) or not isinstance(data, str):
        return None, None
    if customfunction is not None:
        return None, None
    
    assert os.path.exists(data) == 1
    assert os.path.exists(model) == 1
    
    confighash = get_confighash(data, model, layers, conditional, classpath)

    picklefile = confighash + '.obj'

    dir_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    dir_path = dir_path + '/inputdata/.cache/'

    if os.path.exists(dir_path + picklefile) != 1:
        return None, None

    data = None
    with open(dir_path + picklefile, 'rb') as fp:
        data = pickle.load(fp)

    print("is cached")
    return confighash, data

def get_confighash(data, model, layers, conditional=False, classpath=None):
    
    layers_str = ",".join(str(x) for x in layers)
    classpath_str = (classpath, "")[classpath is None]

    data_mt_str =  str(os.path.getmtime(data))
    model_mt_str =  str(os.path.getmtime(model))

    config = data_mt_str + model_mt_str + layers_str + str(conditional) + classpath_str
    confighash = hashlib.md5(str.encode(config)).hexdigest()

    return confighash


