""" Neural network subset scanning """

from multiprocessing import Pool
import os
import argparse
import numpy as np
from tqdm import tqdm

from util.sampler import Sampler
from util.activation_extractor import KerasActivationExtractor, PytorchActivationExtractor
from util.pvalranges_calculator import PvalueCalculator
from util.custom_functions import CustomFunction
from util.utils import write_hyperparams, scan_write_metrics, customsort

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main(args):
    """ main entrypoint  """


    def get_custom_function(**kwargs):

        if args.customfunction == 'reconstructiondiff':
            return  CustomFunction(**kwargs).reconstruction_error
        elif args.customfunction == 'avgpool':
            return CustomFunction().avg_pool2d
        else:
            return None

    layers = args.layers
    resultsfile = args.resultsfile
    clean_ssize = args.clean_ssize
    anom_ssize = args.anom_ssize
    run = args.run
    restarts = args.restarts
    a_fixed = args.a_fixed

    print("//////////////////////////////")
    print("Loading Model and data sets...")
    print("//////////////////////////////")

    if args.model.endswith('.h5'):
        extractor = KerasActivationExtractor(args.model, layers, args.conditional)
    elif args.model.endswith('.pth') or args.model.endswith('.tar'):
        extractor = PytorchActivationExtractor(args.model, layers, args.modelclass, args.conditional)
    else:
        raise NotImplementedError

    print("//////////////////////////////")
    print("Extracting background activations...")
    print("//////////////////////////////")
    
    customfunction = get_custom_function(data=args.bgddata)
    act, _ = extractor.extract_activation(args.bgddata, labels=args.bgdlabels, customfunction=customfunction)
    act = customsort(act, args.conditional)

    print("//////////////////////////////")
    print("Extracting clean activations...")
    print("//////////////////////////////")

    customfunction = get_custom_function(data=args.cleandata)
    extractor.conditional = False
    clean_act, clean_pred_classes = extractor.extract_activation(args.cleandata, customfunction=customfunction)

    print("//////////////////////////////")
    print("Extracting anomalous activations...")
    print("//////////////////////////////")

    customfunction = get_custom_function(data=args.anomdata)
    anom_act, anom_pred_classes = extractor.extract_activation(args.anomdata, customfunction=customfunction)

    pvalcalculator = PvalueCalculator(act)

    print("//////////////////////////////")
    print("Calculating pvalues...")
    print("//////////////////////////////")
    
    if args.conditional:
        records_pvalue_ranges = pvalcalculator.get_pvalue_ranges_newconditional(clean_act, pvaltest=args.pvaltest)
        anom_records_pvalue_ranges = pvalcalculator.get_pvalue_ranges_newconditional(anom_act, pvaltest=args.pvaltest)

    else:
        records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(clean_act, pvaltest=args.pvaltest)
        anom_records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(anom_act, pvaltest=args.pvaltest)

    assert records_pvalue_ranges is not None
    assert anom_records_pvalue_ranges is not None

    print("//////////////////////////////")
    print("Creating test sets...")
    print("//////////////////////////////")

    # for individual scores, instead of randomly sampling a single image from the set
    # run times we rather score each of the clean/anom images.
    # So for example, runs then defaults to number of clean images when cleansize = 1 and anomssize = 0.

    # This way we are guranteed to see every clean image when generating clean score and
    # every anom image when generating anom scores.

    if anom_ssize == 1 and clean_ssize == 0:
        run = anom_records_pvalue_ranges.shape[0]

    elif clean_ssize == 1 and anom_ssize == 0:
        run = records_pvalue_ranges.shape[0]

    samples, sampled_indices = Sampler.sample(records_pvalue_ranges, anom_records_pvalue_ranges, \
        clean_ssize, anom_ssize, run, conditional=args.conditional)
    
    pool = Pool(processes=5)
    calls = []
    

    for r_indx in range(run):
        
        pred_classes = None
        run_sampled_indices = None
        
        if sampled_indices is not None:
            pred_classes =  np.concatenate((clean_pred_classes[sampled_indices[r_indx][0]], \
                anom_pred_classes[sampled_indices[r_indx][1]]))
            run_sampled_indices = sampled_indices[r_indx]
        
        calls.append(pool.apply_async(scan_write_metrics, [
            samples[r_indx], pred_classes, clean_ssize, anom_ssize, resultsfile, restarts, \
                args.conditional, args.constraint, args.scorefunc, a_fixed, run_sampled_indices ]))

    print("Beginning Scanning...")
    for sample in tqdm(calls):
        sample.get()
    
    print("Writing Hyperparamters to file...")

    write_hyperparams(args, resultsfile)


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()

    DIR_PATH = os.path.dirname(os.path.realpath(__file__))

    RESULTSFILEPATH = DIR_PATH + '/results/dummyresults.out'

    # dummy data. Not actually clean/anom
    BGDDATAPATH = DIR_PATH + '/inputdata/bgd_pth.npy'
    BGDLABELPATH = DIR_PATH + '/inputdata/bgd_labels.npy'
    CLEANDATAPATH = DIR_PATH + '/inputdata/clean_pth.npy'
    ANOMDATAPATH = DIR_PATH + '/inputdata/anom_pth.npy'

    MODELPATH = DIR_PATH + '/models/resnet_cifar10.pth'
    MODELCLASS = 'models.classes.resnet.ResNet34'

    PARSER.add_argument('--customfunction', type=str, default=None,
                        help='custom function after extraction of activation (experimental)')
    PARSER.add_argument('--clean_ssize', type=int, default=85,
                        help='sample size of clean records for evaluation')
    PARSER.add_argument('--anom_ssize', type=int, default=15,
                        help='sample size of anomalous records for evaluation')
    PARSER.add_argument('--bgddata', type=str, default=BGDDATAPATH,
                        help='background records')
    PARSER.add_argument('--bgdlabels', type=str, default=None,
                        help='background labels')
    PARSER.add_argument('--cleandata', type=str, default=CLEANDATAPATH,
                        help='clean records')
    PARSER.add_argument('--anomdata', type=str, default=ANOMDATAPATH,
                        help='anom records')
    PARSER.add_argument('--model', type=str, default=MODELPATH,
                        help='path to the model')
    PARSER.add_argument('--modelclass', type=str, default=MODELCLASS,
                        help='path to model class')
    PARSER.add_argument('--conditional', type=bool, default=False,
                        help='whether or not to compute pvalues \
                            ranges conditioned on each class label')
    PARSER.add_argument('--constraint', type=str, default=None,
                        help='search group')
    PARSER.add_argument('--scorefunc', type=str, default='bj',
                        help='scoring function')
    PARSER.add_argument('--pvaltest', type=str, default='1tail',
                        help='type of test')
    PARSER.add_argument('--layers', nargs='+', default=['layer4'],
                        help='name or index of layer(s) to extract')
    PARSER.add_argument('--run', type=int, default=20,
                        help='number of times to sample and run scan')
    PARSER.add_argument('--restarts', type=int, default=2,
                        help='number of times to perform iterative restart')
    PARSER.add_argument('--resultsfile', type=str, default=RESULTSFILEPATH,
                        help='output file containing results')
    PARSER.add_argument('--a_fixed', type=float, default=-1.0, help ='for running naive alternative with fixed alpha')

    PARSER_ARGS = PARSER.parse_args()
    assert os.path.exists(PARSER_ARGS.bgddata) == 1
    assert os.path.exists(PARSER_ARGS.cleandata) == 1
    assert os.path.exists(PARSER_ARGS.anomdata) == 1
    assert os.path.exists(PARSER_ARGS.model) == 1
    assert PARSER_ARGS.pvaltest in ['1tail', '2tail']
    assert PARSER_ARGS.scorefunc in ['bj', 'hc', 'ks']

    if(PARSER_ARGS.bgdlabels is not None):
        assert len(np.load(PARSER_ARGS.bgddata)) == len(np.load(PARSER_ARGS.bgdlabels))

    main(PARSER_ARGS)
