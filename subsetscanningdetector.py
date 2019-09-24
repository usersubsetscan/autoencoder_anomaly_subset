import numpy as np
from tqdm import tqdm

from util.activation_extractor import KerasActivationExtractor, PytorchActivationExtractor
from util.pvalranges_calculator import PvalueCalculator
from util.sampler import Sampler

from util.utils import customsort, scan

class SubsetScanningDetector:

    def __init__(self, model, bgd_data, layers, modelclass=None, bgd_labels=None, conditional=False, \
         customfunction=None, pvaltest='1tail'):

        self.model = model
        self.bgd_data = bgd_data
        self.layers = layers
        self.bgd_labels = bgd_labels
        self.conditional = conditional
        self.customfunction = customfunction
        self.pvaltest = pvaltest

        print("//////////////////////////////")
        print("Loading Model and data sets...")
        print("//////////////////////////////")

        if model.endswith('.h5'):
            self.extractor = KerasActivationExtractor(model, layers, conditional)
        elif model.endswith('.pth') or model.endswith('.tar'):
            self.extractor = PytorchActivationExtractor(model, layers, modelclass, conditional)
        else:
            raise NotImplementedError
        
    def end_end_scan(self, clean_data, anom_data, clean_ssize, anom_ssize, run=2, \
        score_function='bj', a_fixed=-1.0, cleancustomfunction=None,\
             anomcustomfunction=None, constraint=None, write_to_file = False):

        print("//////////////////////////////")
        print("Extracting background activations...")
        print("//////////////////////////////")
        self.extractor.conditional = self.conditional
        act, _ = self.extractor.extract_activation(self.bgd_data, labels=self.bgd_labels,\
             customfunction=self.customfunction)
             
        act = customsort(act, self.conditional)

        print("//////////////////////////////")
        print("Extracting clean activations...")
        print("//////////////////////////////")

        self.extractor.conditional = False
        clean_act, clean_pred_classes = self.extractor.extract_activation(clean_data, customfunction=cleancustomfunction)

        print("//////////////////////////////")
        print("Extracting anomalous activations...")
        print("//////////////////////////////")

        anom_act, anom_pred_classes = self.extractor.extract_activation(anom_data, customfunction=anomcustomfunction)


        pvalcalculator = PvalueCalculator(act)

        print("//////////////////////////////")
        print("Calculating pvalues...")
        print("//////////////////////////////")
        
        if self.conditional:
            records_pvalue_ranges = pvalcalculator.get_pvalue_ranges_newconditional(clean_act, pvaltest=self.pvaltest)
            anom_records_pvalue_ranges = pvalcalculator.get_pvalue_ranges_newconditional(anom_act, pvaltest=self.pvaltest)

        else:
            records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(clean_act, pvaltest=self.pvaltest)
            anom_records_pvalue_ranges = pvalcalculator.get_pvalue_ranges(anom_act, pvaltest=self.pvaltest)


        if anom_ssize == 1 and clean_ssize == 0:
            run = anom_records_pvalue_ranges.shape[0]

        elif clean_ssize == 1 and anom_ssize == 0:
            run = records_pvalue_ranges.shape[0]

        print("run:", run)

        samples, sampled_indices = Sampler.sample(records_pvalue_ranges, anom_records_pvalue_ranges, \
            clean_ssize, anom_ssize, run, conditional=self.conditional)

        bestscores = []
        for r_indx in tqdm(range(run)):
            
            pred_classes = None
            if sampled_indices is not None:
                pred_classes =  np.concatenate((clean_pred_classes[sampled_indices[r_indx][0]], \
                    anom_pred_classes[sampled_indices[r_indx][1]]))
            
            best_score, image_sub, node_sub, optimal_alpha = scan(samples[r_indx],
                pred_classes, clean_ssize, anom_ssize, 2, conditional=self.conditional,
                    constraint=constraint, score_function = score_function, a_fixed=a_fixed,
                    sampled_indices = sampled_indices)
            
            bestscores.append(best_score)

        return bestscores, image_sub