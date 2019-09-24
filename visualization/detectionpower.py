""" Detection power visualization """

import os
import argparse
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

from util.resultparser import ResultParser, ResultSelector

def plot(cleanscores, anomscores):
    """ plot and calculate the detection power """
    
    resultselector = ResultSelector(score=True)
    cleanres = ResultParser.get_results(cleanscores, resultselector)
    anomres = ResultParser.get_results(anomscores, resultselector)

    clean_scores = np.array(cleanres['scores'])
    anom_scores =  np.array(anomres['scores'])

    plt.hist(clean_scores, histtype = 'step')
    plt.hist(anom_scores, histtype = 'step')
    plt.show()

    y_true = np.append([np.ones(len(anom_scores))], [np.zeros(len(clean_scores))])
    all_scores = np.append([anom_scores], [clean_scores])

    fpr, tpr, _ = metrics.roc_curve(y_true, all_scores)
    roc_auc = metrics.auc(fpr,tpr)
    plt.plot(fpr,tpr)
    plt.show()
    print(roc_auc)

if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    DIR_PATH = os.path.dirname(os.path.realpath(__file__))
    PARENT_DIR_PATH = os.path.abspath(DIR_PATH + "/../")
    

    PARSER.add_argument('--cleanscores', type=str, default=PARENT_DIR_PATH+ '/results/clean_individ_act_19.out',
                        help='clean scores file path')
    PARSER.add_argument('--anomscores', type=str, default=PARENT_DIR_PATH+ '/results/bim_02_targ0_individ_act_19.out',
                        help='anomalous scores file path')

    PARSER_ARGS = PARSER.parse_args()
    assert os.path.exists(PARSER_ARGS.cleanscores) == 1
    assert os.path.exists(PARSER_ARGS.anomscores) == 1

    plot(PARSER_ARGS.cleanscores, PARSER_ARGS.anomscores)
