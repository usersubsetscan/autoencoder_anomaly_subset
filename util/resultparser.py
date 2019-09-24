""" Result parser """
import os


class ResultSelector:

    """ Result selector """
    def __init__(self, score=False, precision=False, recall=False, nodesub=False, imagesub=False, optimal_alpha = False, len_nodesub=False, len_imagesub=False):
        
        self.score = score
        self.precision = precision
        self.recall = recall
        self.nodesub = nodesub
        self.imagesub = imagesub
        self.optimal_alpha = optimal_alpha
        self.len_nodesub = len_nodesub
        self.len_imagesub = len_imagesub
        

class ResultParser:
    """ Result parser  """

    @staticmethod
    def get_results(filename, selected_results):
        """ Get results from file based on specification of resultselector """
        assert os.path.exists(filename) == 1
        results  = {'scores': [], 'precisions': [], 'recalls': [], 'image_sub_lengths': [], 'node_sub_lengths':[], 'node_subs': [], 'optimal_alphas': [] }
        with open(filename, 'r') as ins:

            for line in ins:
                if 'hyperparams' in line:
                    continue
                
                score, precision, recall, len_image_sub, len_node_sub, optimal_alpha, node_subs, image_subs = line.split(' ')
                
                if selected_results.score:
                    results['scores'].append(float(score))

                if selected_results.precision:
                    results['precisions'].append(float(precision))

                if selected_results.recall:
                    results['recalls'].append(float(recall))

                if selected_results.nodesub:
                    results['node_subs'].append(node_subs)
                
                if selected_results.imagesub:
                    results['image_subs'].append(image_subs)

                if selected_results.optimal_alpha:
                    results['optimal_alphas'].append(float(optimal_alpha))
                    
                if selected_results.len_nodesub:
                    results['node_sub_lengths'].append(int(len_node_sub))

                if selected_results.len_imagesub:
                    results['image_sub_lengths'].append(int(len_image_sub))

        return results
