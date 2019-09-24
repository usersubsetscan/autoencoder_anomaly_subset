import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
from sklearn import metrics
from util.resultparser import ResultParser, ResultSelector
from keras import objectives
import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import seaborn as sns

###### Pytorch functs

def load_data(cifar=True):
    """
    load data and dataloader for pytorch
    SVHN or CIFAR
    """
    transform_test = transforms.Compose([transforms.ToTensor()])
    if cifar:
        testset = dset.CIFAR10(root='./data/cifar-10-batches-py/', train=False,
                               download=True, transform=transform_test)
    else:
        testset = dset.SVHN(root='../data/', split='test',
                            download=True, transform=transform_test)
    
    testloader = data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=2,
                            sampler=data.SubsetRandomSampler(range(8500, 10000)))
    return testloader, testset

def get_noised_success_attacks(attacker, testloader, classifier, target_l):
    """
    return images that are not the targeted class and their attack
    was OK
    @attacker: ART class
    @testloader: Pytorch data loader
    @classifier: ART class
    @target_l: y_label
    """
    all_noised_samples = []
    for batch_idx, (inputs, labels) in enumerate(testloader):
        print('class targeted {}'.format(target_l))
        print('batch {}'.format(batch_idx))
        # generate target labels with shape of the batch
        targeted = np.tile(target_l, (testloader.batch_size, 1))
        x_test_adv_bim = attacker.generate(inputs.numpy(), y=targeted)

        # filter images that already have that target
        no_target = np.not_equal(labels.numpy(), targeted.ravel())
        # get the predicted values over noised images 
        
        x_test_adv_pred = np.argmax(classifier.predict(x_test_adv_bim), axis=1)
        # print('Full prediction size', x_test_adv_pred.shape)
        x_test_adv_pred = x_test_adv_pred[no_target]
        # print('Filtered prediction size', x_test_adv_pred.shape)
        targeted = targeted[no_target]
        # remove noised images that belong to same class as targeted
        x_test_adv_bim = x_test_adv_bim[no_target]
        nb_correct_adv_pred = np.sum(x_test_adv_pred == np.argmax(targeted, axis=1))
        mask_correct = x_test_adv_pred == np.argmax(targeted, axis=1)

        # keep images of successful attacks
        noised_images = x_test_adv_bim[mask_correct]
        
        print("Adversarial test data (first {} images) for BIM".format(x_test_adv_bim.shape[0]))
        print("Targeted Attack Success: {}".format(nb_correct_adv_pred))
        print("Targeted Attack Fail: {}".format(x_test_adv_bim.shape[0] - nb_correct_adv_pred))
        print('*'*10)
        all_noised_samples.append(noised_images)
        
    return all_noised_samples
    

def iterations_test(C, test_loader):
    """
    get the predicted and real values over all
    batchs in test for pytorch models.
    """
    y_real = list()
    y_pred = list()

    for ii, data_ in enumerate(test_loader):
        input_, label = data_
        val_input = Variable(input_).cuda()
        val_label = Variable(label.type(torch.LongTensor)).cuda()
        score = C(val_input)
        _, y_pred_batch = torch.max(score, 1)
        y_pred_batch = y_pred_batch.cpu().squeeze().numpy()
        y_real_batch = val_label.cpu().data.squeeze().numpy()
        y_real.append(y_real_batch.tolist())
        y_pred.append(y_pred_batch.tolist())

    y_real = [item for batch in y_real for item in batch]
    y_pred = [item for batch in y_pred for item in batch]
    
    return y_real, y_pred

#### Visualization functions

def load_results(clean_fn='results/reconstruction_error/mnist_clean_conv2d_7.out',
                 fake_fn='results/reconstruction_error/mnist_bim_conv2d_7.out'):
    resultselector = ResultSelector(score=True)
    a = ResultParser.get_results(clean_fn, resultselector)
    b = ResultParser.get_results(fake_fn, resultselector)
    clean_scores = np.array(a['scores'])
    anom_scores =  np.array(b['scores'])
    
    return clean_scores, anom_scores

def draw_paper_plot_dist_scores(clean_fn='results/reconstruction_error/mnist_clean_conv2d_7.out',
                                fake_fn_bim='results/reconstruction_error/mnist_bim_conv2d_7.out',
                                fake_fn_fgm='results/reconstruction_error/mnist_fg_conv2d_7.out',
                                path='/tmp/both_density_scores.png',
                                dataset='Fashion MNIST'):
    """
    Plot subset scores distributions for clean input, FGM and BIM noised
    data. Save plot with 300 dpi for paper quality in path.
    """
    clean_scores, anom_scores_bim = load_results(clean_fn=clean_fn, 
                                                 fake_fn=fake_fn_bim)
    clean_scores, anom_scores_fg = load_results(clean_fn=clean_fn, 
                                                  fake_fn=fake_fn_fgm)

    plt.title('Distribution of Subset Scores for {}'.format(dataset))
    sns.kdeplot(clean_scores, shade=True, label='clean data')
    sns.kdeplot(anom_scores_fg, shade=True, label='noised data with FGM ($\epsilon = 0.01$)')
    sns.kdeplot(anom_scores_bim, shade=True, label='noised data with BIM ($\epsilon = 0.01$)')
    plt.ylabel('Density')
    plt.xlabel('Subset Score')
    plt.legend() #bbox_to_anchor=(1.1, 1.05))
    #plt.subplots_adjust(right=0.7)
    plt.savefig(path, dpi=300)
    #scores_auc[13].append(roc_auc)
    #print(roc_auc)
    plt.show()

def draw_paper_plot_auc_scores(clean_scores, anom_scores_bim,
                               anom_scores_fg, path='/tmp/AUC_both.png',
                               dataset='Fashion MNIST'):
    """
    Plot ROC curve and AUC for BIM, FGM attacks in path
    """
    
    ### TODO this should be a loop
    y_true = np.append([np.ones(len(anom_scores_bim))], [np.zeros(len(clean_scores))])
    all_scores = np.append([anom_scores_bim], [clean_scores])
    fpr_bim, tpr_bim, thresholds = metrics.roc_curve(y_true, all_scores)
    roc_auc_bim = metrics.auc(fpr_bim, tpr_bim)
    
    y_true = np.append([np.ones(len(anom_scores_fg))], [np.zeros(len(clean_scores))])
    all_scores = np.append([anom_scores_fg], [clean_scores])
    fpr_fgm, tpr_fgm, thresholds = metrics.roc_curve(y_true, all_scores)
    roc_auc_fg = metrics.auc(fpr_fgm, tpr_fgm)

    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operator Characteristic Curve for {}'.format(dataset))
    plt.plot(fpr_bim,tpr_bim, label='ROC curve for BIM attack (area = %0.2f)' % roc_auc_bim, color='seagreen')
    plt.plot(fpr_fgm,tpr_fgm, label='ROC curve for FGM attack (area = %0.2f)' % roc_auc_fg, color='darkorange')
    plt.plot([0, 1], [0, 1], color='cornflowerblue', lw=2, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.02])
    plt.legend(loc="lower right")
    plt.savefig(path, dpi=300)
    plt.show()

#### Visualization functions

def load_results(clean_fn='results/reconstruction_error/mnist_clean_conv2d_7.out',
                 fake_fn='results/reconstruction_error/mnist_bim_conv2d_7.out'):
    resultselector = ResultSelector(score=True)
    a = ResultParser.get_results(clean_fn, resultselector)
    b = ResultParser.get_results(fake_fn, resultselector)
    clean_scores = np.array(a['scores'])
    anom_scores =  np.array(b['scores'])
    
    return clean_scores, anom_scores

#### Draw anom nodes 

def get_anom_nodes(fn='results/reconstruction_error/mnist_bim_conv2d_7.out'):
    """
    Extract from output file the list of nodes that were found anom.
    remove end of line and return an int list of positions.
    """
    nodes_samples = []
    with open(fn, 'r') as f:
        for line in f.readlines()[:-1]:
            # print(line)
            _,_,_,_,_,_, anom_node = line.split(' ')
            nodes = anom_node.strip().split(',')
            nodes = list(map(int, nodes))
            nodes_samples.append(nodes)
        
    return nodes_samples

def draw_anoms(nodes, class_target='', shape=(28, 28)):
    """
    Mark as 1 anomalous nodes in the shape parameter,
    rest of the nodes is mark as 0.
    """
    labels = ['Non anom', 'Anomalous']
    zeros = np.zeros((784))
    np.put(zeros, nodes, 1)
    values = np.unique(zeros.ravel())
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.Set1(range(3,5)), 2)

    plt.figure(figsize=(8,4))
    plt.title('Anom activations in reconstruction space for class {}'.format(class_target))
    im = plt.imshow(zeros.reshape(shape), cmap=cm)
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label="{}".format(labels[i])) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()
    
#### Code from subsetscanning main.py 
class CustomFunction(object):
    """ User provided custom function on activations """
    def __init__(self, data):
        self.data = data

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

def getReconstructionErrFunction(data):
        return  CustomFunction(np.load(data)).reconstruction_error

def draw_anoms(nodes, class_target='', shape=(28, 28)):
    """
    Mark as 1 anomalous nodes in the shape parameter,
    rest of the nodes is mark as 0.
    """
    labels = ['Non anom', 'Anomalous']
    zeros = np.zeros((784))
    np.put(zeros, nodes, 1)
    values = np.unique(zeros.ravel())
    from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    cm = from_list(None, plt.cm.Set1(range(3,5)), 2)

    plt.figure(figsize=(8,4))
    plt.title('Anom activations in reconstruction space for class {}'.format(class_target))
    im = plt.imshow(zeros.reshape(shape), cmap=cm)
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label="{}".format(labels[i])) for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
    plt.show()
    
#### Code from subsetscanning main.py 
class CustomFunction(object):
    """ User provided custom function on activations """
    def __init__(self, data):
        self.data = data

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

def getReconstructionErrFunction(data):
        return  CustomFunction(np.load(data)).reconstruction_error