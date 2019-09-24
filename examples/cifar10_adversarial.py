import os

import keras
from keras.models import load_model
from keras.datasets import cifar10

from art.utils import load_dataset
from art.classifiers import KerasClassifier
from art.attacks.iterative_method import BasicIterativeMethod
from art.attacks.carlini import CarliniLInfMethod, CarliniL2Method

import numpy as np
from sklearn import metrics


from subsetscanningdetector import SubsetScanningDetector

# Subtracting pixel mean improves accuracy
subtract_pixel_mean = True
num_classes = 10

# Model name, depth and version
#model_type = 'ResNet26v2'

# Load the CIFAR10 data.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data.
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

min_ = np.min(x_train)
max_ = np.max(x_train)

# # ResNet29v2 Model trained for 200 epochs as defined in 
# # https://github.com/keras-team/keras/blob/58399c111a4639526f8d13d4bfa62fc3d0695b02/examples/cifar10_resnet.py

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
save_dir = os.path.join(DIR_PATH, 'saved_models')
model_name = 'cifar10_ResNet29v2_model.196'
filepath = os.path.join(save_dir, model_name + '.h5')

model = load_model(filepath)
classifier = KerasClassifier(model, clip_values=(min_, max_))

epsilon = 0.01
adv_crafter = BasicIterativeMethod(classifier, eps=epsilon, eps_step=0.001)
# adv_crafter = CarliniL2Method(classifier, targeted=False)

# x_train_adv =  adv_crafter.generate(x_train)
x_test_adv =  adv_crafter.generate(x_test[:100])


# Evaluate the classifier on the adversarial samples
preds = np.argmax(classifier.predict(x_test_adv), axis=1)
originalpreds = np.argmax(classifier.predict(x_test[:100]), axis=1)

acc = np.sum(preds == np.argmax(y_test[:100], axis=1)) / y_test.shape[0]
print('Accuracy on adversarial samples: %.2f%%' % (acc * 100))

success_adv_indices = (preds != originalpreds)
success_adv = x_test_adv[success_adv_indices]

np.save(os.path.join(DIR_PATH, 'success_adv_' + model_name + '_cl2'), success_adv)

detector = SubsetScanningDetector(filepath, x_train[:5000], ['average_pooling2d_1'], conditional=True)

# Individual scan 
anomscores, _ = detector.end_end_scan(x_test[-1000:], success_adv, 0, 1, a_fixed=0.1, run=100, score_function='bj')
cleanscores, _ = detector.end_end_scan(x_test[-1000:], success_adv, 1, 0, a_fixed=0.1, run=100, score_function='bj')

y_true = np.append([np.ones(len(anomscores))], [np.zeros(len(cleanscores))])
all_scores = np.append([anomscores], [cleanscores])

fpr, tpr, _ = metrics.roc_curve(y_true, all_scores)
roc_auc = metrics.auc(fpr,tpr)
print("roc_auc:", roc_auc)


#group scan

clean_ssize = 450
anom_ssize = 50

sample_size = clean_ssize + anom_ssize

detector = SubsetScanningDetector(filepath, x_train[:3000], ['average_pooling2d_1'], conditional=False)

anomscores, image_sub = detector.end_end_scan(x_test[-1000:], success_adv, clean_ssize, anom_ssize, run=100, score_function='hc')
cleanscores, _ = detector.end_end_scan(x_test[-1000:], success_adv, sample_size, 0, run=100, score_function='hc')

y_true = np.append([np.ones(len(anomscores))], [np.zeros(len(cleanscores))])
all_scores = np.append([anomscores], [cleanscores])

fpr, tpr, _ = metrics.roc_curve(y_true, all_scores)
roc_auc = metrics.auc(fpr,tpr)

print("roc_auc:", roc_auc)
print(image_sub)

intersection = list(set(range(clean_ssize, sample_size)) & set(image_sub.tolist()))
intersection_size = float(len(intersection))

precision = intersection_size/len(image_sub)
recall = intersection_size/anom_ssize
print("prec-recall:", precision, recall)
