# Code from: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
import itertools
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(cm, classes,
                          outname,
                          normalize=True,
                          title='Confusion Matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)


    #print(cm)

    """
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    """

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(outname)
    plt.clf()

"""
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
"""


def makeTwoPlot(x, y, z, ylegend, zlegend, title, xlabel, ylabel, xscale, yscale, loc, fontsize, labelsize, tickwidth, ticklength):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  trainPlot, = plt.plot(x, y, 'r.')
  validPlot, = plt.plot(x, z, 'b.')
  plt.legend([trainPlot, validPlot], [ylegend, zlegend], fontsize=labelsize)
  plt.title(title, fontsize=fontsize)
  plt.ylabel(ylabel, fontsize=fontsize)
  plt.xlabel(xlabel, fontsize=fontsize)
  plt.xscale(xscale)
  plt.yscale(yscale)
  plt.tick_params(axis='both', labelsize=labelsize, width=tickwidth, length=ticklength)
  plt.tight_layout()
  plt.savefig(loc)
  plt.clf()

class GraphMaker(object):
    def __init__(self, **kwargs):
        self._kwargs = kwargs
    def makeNPlot(x, ySeriesIterable, yLegendIterable, title, xlabel, ylabel, loc):
        makeNPlot(
                x, ySeriesIterable, yLegendIterable, title, xlabel, ylabel,
                self._kwargs["xscale"], self._kwargs["yscale"], loc,
                self._kwargs["fontsize"], self._kwargs["labelsize"],
                self._kwargs["tickwidth"], self._kwargs["ticklength"])

# TODO make one that takes kwargs... TODO how do with the defaults?

def makeNPlotDefault(
        x, ySeriesIterable, yLegendIterable, title, xlabel, ylabel, loc,
        xscale="linear", yscale="linear", fontsize=10, labelsize=10, 
        tickwidth=1, ticklength=1):
    makeNPlot(x, ySeriesIterable, yLegendIterable, title, xlabel, ylabel,
    xscale, yscale, loc, fontsize, labelsize, tickwidth, ticklength)

def makeStackplotDefault(
        x, ySeriesIterable, yLegendIterable, title, xlabel, ylabel, loc,
        xscale="linear", yscale="linear", fontsize=10, labelsize=10, 
        tickwidth=1, ticklength=1):
    makeStackplot(
            x, ySeriesIterable, yLegendIterable, title, xlabel, ylabel, xscale,
            yscale, loc, fontsize, labelsize, tickwidth, ticklength)

def makeStackplot(x, ySeriesIterable, yLegendIterable, *args):
    labelsize = args[7] # womp womp
    plots = plt.stackplot(x, *ySeriesIterable, labels=yLegendIterable)
    plt.legend(plots, yLegendIterable, fontsize=labelsize)
    _formatAndSave(*args)

def makeNPlot(x, ySeriesIterable, yLegendIterable, *args):
  labelsize = args[7] # womp womp
  plots = []
  for ySeries in ySeriesIterable:
    myplot, = plt.plot(x, ySeries)
    plots.append(myplot)
  plt.legend(plots, yLegendIterable, fontsize=labelsize)
  _formatAndSave(*args)

def _formatAndSave(title, xlabel, ylabel, xscale,
    yscale, loc, fontsize, labelsize, tickwidth, ticklength):
  plt.title(title, fontsize=fontsize)
  plt.ylabel(ylabel, fontsize=fontsize)
  plt.xlabel(xlabel, fontsize=fontsize)
  plt.xscale(xscale)
  plt.yscale(yscale)
  plt.tick_params(axis='both', labelsize=labelsize, width=tickwidth, length=ticklength)
  plt.tight_layout()
  plt.savefig(loc)
  plt.clf()
