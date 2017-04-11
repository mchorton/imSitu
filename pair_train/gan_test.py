# This is a (weak) integration test of parzen window.
import unittest
import gan
import torch.nn as nn
import torch.utils.data as td
import torch
import numpy as np

class TestG(nn.Module):
    def __init__(self, insize):
        super(TestG, self).__init__()
        self.inputSize = insize # Needed for generators
        #self.layer = nn.Linear(insize, outsize)
    def forward(self, noise, *args, **kwargs):
        return noise

class TestParzen(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dims = 2
        cls.netg = TestG(cls.dims)
        xdata = torch.rand(100, cls.dims)
        ydata = torch.zeros(100, cls.dims)
        ydata[50:] = torch.ones(50, cls.dims)
        cls.devdata = td.TensorDataset(xdata, ydata)
    def test_parzen(self):
        observed = gan.parzenWindowProb(
                self.netg, self.devdata, nSamples=100, nTestSamples=100)
        for k,v in observed.iteritems():
            mean = np.mean(v)
            # Seems about right for the given cls.dims == 2
            self.assertTrue(0.05 <= mean <= 0.15)
