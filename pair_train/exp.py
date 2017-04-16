import pair_train.gan as gan
import pair_train.nn as pairnn
import utils.methods as mt
import os
import split.splitters as spsp
import split.rp_experiments as rpe
import split.splitters as spsp
import split.rp_experiments as rpe
import utils.mylogger as logging

# This should probably be multiple objects?
# TODO why am I not using zsDev.json anywhere?
class DirConfig(object):
    def __init__(self, basedir="."):
        self._basedir = basedir
        mt.makeDirIfNeeded(self._basedir)
        # Absolute directories
        self.featdir = "data/comp_fc7/"
        self.datadir = self._rebase("data/split/")

        # directories relative to base directory
        self.distdir = self._rebase("data/distance/")
        self.pairdir = self._rebase("data/pairs/")
        self.vrndir = self._rebase("data/vrndata/")
        self.multigandir = self._rebase("multigan/")
        self.multiganlogdir = self._rebase("multiganlogdir/")

        self.trainSetName = os.path.join(self.datadir, "zsTrain.json")
        self.vrnDataName = os.path.join(self.vrndir, "vrnData.json")
        self.pairDataTrain = os.path.join(self.pairdir, "pairtrain.pyt")
        self.pairDataDev = os.path.join(self.pairdir, "pairdev.pyt")
    def _rebase(self, directory):
        return os.path.join(self._basedir, directory)

class DataGenerator(object):
    def __init__(self, dirConfig, test=False):
        self._config = dirConfig
        self._test = test
    def generate(self):
        spsp.splitTrainDevTestMinInTrain(self._config.datadir, test=self._test)
        rpe.generateAllDistVecStyle(
                self._config.distdir,
                self._config.trainSetName)
        # TODO how best to manage these parameters?
        logging.getLogger(__name__).info("vrndir: %s" % self._config.vrndir)
        rpe.getVrnData(
                self._config.distdir, self._config.trainSetName,
                self._config.vrndir, thresh=float('inf'), freqthresh=10,
                blacklistprs = [], bestNounOnly = True, noThreeLabel = True,
                includeWSC=True, noOnlyOneRole=True, strictImgSep=True)
        # Now, get the pairwise gan data.
        # TODO get some metadata / json stuff here!
        pairnn.makeData(
                self._config.pairDataTrain,
                self._config.pairDataDev,
                self._config.featdir,
                self._config.vrnDataName,
                mode="max")

class MultiganExperimentRunner(object):
    def __init__(self):
        pass
    def _run_rerooted(self, func, *args, **kwargs):
        func(*args, **kwargs) # TODO ...
    def generateData(self, dataGenerator):
        self._run_rerooted(dataGenerator.generate)
    def generateGanModels(self, ganTrainer):
        self._run_rerooted(ganTrainer.generate)

class MultiganParameters(object):
    def __init__(self, dirConfig):
        self._config = dirConfig
        self.args = [
                self._config.multigandir,
                self._config.pairDataTrain,
                self._config.multiganlogdir]
        self.kwargs = {
                "epochs": 200,
                "logPer": 3,
                "genDepth": 32,
                "depth": 32,
                "procsPerGpu": 1,
                "lr": 1e-2,
                "lam": 1e-2,
                "minDataPts": 50,
                "decayPer": 10,
                "decayRate": 0.7,
                "batchSize": 32,
                "gdropout": 0.05,
                "useScore": False,
                "style": "trgan"}

class MultiganTrainer(object):
    def __init__(self, parameters):
        self._parameters = parameters
    def generate(self):
        genDataAndTrainIndividualGans(
                *self._parameters.args,
                **self._parameters.kwargs)

def runTestExp():
    dirconfig = DirConfig("data/test_exp/")
    runner = MultiganExperimentRunner()
    runner.generateData(DataGenerator(dirconfig, test=True))

    mp = MultiganParameters(dirconfig)
    mp.kwargs["epochs"] = 2
    mp.kwargs["depth"] = 2
    mp.kwargs["genDepth"] = 2
    mp.kwargs["minDataPoints"] = 5
    mp.kwargs["procsPerGpu"] = 5
    mp.kwargs["lr"] = 1e-5

    runner.generateGanModels(MultiganTrainer(mp))
