import pair_train.gan as gan
import pair_train.nn as pairnn
import utils.methods as mt
import os
import split.splitters as spsp
import split.rp_experiments as rpe
import split.splitters as spsp
import split.rp_experiments as rpe
import utils.mylogger as logging
import split.v2pos.htmlgen as html
import itertools as it

# This should probably be multiple objects?
# TODO why am I not using zsDev.json anywhere?
class DirConfig(object):
    def __init__(self, basedir=".", cautious=True):
        self.basedir = basedir
        if cautious and os.path.exists(self.basedir):
            raise ValueError(
                    "Base directory '%s' already exists. Cowardly exiting "
                    "to avoid overwriting experiment" % self.basedir)
        mt.makeDirIfNeeded(self.basedir)
        # Absolute directories
        self.featdir = "data/comp_fc7/"
        self.splitdir = "splits/"

        # directories relative to base directory
        self.localsplitdir = self._rebase("data/split/")
        self.distdir = self._rebase("data/distance/")
        self.pairdir = self._rebase("data/pairs/")
        self.vrndir = self._rebase("data/vrndata/")
        self.multigandir = self._rebase("multigan/")
        self.multiganlogdir = self._rebase("multiganlogdir/")

        self.trainSetName = os.path.join(self.localsplitdir, "zsTrain.json")
        self.vrnDataName = os.path.join(self.vrndir, "vrnData.json")
        # TODO these should be automatic
        self.pairDataTrain = os.path.join(self.pairdir, "pairtrain.pyt")
        self.pairDataDev = os.path.join(self.pairdir, "pairdev.pyt")
    def _rebase(self, directory):
        return os.path.join(self.basedir, directory)

class DataGenerator(object):
    def __init__(self, dirConfig, test=False):
        self._config = dirConfig
        self._test = test
    def generate(self):
        spsp.copyDataSplit(
                self._config.localsplitdir, self._config.splitdir,
                test=self._test)
        rpe.generateAllDistVecStyle(
                self._config.distdir,
                self._config.trainSetName)
        logging.getLogger(__name__).info("vrndir: %s" % self._config.vrndir)
        rpe.getVrnData(
                self._config.distdir, self._config.trainSetName,
                self._config.vrndir, thresh=float('inf'), freqthresh=10,
                blacklistprs = [], bestNounOnly = True, noThreeLabel = True,
                includeWSC=True, noOnlyOneRole=True, strictImgSep=True)
        pairnn.makeDataNice(
                self._config.pairDataTrain,
                self._config.pairDataDev,
                self._config.featdir,
                self._config.vrnDataName,
                mode="max")

class PhpGenerator(object):
    def __init__(self, rootDir):
        self._rootDir = rootDir
    def generate(self):
        htmlMaker = html.HtmlMaker()
        htmlMaker.addTitle("Experiment Dashboard")
        htmlMaker.addElement(html.Heading(1, "Experiment Dashboard"))
        for dirpath, dirnames, filenames in os.walk(self._rootDir):
            for filename in it.ifilter(lambda x: x.endswith(".php"), filenames):
                link = "/%s" % os.path.join(
                        self._rootDir, os.path.relpath(
                                os.path.join(dirpath, filename), self._rootDir))
                htmlMaker.addElement(
                        html.HRef(link, link))
                htmlMaker.addElement(html.Paragraph("\n"))
        htmlMaker.save(os.path.join(self._rootDir, "index.php"))


class MultiganExperimentRunner(object):
    def __init__(self):
        pass
    def _run_rerooted(self, func, *args, **kwargs):
        func(*args, **kwargs)
    def generateData(self, dataGenerator):
        self._run_rerooted(dataGenerator.generate)
    def generateGanModels(self, ganTrainer):
        self._run_rerooted(ganTrainer.generate)
    def generatePhpDirectory(self, phpGenerator):
        phpGenerator.generate()

class MultiganParameters(object):
    def __init__(self, dirConfig):
        self._config = dirConfig
        self.args = [
                self._config.multigandir,
                self._config.pairDataTrain,
                self._config.multiganlogdir,
                self._config.pairdir,
                self._config.featdir]
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
        gan.genDataAndTrainIndividualGans(
                *self._parameters.args,
                **self._parameters.kwargs)

def runTestExp():
    if os.path.exists("data/test_exp/"):
        import shutil
        shutil.rmtree("data/test_exp/")
    dirconfig = DirConfig("data/test_exp/")
    runner = MultiganExperimentRunner()
    runner.generateData(DataGenerator(dirconfig, test=True))

    mp = MultiganParameters(dirconfig)
    mp.kwargs["epochs"] = 2
    mp.kwargs["logPer"] = 1
    mp.kwargs["depth"] = 2
    mp.kwargs["genDepth"] = 2
    mp.kwargs["minDataPts"] = 3
    mp.kwargs["procsPerGpu"] = 5
    mp.kwargs["lr"] = 1e-5
    mp.kwargs["seqOverride"] = False

    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig.basedir))

# TODO why do "mode='all'" and "mode='max'" produce diff numbers of pairs?
def runDefaultExp():
    dirconfig = DirConfig("data/manygan_default/")
    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig.basedir))
    runner.generateData(DataGenerator(dirconfig))

    mp = MultiganParameters(dirconfig)
    runner.generateGanModels(MultiganTrainer(mp))

def runLowlearn():
    dirconfig = DirConfig("data/manygan_lowlearn/", cautious=False)

    runner = MultiganExperimentRunner()
    """
    runner.generatePhpDirectory(PhpGenerator(dirconfig.basedir))
    runner.generateData(DataGenerator(dirconfig))
    """

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-5
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig.basedir))
