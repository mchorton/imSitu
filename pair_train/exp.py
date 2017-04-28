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
import data as dataman

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
    def __init__(self, dirConfig, test=False, mode="max"):
        self._config = dirConfig
        self._test = test
        self._mode = mode
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
                mode=self._mode)

class PhpGenerator(object):
    def __init__(self, dirconfig):
        self._dirconfig = dirconfig
    def generate(self):
        htmlMaker = html.HtmlMaker()
        htmlMaker.addTitle("Experiment Dashboard")
        htmlMaker.addElement(html.Heading(1, "Experiment Dashboard"))

        for indexfile in self.getIndexFiles2():
            link = "/%s" % indexfile
            htmlMaker.addElement(
                    html.HRef(link, link))
            htmlMaker.addElement(html.Paragraph("\n"))
        htmlMaker.save(os.path.join(self._dirconfig.basedir, "index.php"))
    def getIndexFiles(self):
        # This is how I wanted to do things, but the files aren't ready yet
        # So, I'm using a different version of this func.
        ret = []
        for dirpath, dirnames, filenames in os.walk(self._dirconfig.basedir):
            for filename in it.ifilter(lambda x: x.endswith(".php"), filenames):
                ret.append(os.path.join(
                        self._dirconfig.basedir, os.path.relpath(
                                os.path.join(dirpath, filename),
                                self._dirconfig.basedir)))
        return ret
    def getIndexFiles2(self):
        return [
                self._dirconfig.pairdir,
                self._dirconfig.vrndir,
                self._dirconfig.multiganlogdir]

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
                "gUpdates": 1,
                "dUpdates": 1,
                "dUpdates": 1,
                "logPer": 3,
                "genDepth": 32,
                "depth": 32,
                "procsPerGpu": 1,
                "lr": 1e-2,
                "lam": 1e-2,
                "ignoreCond": False,
                "minDataPts": 50,
                "decayPer": 10,
                "trgandnoImg": False,
                "decayRate": 0.7,
                "batchSize": 32,
                "hiddenSize": 1024,
                "gdropout": 0.05,
                "graphPerIter": 1e10,
                "useScore": False,
                "nz": 0,
                "activeGpus": [0, 1, 2, 3],
                "style": "trgan"}

class MultiganTrainer(object):
    def __init__(self, parameters):
        self._parameters = parameters
    def generate(self):
        gan.genDataAndTrainIndividualGans(
                *self._parameters.args,
                **self._parameters.kwargs)

def runTestExp():
    expdir = "data/test_exp/"
    if os.path.exists(expdir):
        import shutil
        shutil.rmtree(expdir)
    dirconfig = DirConfig(expdir)

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    runner.generateData(DataGenerator(dirconfig, test=True))

    mp = MultiganParameters(dirconfig)
    mp.kwargs["epochs"] = 2
    mp.kwargs["logPer"] = 1
    mp.kwargs["depth"] = 2
    mp.kwargs["genDepth"] = 2
    mp.kwargs["minDataPts"] = 3
    mp.kwargs["measurePerf"] = True
    mp.kwargs["procsPerGpu"] = 5
    mp.kwargs["graphPerIter"] = 1
    mp.kwargs["lr"] = 1e-5
    mp.kwargs["seqOverride"] = False

    runner.generateGanModels(MultiganTrainer(mp))

def runPartialTestExp(expdir="data/test_exp/", mode="max"):
    """
    if os.path.exists("data/test_exp/"):
        import shutil
        shutil.rmtree("data/test_exp/")
    """
    dirconfig = DirConfig(expdir, False)
    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    runner = MultiganExperimentRunner()
    #runner.generateData(DataGenerator(dirconfig, test=True, mode=mode))

    #datasetDir = os.path.join(dirconfig.multigandir, "nounpair_data/")
    #pdm = dataman.PairDataManager(dirconfig.pairdir, dirconfig.featdir)
    #dataman.shardAndSave(pdm, datasetDir, minDataPts=0)

    mp = MultiganParameters(dirconfig)
    mp.kwargs["epochs"] = 20
    mp.kwargs["logPer"] = 1
    mp.kwargs["logDevPer"] = 3
    mp.kwargs["depth"] = 2
    mp.kwargs["genDepth"] = 2
    mp.kwargs["minDataPts"] = 3
    mp.kwargs["onlyN"] = 3
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["seqOverride"] = True
    mp.kwargs["nSamples"] = 1000
    mp.kwargs["measurePerf"] = False
    mp.kwargs["activeGpus"] = [0, 1, 2, 3]
    mp.kwargs["updateAblationPer"] = 10
    mp.kwargs["updateParzenPer"] = 10
    mp.kwargs["procsPerGpu"] = 1
    mp.kwargs["graphPerIter"] = 20

    runner.generateGanModels(MultiganTrainer(mp))
    """
    """

def runDefaultExp():
    dirconfig = DirConfig("data/manygan_default/")
    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    runner.generateData(DataGenerator(dirconfig))

    mp = MultiganParameters(dirconfig)
    runner.generateGanModels(MultiganTrainer(mp))

def runLowlearn():
    dirconfig = DirConfig("data/manygan_lowlearn/", cautious=True)

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    runner.generateData(DataGenerator(dirconfig))

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["bw_method"] = 2 ** 18
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig))

def runLowlearnNice():
    dirconfig = DirConfig("data/manygan_lowlearn_nice/", cautious=True)

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    runner.generateData(DataGenerator(dirconfig, test=False))

    mp = MultiganParameters(dirconfig)
    # TODO too low?
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["bw_method"] = 2 ** 18
    mp.kwargs["updateAblationPer"] = 15
    mp.kwargs["updateParzenPer"] = 10
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig))

def runLowlearnNice2():
    dirconfig = DirConfig("data/manygan_lowlearn_nice_3/", cautious=False)

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    #runner.generateData(DataGenerator(dirconfig, test=False))

    mp = MultiganParameters(dirconfig)
    # TODO too low?
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["bw_method"] = 2 ** 18
    mp.kwargs["decayPer"] = 100
    mp.kwargs["epochs"] = 1000
    mp.kwargs["updateAblationPer"] = 150
    mp.kwargs["updateParzenPer"] = 100
    mp.kwargs["nSamples"] = 50 # TODO could be larger...
    mp.kwargs["nTestSamples"] = 10 # TODO could be larger...
    mp.kwargs["procsPerGpu"] = 2 # TODO can be higher
    mp.kwargs["depth"] = 32 # TODO
    mp.kwargs["genDepth"] = 32
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 1
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig))

def runProfile(cautious=True):
    dirconfig = DirConfig("data/profile/", cautious)

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    #runner.generateData(DataGenerator(dirconfig, test=False))

    mp = MultiganParameters(dirconfig)
    # TODO too low?
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["bw_method"] = 2 ** 18
    mp.kwargs["decayPer"] = 100
    mp.kwargs["decayRate"] = 0.7
    mp.kwargs["epochs"] = 2000
    mp.kwargs["updateAblationPer"] = 50
    mp.kwargs["updateParzenPer"] = 50
    mp.kwargs["nSamples"] = 50 # TODO could be larger...
    mp.kwargs["nTestSamples"] = 10 # TODO could be larger...
    mp.kwargs["procsPerGpu"] = 1 # TODO can be higher
    mp.kwargs["depth"] = 32 # TODO
    mp.kwargs["genDepth"] = 32
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 1
    mp.kwargs["seqOverride"] = False
    mp.kwargs["lam"] = 1e-2
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["minDataPts"] = 50
    #mp.kwargs["onlyN"] = 10 # TODO remove this
    mp.kwargs["graphPerIter"] = 300
    mp.kwargs["measurePerf"] = False
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig))

def smalltest(cautious=True, seqOverride=False):
    dirconfig = DirConfig("data/smalltest3/", cautious)

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    #runner.generateData(DataGenerator(dirconfig, test=False))

    mp = MultiganParameters(dirconfig)
    # TODO too low?
    mp.kwargs["lr"] = 1e-3
    mp.kwargs["bw_method"] = 2 ** 18 # TODO 
    mp.kwargs["decayPer"] = 100
    mp.kwargs["decayRate"] = 0.7
    mp.kwargs["epochs"] = 1500
    mp.kwargs["updateAblationPer"] = 50
    mp.kwargs["updateParzenPer"] = 50
    mp.kwargs["nSamples"] = 50 # TODO could be larger...
    mp.kwargs["nTestSamples"] = 10 # TODO could be larger...
    mp.kwargs["procsPerGpu"] = 4 # TODO can be higher
    mp.kwargs["depth"] = 1 # TODO
    mp.kwargs["genDepth"] = 1
    mp.kwargs["skipShardAndSave"] = True
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 5
    mp.kwargs["logDevPer"] = 10
    mp.kwargs["seqOverride"] = seqOverride
    mp.kwargs["lam"] = 1
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["minDataPts"] = 50
    mp.kwargs["graphPerIter"] = 500
    mp.kwargs["measurePerf"] = False
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig))

if __name__ == '__main__':
    runLowlearnNice2()
