import os
from os.path import join
from collections import defaultdict
import json

import numpy as np

import pair_train.gan as gan
import pair_train.nn as pairnn
import utils.methods as mt
import split.splitters as spsp
import split.rp_experiments as rpe
import split.splitters as spsp
import split.rp_experiments as rpe
import utils.mylogger as logging
import split.v2pos.htmlgen as html
import itertools as it
import data as dataman
import generate_samples as gs
import torch

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
        self.chimdir = self._rebase("chimeras/")

        self.trainSetName = os.path.join(self.localsplitdir, "zsTrain.json")
        self.vrnDataName = os.path.join(self.vrndir, "vrnData.json")
        self.heldoutVrnDataName = os.path.join(
                self.vrndir, "_heldout_vrnData.json")
        # TODO these should be automatic
        self.pairDataTrain = os.path.join(self.pairdir, "pairtrain.pyt")
        self.pairDataDev = os.path.join(self.pairdir, "pairdev.pyt")
        self.chim_base_name = join(self.chimdir, "chimeras")
        self.heldback_img_names = join(self.vrndir, "heldback.json")

    def _rebase(self, directory):
        return os.path.join(self.basedir, directory)

class DataGenerator(object):
    def __init__(self, parameters, test=False):
        self._parameters = parameters
        self._test = test
    @property
    def _config(self):
        return self._parameters.dirconfig
    @property
    def _kwargs(self):
        return self._parameters.kwargs
    def generate(self, **kwargs):
        spsp.copyDataSplit(
                self._config.localsplitdir, self._config.splitdir,
                test=self._test)
        if not self._kwargs["skip_generate_dists"]:
            rpe.generateAllDistVecStyle(
                    self._config.distdir,
                    self._config.trainSetName,
                    self._kwargs["diststyle"])
        logging.getLogger(__name__).info("vrndir: %s" % self._config.vrndir)
        rpe.getVrnData(
                self._config.distdir, self._config.trainSetName,
                self._config.vrndir, thresh=float('inf'), freqthresh=10,
                blacklistprs = [], bestNounOnly = True, noThreeLabel = True,
                includeWSC=True, noOnlyOneRole=True, strictImgSep=True,
                held_out_pairs=self._kwargs["held_out_pairs"])
        pairnn.makeDataNice(
                self._config.pairDataTrain,
                self._config.pairDataDev,
                self._config.featdir,
                self._config.vrnDataName,
                mode=self._kwargs["mode"])

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
    def __init__(self, parameters, test=False):
        self._parameters = parameters
        self._test = test
    def run(self):
        PhpGenerator(self._parameters.dirconfig).generate()
        if not self._parameters.kwargs.get("skip_data_generator", None):
            DataGenerator(
                    self._parameters,
                    test=self._test).generate()
        else:
            logging.getLogger(__name__).info("Skipping data generation")
        MultiganTrainer(self._parameters).generate()
    def generateData(self, dataGenerator):
        self._run_rerooted(dataGenerator.generate)
    def generateGanModels(self, ganTrainer):
        self._run_rerooted(ganTrainer.generate)
    def generatePhpDirectory(self, phpGenerator):
        phpGenerator.generate()

class MultiganParameters(object):
    def __init__(self, dirconfig):
        self.dirconfig = dirconfig
        self.args = [
                self.dirconfig.multigandir,
                self.dirconfig.pairDataTrain,
                self.dirconfig.multiganlogdir,
                self.dirconfig.pairdir,
                self.dirconfig.featdir]
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
                "losstype": "log",
                "ignoreCond": False,
                "minDataPts": 50,
                "decayPer": 10,
                "trgandnoImg": False,
                "decayRate": 0.7,
                "batchSize": 32,
                "hiddenSize": 1024,
                "gdropout": 0.05,
                "held_out_pairs": {},
                "graphPerIter": 1e10,
                "bw_method": 2 ** 6,
                "useScore": False,
                "nz": 0,
                "mode": "max",
                "updateRankPer": 40,
                "skip_generate_dists": False,
                "diststyle": "",
                "nn_sampled_pts": 20,
                "nn_considered": int(1e9),
                "activeGpus": [0, 1, 2, 3],
                "noisy_labels": True,
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
    mp.kwargs["skip_generate_dists"] = False
    mp.kwargs["held_out_pairs"] = {
            ("n04256520", "n02818832"): 0.5, # sofa -> bed
            ("n02374451", "n02402425"): 0.5, # horse -> cattle
            ("n02084071", "n02121620"): 0.5, # dog -> cat
            ("n02958343", "n04490091"): 0.5, # car -> truck
            ("n02958343", "n02930766"): 0.5, # car -> cab
            ("n01503061", "n01605630"): 0.5, # bird -> hawk
            ("n03948459", "n04090263"): 0.5, # pistol -> rifle
            ("n07747607", "n07749582"): 0.5, # orange -> lemon
            ("n11669921", "n13104059"): 0.5, # flower -> tree
            ("n08613733", "n08438533"): 0.5, # outdoors -> forest
        }

    runner = MultiganExperimentRunner(mp, test=True)
    runner.run()

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
    mp.kwargs["losstype"] = "square"

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

def weakfull(cautious=True, **kwargs):
    dirconfig = DirConfig("data/weakfull/", cautious)

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    #runner.generateData(DataGenerator(dirconfig, test=False))

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-3
    mp.kwargs["decayPer"] = 100
    mp.kwargs["decayRate"] = 0.7
    mp.kwargs["epochs"] = 1500
    mp.kwargs["updateAblationPer"] = 50
    mp.kwargs["updateParzenPer"] = 1
    mp.kwargs["nSamples"] = 50
    mp.kwargs["nTestSamples"] = 10
    mp.kwargs["updateRankPer"] = 1
    mp.kwargs["procsPerGpu"] = 4
    mp.kwargs["depth"] = 1
    mp.kwargs["genDepth"] = 1
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 5
    mp.kwargs["dUpdates"] = 1
    mp.kwargs["logDevPer"] = 10
    mp.kwargs["losstype"] = "square"
    mp.kwargs["lam"] = 1e-2
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["minDataPts"] = 50
    mp.kwargs["graphPerIter"] = 1
    mp.kwargs["measurePerf"] = False
    mp.kwargs.update(kwargs)
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig))

def smalltest(cautious=True, seqOverride=False):
    dirconfig = DirConfig("data/smalltest_deeper/", cautious)

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-3
    mp.kwargs["decayPer"] = 100
    mp.kwargs["decayRate"] = 0.7
    mp.kwargs["epochs"] = 1500
    mp.kwargs["updateAblationPer"] = 50
    mp.kwargs["updateParzenPer"] = 50
    mp.kwargs["nSamples"] = 50
    mp.kwargs["nTestSamples"] = 10
    mp.kwargs["procsPerGpu"] = 1
    mp.kwargs["depth"] = 8
    mp.kwargs["genDepth"] = 8
    mp.kwargs["skipShardAndSave"] = False
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 5
    mp.kwargs["dUpdates"] = 1
    mp.kwargs["logDevPer"] = 10
    mp.kwargs["losstype"] = "square"
    mp.kwargs["seqOverride"] = seqOverride
    mp.kwargs["lam"] = 1e-2
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["minDataPts"] = 50
    mp.kwargs["graphPerIter"] = 500
    mp.kwargs["measurePerf"] = False

    runner = MultiganExperimentRunner()
    runner.generatePhpDirectory(PhpGenerator(dirconfig))
    runner.generateData(DataGenerator(dirconfig, test=False, **mp.kwargs))
    runner.generateGanModels(MultiganTrainer(mp))
    runner.generatePhpDirectory(PhpGenerator(dirconfig))

def holdout(cautious=True, **kwargs):
    dirconfig = DirConfig("data/holdout2/", cautious)
    dirconfig.featdir = "data/regression_filtered_fc7/"

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["decayPer"] = 1000
    mp.kwargs["decayRate"] = 0.8
    mp.kwargs["epochs"] = 15000
    mp.kwargs["updateAblationPer"] = 500
    mp.kwargs["updateParzenPer"] = 500
    mp.kwargs["nSamples"] = 50
    mp.kwargs["nTestSamples"] = 10
    mp.kwargs["procsPerGpu"] = 1
    mp.kwargs["depth"] = 16
    mp.kwargs["genDepth"] = 16
    mp.kwargs["skipShardAndSave"] = False
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 5
    mp.kwargs["dUpdates"] = 1
    mp.kwargs["logDevPer"] = 100
    mp.kwargs["losstype"] = "square"
    mp.kwargs["seqOverride"] = False
    mp.kwargs["lam"] = 5e-2
    mp.kwargs["trgandnoImg"] = False
    mp.kwargs["minDataPts"] = 0
    mp.kwargs["graphPerIter"] = 300
    mp.kwargs["measurePerf"] = False
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["skip_generate_dists"] = True
    mp.kwargs["gdropout"] = 0.5
    mp.kwargs["trnodctx"] = True
    mp.kwargs["held_out_pairs"] = {
            ("n04256520", "n02818832"): 0.5, # sofa -> bed
            ("n02374451", "n02402425"): 0.5, # horse -> cattle
            ("n02084071", "n02121620"): 0.5, # dog -> cat
            ("n02958343", "n04490091"): 0.5, # car -> truck
            ("n02958343", "n02930766"): 0.5, # car -> cab
            ("n01503061", "n01605630"): 0.5, # bird -> hawk
            ("n03948459", "n04090263"): 0.5, # pistol -> rifle
            ("n07747607", "n07749582"): 0.5, # orange -> lemon
            ("n11669921", "n13104059"): 0.5, # flower -> tree
            ("n08613733", "n08438533"): 0.5, # outdoors -> forest
        }
    mp.kwargs["train_only"] = [
            ("n04256520", "n02818832"), # sofa -> bed
            ("n02374451", "n02402425"), # horse -> cattle
            ("n02084071", "n02121620"), # dog -> cat
            ("n02958343", "n04490091"), # car -> truck
            ("n02958343", "n02930766"), # car -> cab
            ("n01503061", "n01605630"), # bird -> hawk
            ("n03948459", "n04090263"), # pistol -> rifle
            ("n07747607", "n07749582"), # orange -> lemon
            ("n11669921", "n13104059"), # flower -> tree
            ("n08613733", "n08438533"), # outdoors -> forest
        ]
    mp.kwargs.update(kwargs)

    runner = MultiganExperimentRunner(mp)
    runner.run()
    gs.make_save_chimeras(mp)

def holdout_ctx(cautious=True, **kwargs):
    dirconfig = DirConfig("data/holdout_withctx/", cautious)
    dirconfig.featdir = "data/regression_filtered_fc7/"

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["decayPer"] = 1000
    mp.kwargs["decayRate"] = 0.8
    mp.kwargs["epochs"] = 15000
    mp.kwargs["updateAblationPer"] = 500
    mp.kwargs["updateParzenPer"] = 500
    mp.kwargs["nSamples"] = 50
    mp.kwargs["nTestSamples"] = 10
    mp.kwargs["procsPerGpu"] = 1
    mp.kwargs["depth"] = 16
    mp.kwargs["genDepth"] = 16
    mp.kwargs["skipShardAndSave"] = False
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 5
    mp.kwargs["dUpdates"] = 1
    mp.kwargs["logDevPer"] = 100
    mp.kwargs["losstype"] = "square"
    mp.kwargs["seqOverride"] = False
    mp.kwargs["lam"] = 5e-2
    mp.kwargs["trgandnoImg"] = False
    mp.kwargs["minDataPts"] = 0
    mp.kwargs["graphPerIter"] = 300
    mp.kwargs["measurePerf"] = False
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["skip_generate_dists"] = False
    mp.kwargs["gdropout"] = 0.5
    mp.kwargs["trnodctx"] = False
    mp.kwargs["held_out_pairs"] = {
            ("n04256520", "n02818832"): 0.5, # sofa -> bed
            ("n02374451", "n02402425"): 0.5, # horse -> cattle
            ("n02084071", "n02121620"): 0.5, # dog -> cat
            ("n02958343", "n04490091"): 0.5, # car -> truck
            ("n02958343", "n02930766"): 0.5, # car -> cab
            ("n01503061", "n01605630"): 0.5, # bird -> hawk
            ("n03948459", "n04090263"): 0.5, # pistol -> rifle
            ("n07747607", "n07749582"): 0.5, # orange -> lemon
            ("n11669921", "n13104059"): 0.5, # flower -> tree
            ("n08613733", "n08438533"): 0.5, # outdoors -> forest
        }
    mp.kwargs["train_only"] = [
            ("n04256520", "n02818832"), # sofa -> bed
            ("n02374451", "n02402425"), # horse -> cattle
            ("n02084071", "n02121620"), # dog -> cat
            ("n02958343", "n04490091"), # car -> truck
            ("n02958343", "n02930766"), # car -> cab
            ("n01503061", "n01605630"), # bird -> hawk
            ("n03948459", "n04090263"), # pistol -> rifle
            ("n07747607", "n07749582"), # orange -> lemon
            ("n11669921", "n13104059"), # flower -> tree
            ("n08613733", "n08438533"), # outdoors -> forest
        ]
    mp.kwargs.update(kwargs)

    runner = MultiganExperimentRunner(mp)
    runner.run()

    gs.make_save_chimeras(mp)

def holdout_ctx_small(cautious=True, **kwargs):
    dirconfig = DirConfig("data/holdout_ctx_small/", cautious)
    dirconfig.featdir = "data/regression_filtered_fc7/"

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["decayPer"] = 1000
    mp.kwargs["decayRate"] = 0.8
    mp.kwargs["epochs"] = 15000
    mp.kwargs["updateAblationPer"] = 500
    mp.kwargs["updateParzenPer"] = 500
    mp.kwargs["nSamples"] = 50
    mp.kwargs["nTestSamples"] = 10
    mp.kwargs["procsPerGpu"] = 1
    mp.kwargs["depth"] = 2
    mp.kwargs["genDepth"] = 2
    mp.kwargs["skipShardAndSave"] = False
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 5
    mp.kwargs["dUpdates"] = 1
    mp.kwargs["logDevPer"] = 100
    mp.kwargs["losstype"] = "square"
    mp.kwargs["seqOverride"] = False
    mp.kwargs["lam"] = 5e-2
    mp.kwargs["trgandnoImg"] = False
    mp.kwargs["minDataPts"] = 0
    mp.kwargs["graphPerIter"] = 300
    mp.kwargs["measurePerf"] = False
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["skip_generate_dists"] = False
    mp.kwargs["gdropout"] = 0.5
    mp.kwargs["trnodctx"] = False
    mp.kwargs["held_out_pairs"] = {
            ("n04256520", "n02818832"): 0.5, # sofa -> bed
            ("n02374451", "n02402425"): 0.5, # horse -> cattle
            ("n02084071", "n02121620"): 0.5, # dog -> cat
            ("n02958343", "n04490091"): 0.5, # car -> truck
            ("n02958343", "n02930766"): 0.5, # car -> cab
            ("n01503061", "n01605630"): 0.5, # bird -> hawk
            ("n03948459", "n04090263"): 0.5, # pistol -> rifle
            ("n07747607", "n07749582"): 0.5, # orange -> lemon
            ("n11669921", "n13104059"): 0.5, # flower -> tree
            ("n08613733", "n08438533"): 0.5, # outdoors -> forest
        }
    mp.kwargs["train_only"] = [
            ("n04256520", "n02818832"), # sofa -> bed
            ("n02374451", "n02402425"), # horse -> cattle
            ("n02084071", "n02121620"), # dog -> cat
            ("n02958343", "n04490091"), # car -> truck
            ("n02958343", "n02930766"), # car -> cab
            ("n01503061", "n01605630"), # bird -> hawk
            ("n03948459", "n04090263"), # pistol -> rifle
            ("n07747607", "n07749582"), # orange -> lemon
            ("n11669921", "n13104059"), # flower -> tree
            ("n08613733", "n08438533"), # outdoors -> forest
        ]
    mp.kwargs.update(kwargs)

    runner = MultiganExperimentRunner(mp)
    runner.run()

    gs.make_save_chimeras(mp)

def holdout_small(cautious=True, **kwargs):
    dirconfig = DirConfig("data/holdout_small/", cautious)
    dirconfig.featdir = "data/regression_filtered_fc7/"

    mp = MultiganParameters(dirconfig)
    mp.kwargs["lr"] = 1e-4
    mp.kwargs["decayPer"] = 1000
    mp.kwargs["decayRate"] = 0.8
    mp.kwargs["epochs"] = 15000
    mp.kwargs["updateAblationPer"] = 500
    mp.kwargs["updateParzenPer"] = 500
    mp.kwargs["nSamples"] = 50
    mp.kwargs["nTestSamples"] = 10
    mp.kwargs["procsPerGpu"] = 1
    mp.kwargs["depth"] = 2
    mp.kwargs["genDepth"] = 2
    mp.kwargs["skipShardAndSave"] = False
    mp.kwargs["batchSize"] = 128
    mp.kwargs["logPer"] = 5
    mp.kwargs["dUpdates"] = 1
    mp.kwargs["logDevPer"] = 100
    mp.kwargs["losstype"] = "square"
    mp.kwargs["seqOverride"] = False
    mp.kwargs["lam"] = 5e-2
    mp.kwargs["trgandnoImg"] = False
    mp.kwargs["minDataPts"] = 0
    mp.kwargs["graphPerIter"] = 300
    mp.kwargs["measurePerf"] = False
    mp.kwargs["trgandnoImg"] = True
    mp.kwargs["skip_generate_dists"] = False
    mp.kwargs["gdropout"] = 0.5
    mp.kwargs["trnodctx"] = True
    mp.kwargs["held_out_pairs"] = {
            ("n04256520", "n02818832"): 0.5, # sofa -> bed
            ("n02374451", "n02402425"): 0.5, # horse -> cattle
            ("n02084071", "n02121620"): 0.5, # dog -> cat
            ("n02958343", "n04490091"): 0.5, # car -> truck
            ("n02958343", "n02930766"): 0.5, # car -> cab
            ("n01503061", "n01605630"): 0.5, # bird -> hawk
            ("n03948459", "n04090263"): 0.5, # pistol -> rifle
            ("n07747607", "n07749582"): 0.5, # orange -> lemon
            ("n11669921", "n13104059"): 0.5, # flower -> tree
            ("n08613733", "n08438533"): 0.5, # outdoors -> forest
        }
    mp.kwargs["train_only"] = [
            ("n04256520", "n02818832"), # sofa -> bed
            ("n02374451", "n02402425"), # horse -> cattle
            ("n02084071", "n02121620"), # dog -> cat
            ("n02958343", "n04490091"), # car -> truck
            ("n02958343", "n02930766"), # car -> cab
            ("n01503061", "n01605630"), # bird -> hawk
            ("n03948459", "n04090263"), # pistol -> rifle
            ("n07747607", "n07749582"), # orange -> lemon
            ("n11669921", "n13104059"), # flower -> tree
            ("n08613733", "n08438533"), # outdoors -> forest
        ]
    mp.kwargs.update(kwargs)

    runner = MultiganExperimentRunner(mp)
    runner.run()

    gs.make_save_chimeras(mp)

if __name__ == '__main__':
    runLowlearnNice2()
