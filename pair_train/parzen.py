import data
import gan as gan
import pair_train.nn as nn
import utils.mylogger as logging

def try_parzen():
    ganfile = "data/smalltest3/multigan/trained_models/1124_1332.gan"
    gantrainset = "data/smalltest3/multigan/nounpair_data/1124_1332.data"
    # TODO cross-validate?
    for factor in xrange(0, 25):
        method = 2 ** factor
        #method = "scott"
        pdm = data.PairDataManager("data/smalltest3/data/pairs/", "data/regression_fc7/")
        probs = gan.parzenWindowFromFile(ganfile, gantrainset, pdm, logPer=100, bw_method=method, test=False, gpu_id=0, batchSize=64)
        logging.getLogger(__name__).info("base=%d, probs=%s" % (factor, str(probs.values())))

if __name__ == '__main__':
    try_parzen()
