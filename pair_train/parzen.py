import data
import gan as gan
import pair_train.nn as nn
import utils.mylogger as logging

def try_parzen():
    ganfile = "data/manygan_lowlearn/multigan/trained_models/1124_1332.gan"
    gantrainset = "data/manygan_lowlearn/multigan/nounpair_data/1124_1332.data"
    # TODO cross-validate?
    for factor in xrange(0, 24):
        method = 2 ** factor
        #method = "scott"
        probs = gan.parzenWindowFromFile(ganfile, gantrainset, "data/manygan_lowlearn/data/pairs/", "data/comp_fc7/", logPer=1, bw_method=method, test=False)
        logging.getLogger(__name__).info("base=%d, probs=%s" % (factor, str(probs.values())))

if __name__ == '__main__':
    try_parzen()
