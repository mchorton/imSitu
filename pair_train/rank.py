import collections
import random as rd
import numpy as np
import torch
import torch.utils.data as td
import torch.autograd as ag
import data as dataman
import utils.mylogger as logging
# Evaluate the rank of generated images and their nearest neighbors.
"""
1. Find semantics of a dev set point.
2. Generate the given semantics from the source image.
3. Find nearest neighbors of that generated image.
4a. Look at how prominent the generated image's label is among its nearest nbrs
4b. Look at the mean rank of nearest neighbors that have the correct label 
    (compared to those that don't).
5. Consider normalizing by nearest neighbors of the actual target images. Are
    they even clustered?
"""

def l2(im1, im2):
    return torch.sum(torch.pow((im1 - im2), 2.))

def evalRank(netg, dev_dataset, pdm, **kwargs):
    logging.getLogger(__name__).info("Evaluating rank")
    def cond_by(full_conditional):
        #logging.getLogger(__name__).info("type(full_conditional)=%s" % str(full_conditional))
        #logging.getLogger(__name__).info("conditional is: %s" % str(pdm.condToString(full_conditional)))
        #sys.exit(1)
        # Should be 1xFeatureSize
        # Only want to condition by target semantics
        an, role, n1, n2, im_and_score = pdm.decodeCond(full_conditional)
        target_an = an.clone()
        # Replace the role-n1 pair with role-n2
        """
        logging.getLogger(__name__).info("target annotation is %s" % str(an))
        logging.getLogger(__name__).info("role is %s" % str(role))
        logging.getLogger(__name__).info("n1 is %s" % str(n1))
        logging.getLogger(__name__).info("target annotation size is %s" % str(target_an.size()))
        """
        role = role[(0,)]
        n1 = n1[(0,)]
        n2 = n2[(0,)]
        for i in range(0, target_an.size(1), 2):
            """
            logging.getLogger(__name__).info("target_an[(0, i)] is %s" % str(target_an[(0, i)]))
            logging.getLogger(__name__).info("target_an[(0, i + 1)] is %s" % str(target_an[(0, i + 1)]))
            """
            if target_an[(0, i)] == role and target_an[(0, i + 1)] == n1:
                target_an[(0, i + 1)] = n2
            return target_an
        raise ValueError(
                "Invalid conditional: an=%s, role=%s, n1=%s, n2=%s, "
                "im_and_score=%s" % (an, role, n1, n2, im_and_score))
    """
    logging.getLogger(__name__).info("dev_dataset is %s" % str(dev_dataset))
    """
    sharder = dataman.CondSharder(kwargs["style"])
    shards = sharder.shard(cond_by, dev_dataset)
    """
    logging.getLogger(__name__).info("shards=%s" % str(shards))
    """
    # algorithm:
    # 1. randomly choose a point. generate images. (just make list with shuffled
    # index pointers, or sth)
    # 2. choose 10 points.
    choices = [(key, datapt) for key, list_ in shards.iteritems() for datapt in list_]
    logging.getLogger(__name__).info("len(Choices) is %d" % len(choices))
    rd.shuffle(choices)
    chosen = choices[:kwargs["nn_sampled_pts"]]
    noise = ag.Variable(torch.FloatTensor(
            1, kwargs["nz"]) # batch size is 1
            .cuda(kwargs["gpu_id"]))

    results = collections.defaultdict(list)
    #logging.getLogger(__name__).info("Choices is %s" % str(choices))

    for target_an, data in chosen:
        """
        logging.getLogger(__name__).info(type(chosen))
        logging.getLogger(__name__).info("target_an=%s" % str(target_an))
        logging.getLogger(__name__).info("data=%s" % str(data))
        """
        conditional, im_and_score = data
        conditional = ag.Variable(conditional).cuda(kwargs["gpu_id"])
        im_and_score = ag.Variable(im_and_score).cuda(kwargs["gpu_id"])
        """
        logging.getLogger(__name__).info("conditional=%s" % str(conditional))
        logging.getLogger(__name__).info("im_and_score=%s" % str(im_and_score))
        """
        noise.data.normal_(0, 1)
        # TODO still using train=True, e.g. dropout for generation. TODO?
        gen = netg(
                noise, conditional, train=True,
                ignoreCond=kwargs.get("ignoreCond", False))
        dists = [l2(gen.data, pdm.decodeFeatsAndScore(datapt[1])[0].cuda(kwargs["gpu_id"])) for _, datapt in choices]
        dist_and_indices = zip(dists, xrange(len(dists)))
        dist_and_indices = sorted(dist_and_indices, key=lambda x: x[0])
        """
        logging.getLogger(__name__).info("len(dai: %s" % str(len(dist_and_indices)))

        logging.getLogger(__name__).info("nn_considered is %s" % str(kwargs["nn_considered"]))
        """
        dist_and_indices = dist_and_indices[:kwargs["nn_considered"]]
        """
        logging.getLogger(__name__).info("conditional is %s" % str(conditional))
        logging.getLogger(__name__).info("conditional is %s" % str(conditional))
        """
        target_an = cond_by(conditional.data).cpu()
        target_rank = 0
        n_correct_an = 0
        for i, (_, choice_index) in enumerate(dist_and_indices, start=1):
            if torch.equal(cond_by(choices[choice_index][1][0]), target_an):
                target_rank += i
                n_correct_an += 1
        nsamples = len(dist_and_indices)
        total_avg_rank = (nsamples + 1) / 2
        best_avg_rank = (n_correct_an + 1) / 2
        worst_avg_rank = sum(xrange(len(choices) - n_correct_an, len(choices))) / n_correct_an
        target_avg_rank = target_rank / n_correct_an
        #logging.getLogger(__name__).info("target_rank=%d" % target_rank)
        #logging.getLogger(__name__).info("n_correct_an=%d" % n_correct_an)
        results["total_avg_rank"].append(total_avg_rank)
        results["best_avg_rank"].append(best_avg_rank)
        results["target_avg_rank"].append(target_avg_rank)
        results["worst_avg_rank"].append(worst_avg_rank)
    logging.getLogger(__name__).info("results is %s" % str(results))
    ret = {
            "total_avg_rank": np.mean(results["total_avg_rank"]),
            "best_avg_rank": np.mean(results["best_avg_rank"]),
            "target_avg_rank": np.mean(results["target_avg_rank"]),
            "worst_avg_rank": np.mean(results["worst_avg_rank"])}
    logging.getLogger(__name__).info("ret is %s" % str(ret))
    return ret
