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
        # Only want to condition by target semantics
        an, role, n1, n2, im_and_score = pdm.decodeCond(full_conditional)
        target_an = an.clone()
        # Replace the role-n1 pair with role-n2
        role = role[(0,)]
        n1 = n1[(0,)]
        n2 = n2[(0,)]
        for i in range(0, target_an.size(1), 2):
            if target_an[(0, i)] == role and target_an[(0, i + 1)] == n1:
                target_an[(0, i + 1)] = n2
            return target_an
        raise ValueError(
                "Invalid conditional: an=%s, role=%s, n1=%s, n2=%s, "
                "im_and_score=%s" % (an, role, n1, n2, im_and_score))
    sharder = dataman.CondSharder(kwargs["style"])
    shards = sharder.shard(cond_by, dev_dataset)
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

    for target_an, data in chosen:
        conditional, im_and_score = data
        conditional = ag.Variable(conditional).cuda(kwargs["gpu_id"])
        im_and_score = ag.Variable(im_and_score).cuda(kwargs["gpu_id"])
        noise.data.normal_(0, 1)
        gen = netg(
                noise, conditional, train=True,
                ignoreCond=kwargs.get("ignoreCond", False))
        dists = [l2(gen.data, pdm.decodeFeatsAndScore(datapt[1])[0].cuda(kwargs["gpu_id"])) for _, datapt in choices]
        dist_and_indices = zip(dists, xrange(len(dists)))
        dist_and_indices = sorted(dist_and_indices, key=lambda x: x[0])
        dist_and_indices = dist_and_indices[:kwargs["nn_considered"]]
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
