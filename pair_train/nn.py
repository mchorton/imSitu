# Training the image transformation logic using a neural network.
import json
import torch.nn as nn
import numpy as np
import torch
import random
import torch.utils.data as td
import torch.optim as optim
import torch.autograd as ag
import torch.nn.functional as F
import time
from ast import literal_eval as make_tuple
import math
import utils.mylogger as logging
import tqdm
import collections
import os
import split.rp_experiments as rpe
import utils.methods as mt
import itertools as it
import split.v2pos.htmlgen as html
import split.data_utils as du
import constants as pc # short for pair_train constants
import split.splitutils as sut
import data as dataman

TORCHCOMPTRAINDATA = "data/pairLearn/comptrain.pyt"
TORCHCOMPDEVDATA = "data/pairLearn/compdev.pyt"
TORCHREGTRAINDATA = "data/pairLearn/regtrain.pyt"
TORCHREGDEVDATA = "data/pairLearn/regdev.pyt"

TORCHCOMPTRAINDATATEST = "data/pairLearn/comptrain_test.pyt"
TORCHCOMPDEVDATATEST = "data/pairLearn/compdev_test.pyt"
TORCHREGTRAINDATATEST = "data/pairLearn/regtrain_test.pyt"
TORCHREGDEVDATATEST = "data/pairLearn/regdev_test.pyt"

VRNDATATEST = "data/pairLearn/vrn_test.json"
VRNDATA = "data/pairLearn/vrn.json"
COMPFEATDIR = "data/comp_fc7/"
REGFEATDIR = "data/regression_fc7/"

# TODO I changed thresh to 'inf', from '2'
# TODO should I just remove all filtering?
# TODO: Create a function that runs everything from end to end, making all
# necessary folders, etc. Get a few clean runs of experimentation.
# it should also run all old/new style GANs, and make dashboards that look nice.
# TODO save the logs from function calls to the folder for later investigation.
def makeVrnData():
  sys.exit(1) # this is broken TODO
  datadir = "data/split/"
  import split.splitters as spsp
  spsp.splitTrainDevTestMinInTrain(datadir)

  trainSetName = os.path.join(datadir, "zsTrain.json")

  distdir = "data/vecStyle/"
  import split.rp_experiments as rpe
  rpe.generateAllDistVecStyle(distdir, trainSetName)

  vrndir = "data/vrnData/"
  rpe.getVrnData(distdir, trainSetName, vrndir, thresh=float('inf'), freqthresh=10, blacklistprs = [], bestNounOnly = True, noThreeLabel = True, includeWSC=True, noOnlyOneRole=True, strictImgSep=True, outLoc=VRNDATA + "_MOD")

def getContextVectors(
    contextVREmbedding, contextWordEmbedding, context, batchSize):
  context_vectors = []
  for i in range(6):
    emb = contextVREmbedding(context[:,2*i].long()).view(batchSize, -1)
    context_vectors.append(emb)
    context_vectors.append(contextWordEmbedding(context[:,1 + 2*i].long()).view(batchSize,-1))
  return context_vectors

class ImTransNet(nn.Module):
    def __init__(self, imFeatures, depth, nHidden, nOutput, nWords, WESize, nVRs, vrESize):
        super(ImTransNet, self).__init__()
        nInput = 0
        li = 1e-3
        nInput += imFeatures
        change_size = 2* WESize + vrESize
        #context_size = 0
        context_size = 6 * (WESize + vrESize)
        nInput +=  change_size#2 * WESize + vrESize
        nInput += context_size# 6 * (WESize + vrESize)
        print nInput

        #self.upsample_features = nn.Linear(imFeatures, 128000)
        #self.upsample_noun1 = nn.Linear(WESize, 128000)        
        #self.upsample_noun2 = nn.Linear(WESize, 128000)
        #self.upsample_role = nn.Linear(vrESize, 128000)

        self.input_layer = nn.Linear(nInput + 2*1024, nHidden)
        self.input_layer.weight.data.uniform_(-li, li)
        self.input_layer.bias.data.uniform_(-li, li)

        self.ft = nn.Linear(1024, 1024)
        self.ft.weight.data.uniform_(-li, li)
        self.ft.bias.data.uniform_(-li, li)

        linears = [nn.Linear(nHidden + context_size + change_size , nHidden) for i in range(0,depth)]
        for _l in linears: 
           #control initilization
           _l.weight.data.uniform_(-li, li)
           _l.bias.data.uniform_(-li, li)        
        self.hidden_layers = nn.ModuleList( linears )
        self.output = nn.Linear(nHidden + context_size + change_size , nOutput)
        self.output.weight.data.uniform_(-li, li)
        self.output.bias.data.uniform_(-li, li)
        
        # TODO try initializing these differently.
        #print "CONSIDER MODIFYING INITIALIZATION OF EMBEDDINGS"
        self.wordEmbedding = nn.Embedding(nWords + 1, WESize)
        self.vrEmbedding = nn.Embedding(nVRs+1 , vrESize)
     
        self.att_w = nn.Embedding(nWords, 32)
        self.att_r = nn.Embedding(nVRs, 1024*32)
        self.att_rb = nn.Embedding(nVRs, 32)
       
        self.att_w.weight.data.uniform_(-1,1)
        self.att_r.weight.data.uniform_(-li,li)
        self.att_rb.weight.data.uniform_(-li,li)

        #self.context_wordEmbedding = nn.Embedding(nWords+1, WESize)
        #self.context_vrEmbedding = nn.Embedding(nVRs+1, vrESize)
 
        print "WESize: %d" % WESize
        print "vrESize: %d" % vrESize
        print "nhidden: %d" % nHidden

    def forward(self, x, train):
        """
        x looks like [batchDim][role, noun1, noun2, features...]
        where role and noun1/noun2 are indices
        """
        # get the embedding vector for the roles.
        context = x[:,0:12]
        context_vectors = F.dropout(torch.cat(getContextVectors(self.vrEmbedding, self.wordEmbedding, context, len(x)), 1), training = train)

        roles = x[:,12]
        #print "shape of roles: %s" % str(roles.size())
        #print "roles is: %s" % str(roles)
        roleEmbeddings = F.dropout(self.vrEmbedding(roles.long()), training = train) # This is now [batchDim][1][embedding size]
        #print "shape of roleEmbeddings: %s" % str(roleEmbeddings.size())
        roleEmbeddings = roleEmbeddings.view(len(x), -1)
        #print "after reshape , shape of roleEmbeddings: %s" % str(roleEmbeddings.size())

        noun1 = x[:,13]
        noun2 = x[:,14]
        we1 = F.dropout(self.wordEmbedding(noun1.long()), training = train).view(len(x), -1)
        we2 = F.dropout(self.wordEmbedding(noun2.long()), training = train).view(len(x), -1)
       
        f = F.dropout(x[:, 15:], training = train)
        
        we1_att =  self.att_w(noun1.long()).view(len(x),-1)
        we2_att =  self.att_w(noun2.long()).view(len(x),-1)
        role_m =  self.att_r(roles.long()).view(len(x),32,1024)
        role_b = self.att_rb(roles.long()).view(len(x),32,1)        

        fp = f.view(len(x), 1024, 1)
        fp_ = self.ft(f) 
        #print "shape of role: %s" %str(role_m.size())
        #print "shape of feature: %s" %str(fp.size()) 
        role_att = torch.bmm(role_m, fp).view(len(x), -1)     
        role_att = role_att + role_b       
 
        w1r1 = torch.bmm(we2_att.view(len(x),-1,1),role_att.view(len(x),1,-1)).view(len(x),-1)
        w2r1 = torch.bmm(we1_att.view(len(x),-1,1),role_att.view(len(x),1,-1)).view(len(x),-1)
        
        #w1r1 = torch.mul(w1r1, fp_)
        #w2r1 = torch.mul(w2r1, fp_)
        #w2r1 = torch.bmm(we1_att.view(len(x),-1,1),role_att.view(len(x),1,-1)).view(len(x),-1)
        
        #attnw1 = torch.mul(w1r1, f)
        attnw2 = F.dropout(torch.cat([w1r1, w2r1], 1), training = train) #role_att #torch.mul(w2r1, f)      
        #print attnw2
        #wordEmbeddings = F.dropout(self.wordEmbedding(words.long()), training = train) # This is now [batchDim][1][embedding size]
        #wordEmbeddings = wordEmbeddings.view(len(x), -1)
        #x = torch.cat (( [context_vectors, roleEmbeddings, wordEmbeddings]),1)
        #attn = torch.mul(self.upsample_role(roleEmbeddings),torch.mul(self.upsample_noun2(we2),torch.mul(self.upsample_features(f), self.upsample_noun1(we1))))

# self.upsample_change( torch.cat( [roleEmbeddings, wordEmbeddings], 1)) )
        x = torch.cat( ( [attnw2, f, context_vectors, roleEmbeddings, we1, we2] ), 1)
        #x = torch.cat ( ( [context_vectors ,f] ), 1 )
        #x = f
        #x = torch.cat( ( [roleEmbeddings, wordEmbeddings, f] ), 1)
        #print "final x.shape: %s" % str(x.size())
        x = F.dropout(F.leaky_relu(self.input_layer(x)), training=train)
        #x += f 
        for i in range(0, len(self.hidden_layers)):
          if i == 0: x_prev = x
          x = torch.cat( ( [ context_vectors, x , roleEmbeddings, we1, we2]), 1)
          #x = torch.cat( ([ context_vectors , x ]), 1)
          x = F.dropout(F.leaky_relu(self.hidden_layers[i](x)), training=train)
          #add skip connections for depth
          #if i > 0 and i % 1 == 0: 
          x = x + x_prev
          x_prev = x
        x = torch.cat( ( [context_vectors, x , roleEmbeddings, we1,we2]), 1)
        #x = torch.cat( ( [context_vectors, x ]), 1)
        #x = F.dropout(F.relu(self.hidden2(x)), training=train)
        #x = F.dropout(F.relu(self.hidden3(x)), training=train)
        x = self.output(x)
        return x

class MatrixDot(nn.Module):
  def __init__(self, nWords, nVRs, eSize, imSize=1024):
    super(MatrixDot, self).__init__()
    self.avrEmbedding = nn.Embedding(nVRs, imSize * eSize)
    self.wordEmbedding = nn.Embedding(nWords, eSize)
    self.imSize = imSize
    # TODO consider modifying this.
    self.avrEmbedding.weight.data = (torch.rand(self.avrEmbedding.weight.data.size()) - 0.5) * 2e-4
    self.wordEmbedding.weight.data = (torch.rand(self.wordEmbedding.weight.data.size()) - 0.5) * 2e-1

  def forward(self, x):
    """
    x looks like [batchDim][role, noun1, noun2, features...]
    """
    roles = x[:,0]
    noun1s = x[:,1]
    noun2s = x[:,2]
    imFeatures = x[:,3:]

    Avr = self.avrEmbedding(roles.long()).view(len(x), self.imSize, -1) #.view(self.imSize, -1)
    #print "Avr SIZE: %s" % str(Avr.size())

    #expandedAvr = Avr.view(1, *Avr.size()).expand(len(x), *Avr.size())
    #print "ExpAvr SIZE: %s" % str(expandedAvr.size())

    n1Emb = self.wordEmbedding(noun1s.long())
    n2Emb = self.wordEmbedding(noun2s.long())
    embDif = n2Emb - n1Emb
    #print "embDif %s" % str(embDif.size())
    #print "features %s" % str(imFeatures.size())

    resizedEmbDif = embDif.view(embDif.size()[0], embDif.size()[1], 1)
    #print "resizedEmbDif SIZE: %s" % str(resizedEmbDif.size())

    finalDiff = torch.bmm(Avr, resizedEmbDif).view(len(x), -1)
    #print "finalDiff SIZE: %s" % str(finalDiff.size())
    x = imFeatures + finalDiff
    return x

class MatrixCross(nn.Module):
  def __init__(self, nWords, wESize, nVRs, vrESize, imSize=1024):
    super(MatrixCross, self).__init__()
    print "CONSIDER CHANGING EMBEDDING INITIALIZATION"
    self.avrEmbedding = nn.Embedding(nVRs, (vrESize * imSize))
    self.wordEmbedding = nn.Embedding(nWords, wESize)
    self.avrEmbedding.weight.data = (torch.rand(self.avrEmbedding.weight.data.size()) - 0.5) * 2e-4
    self.wordEmbedding.weight.data = (torch.rand(self.wordEmbedding.weight.data.size()) - 0.5) * 2e-1
    self.B = nn.Linear(vrESize * wESize, imSize)
    self.imSize = imSize
    self.vrESize = vrESize
    self.wESize = wESize

  def forward(self, x):
    """
    x looks like [batchDim][role, noun1, noun2, features...]
    """
    roles = x[:,0]
    noun1s = x[:,1]
    noun2s = x[:,2]
    imFeatures = x[:,3:].contiguous() 
    Avr = self.avrEmbedding(roles.long()).view(len(x), self.vrESize, -1)
    #Avr = Avr.view(self.vrESize, -1)
    n1Emb = self.wordEmbedding(noun1s.long())
    n2Emb = self.wordEmbedding(noun2s.long())

    # This is [batchDim][vrEsize][1]
    avrDotI = torch.bmm(Avr, imFeatures.view(imFeatures.size()[0], imFeatures.size()[1], 1))
    avrDotI = avrDotI.view(avrDotI.size()[0], avrDotI.size()[1])

    batchOP1 = torch.bmm(avrDotI.view(len(x), -1, 1), n1Emb.view(len(x), 1, -1)).view(len(x), -1)
    f1 = self.B(batchOP1)

    batchOP2 = torch.bmm(avrDotI.view(len(x), -1, 1), n2Emb.view(len(x), 1, -1)).view(len(x), -1)
    f2 = self.B(batchOP2)

    x = imFeatures - f1 + f2
    return x

# Our json object VRNData looks like:
# pairing_score, image1, image2, transformation_role, image1_noun_value, image2_noun_value, image1_merged_reference, image2_merged_reference
# TODO try running new model with all data.
def makeAllData():
  global TORCHCOMPTRAINDATA
  global TORCHCOMPDEVDATA
  global COMPFEATDIR
  global TORCHREGTRAINDATA
  global TORCHREGDEVDATA
  global REGFEATDIR
  global VRNDATA
  global VRNDATATEST
  global TORCHCOMPTRAINDATATEST
  global TORCHCOMPDEVDATATEST
  global TORCHREGTRAINDATATEST
  global TORCHREGDEVDATATEST

  makeData(
      TORCHCOMPTRAINDATATEST, TORCHCOMPDEVDATATEST, COMPFEATDIR, VRNDATATEST,
      mode="all")
  makeData(
      TORCHREGTRAINDATATEST, TORCHREGDEVDATATEST, REGFEATDIR, VRNDATATEST,
      mode="all")
  makeData(
      TORCHCOMPTRAINDATA, TORCHCOMPDEVDATA, COMPFEATDIR, VRNDATA, mode="all")
  makeData(TORCHREGTRAINDATA, TORCHREGDEVDATA, REGFEATDIR, VRNDATA, mode="all")

  makeData(
      TORCHCOMPTRAINDATATEST, TORCHCOMPDEVDATATEST, COMPFEATDIR, VRNDATATEST)
  makeData(TORCHREGTRAINDATATEST, TORCHREGDEVDATATEST, REGFEATDIR, VRNDATATEST)
  makeData(TORCHCOMPTRAINDATA, TORCHCOMPDEVDATA, COMPFEATDIR, VRNDATA)
  makeData(TORCHREGTRAINDATA, TORCHREGDEVDATA, REGFEATDIR, VRNDATA)

def makeDataNice(*args, **kwargs):
    trainDir = os.path.dirname(args[0]) + "/"
    mt.makeDirIfNeeded(trainDir)
    with open(os.path.join(trainDir, "args.txt"), "w+") as argfile:
            argfile.write("ARGS:%s\nKWARGS:%s\n" % (str(args), str(kwargs)))
    makeData(*args, **kwargs)
    featDir = args[2]
    makeDataHtml(trainDir, featDir)

def strGetDict(mydict):
    return lambda x: mydict.get(x, "")

def intGetDict(mydict):
    return lambda x: mydict.get(x, 0)

def makeDataHtml(outDir, featDir):
    """
    Create an HTML file inside outDir that shows useful things about the dataset
    """
    maker = html.HtmlMaker()

    argsFile = os.path.join(outDir, "args.txt")
    arguments = html.PhpTextFile(os.path.abspath(argsFile))

    maker.addElement(arguments)

    table = html.HtmlTable()
    table.addRow(
            "ANNOTATIONS", "ROLE", "N1", "N2", "SOURCE FEATURES",
            "TARGET FEATURES", "SOURCE IMG", "TARGET IMG", "SCORE")

    dataset = torch.load(os.path.join(outDir, "pairtrain.pyt"))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    pdm = dataman.PairDataManager(outDir, featDir)
    for i, data in tqdm.tqdm(
            enumerate(it.islice(dataloader, 20), 0),
            total=min(len(dataloader), 20), desc="gathering samples"):
        cond, imfeatsAndScore = data
        annotations, role, n1, n2, sourceFeats = pdm.decodeCond(cond)
        targetFeats, score = pdm.decodeFeatsAndScore(imfeatsAndScore)

        partlyDecodedAn = pdm.annotationsintsToCodes(annotations)
        fullyDecodedAn = pdm.partdecodeToFulldecode(partlyDecodedAn)
        annotationsStr = "%s\n%s\n%s" % \
                (str(annotations), str(partlyDecodedAn), str(fullyDecodedAn))
        roleStr = "%s\n%s" % (str(role), str(pdm.int2role.get(role[0,], "MISSING")))
        def nounstr(noun):
            return "%s\n%s\n%s" % (
                    str(noun), str(pdm.int2noun.get(noun, "MISSING")),
                    du.decodeNoun(pdm.int2noun[noun]))
        n1Str = nounstr(n1[0,])
        n2Str = nounstr(n2[0,])
        srcName = pdm.getImnameFromFeats(sourceFeats)
        tarName = pdm.getImnameFromFeats(targetFeats)
        sourceStr = "%s\n%s" % (str(sourceFeats), srcName)
        targetStr = "%s\n%s" % (str(targetFeats), tarName)
        srcIm = html.ImgRef(src=sut.getUrl(srcName))
        tarIm = html.ImgRef(src=sut.getUrl(tarName))
        table.addRow(
                annotationsStr, roleStr, n1Str, n2Str, sourceStr, targetStr,
                srcIm, tarIm, score)
        
    maker.addElement(table)
    outname = os.path.join(outDir, "index.php")
    logging.getLogger(__name__).info("Saving php to '%s'" % outname)
    maker.save(os.path.join(outDir, "index.php"))

def makeData(trainLoc, devLoc, featDir, vrndatafile, mode="max"):
  mt.makeDirIfNeeded(trainLoc)

  logging.getLogger(__name__).info(
      "Making data, train='%s', dev='%s'" % (trainLoc, devLoc))
  # prep_work
  vrnData = json.load(open(vrndatafile))
  allNames = list(set([pt[1] for pt in vrnData]))
  random.seed(79569) #a really random seed
  random.shuffle(allNames) 
  trainImgNames = set(allNames[:len(allNames)/2])
  devImgNames = set(allNames[len(allNames)/2:])

  trainImgNameFile = "%s_%s" % (trainLoc, "trainImgNames.json")
  json.dump(list(trainImgNames), open(trainImgNameFile, "w+"))

  devImgNameFile = "%s_%s" % (devLoc, "devImgNames.json")
  logging.getLogger(__name__).info("Saving img names to %s" % devImgNameFile)
  json.dump(list(devImgNames), open(devImgNameFile, "w+"))

  dataSet = makeDataSet(
      trainLoc, featDir, vrnData, trainImgNames, mode=mode)
  logging.getLogger(__name__).info("Saving data to %s" % trainLoc)
  torch.save(dataSet, trainLoc)

  dataSet = makeDataSet(
      devLoc, featDir, vrnData, devImgNames, mode=mode)
  logging.getLogger(__name__).info("Saving data to %s" % devLoc)
  torch.save(dataSet, devLoc)



# TODO refactor this a bit. Also, make some stability tests!
def makeDataSet(
    trainLoc, featureDirectory, vrnData, whitelistedImgNames, mode="all"):
  """
  Create a pytorch TensorDataset at 'outFileName'. It contains input suitable
  for the models trained to generate image features.
  trainLoc - the destination to which this dataset will be saved. (Note this
             function doesn't save the dataset, but it needs a pathname from
             which to generate intermediate file names)
  ...
  mode - if "max", a unique (im2Name, str(tRole), noun2) will only allow a
         single im1Name to be paired with it.

  """
  # Split it into lists of the roles, noun1, noun2, verb
  scores = [pt[0] for pt in vrnData]
  im1Names = [pt[1] for pt in vrnData]
  im2Names = [pt[2] for pt in vrnData]
  tRoles = [pt[3] for pt in vrnData] # This is a tuple btw.
  n1s = [pt[4] for pt in vrnData]
  n2s = [pt[5] for pt in vrnData]
  full_an1 = [pt[6] for pt in vrnData]
  full_an2 = [pt[7] for pt in vrnData]

  def getUniqueVrnNounpairs(vrnData):
    # Useful for debugging.
    n1s = [pt[4] for pt in vrnData]
    n2s = [pt[5] for pt in vrnData]
    nUP = len(frozenset(zip(n1s, n2s)))
    logging.getLogger(__name__).info("There are %d unique noun pairs" % nUP)
    logging.getLogger(__name__).info(
        "There are %d points in vrnData" % len(vrnData))

  logging.getLogger(__name__).info("Beginning makeDataSet:")
  getUniqueVrnNounpairs(vrnData)

  logging.getLogger(__name__).info("Done reading data")
  
  verb_len = {}
  all_verbs = set()
  all_roles = set()
  all_nouns = set()
  for _map in full_an1:
    for (k,v) in _map.items():
      (verb,role) = make_tuple(k)
      all_verbs.add(verb)
      all_roles.add(make_tuple(k))
      all_nouns.add(v)     
    verb_len[verb] = len(_map)

  logging.getLogger(__name__).debug("Done getting verb lengths")

  rolesAsTuples = all_roles 
  role2Int = \
      {role:index for index, role in enumerate(sorted(list(rolesAsTuples)))}
  allNounsUsed = all_nouns 
  noun2Int = \
      {noun: index for index, noun in enumerate(sorted(list(allNounsUsed)))}
  # TODO I think this is unused, since we only go from roles to ints.
  verb2Int = {verb: index for index, verb in enumerate(sorted(list(all_verbs)))}
  logging.getLogger(__name__).debug("Done getting maps")

  torch.save(role2Int, "%s_role2Int" % trainLoc)
  torch.save(noun2Int, "%s_noun2Int" % trainLoc)
  torch.save(verb2Int, "%s_verb2Int" % trainLoc)

  logging.getLogger(__name__).info("verb2Int size: %d" % len(verb2Int))
  logging.getLogger(__name__).info("role2int size: %d" % len(role2Int))
  logging.getLogger(__name__).info("noun2Int size: %d" % len(noun2Int))

  imToFeatures = dataman.getIm2Feats(featureDirectory, im1Names)

  # Loop through, building train and dev sets.
  wasted = 0
  def l2(v1,v2):
    rv = 0
    for i in range(0, len(v1)) : rv += (v1[i]-v2[i])*(v1[i]-v2[i])
    return math.sqrt(rv)

  # TODO we should enforce that the above only happens once, to ensure that the
  # mapping from roles to ints is the same for training / dev sets. For now,
  # because we only filter vrnData at this stage, things will be consistent.

  # Now, cut down vrnData to only the whitelisted elements.
  logging.getLogger(__name__).info("Before filter: vrnData has %d elems" % len(vrnData))
  getUniqueVrnNounpairs(vrnData)
  vrnData = filter(lambda x: x[1] in whitelistedImgNames, vrnData)
  logging.getLogger(__name__).info("After filter: vrnData has %d elems" % len(vrnData))
  getUniqueVrnNounpairs(vrnData)

  if mode != "all":
    img_semchange = {}
    for i, (score, im1Name, im2Name, tRole, noun1, noun2, an1, an2) in tqdm.tqdm(enumerate(vrnData), total=len(vrnData)):
      key = (im1Name, im2Name, str(tRole), noun2)
      if key not in img_semchange: img_semchange[key] = []
      img_semchange[key].append( (score, im1Name, im2Name, tRole, noun1, noun2, an1, an2) )
      
    vrnData = []
    i = 0
    for (k, v) in tqdm.tqdm(img_semchange.items(), total=len(img_semchange)):
      i+=1
      ftgt = imToFeatures[k[0]]
      if mode == "max":
        mv = float('inf')
        best = None
        for _item in v:
          d = l2(imToFeatures[_item[1]], ftgt)
          if d < mv: 
            mv = d
            best = _item
        vrnData.append(best)
  getUniqueVrnNounpairs(vrnData)

  return get_dataset(vrnData, role2Int, noun2Int, imToFeatures)

def get_dataset_pdm(vrn_data, pdm):
    return get_dataset(vrn_data, pdm.role2int, pdm.noun2int, pdm.name2feat)

def get_dataset(vrn_data, role2int, noun2int, im_2_features):
  xData = []
  yData = []
  for i, (score, im1Name, im2Name, tRole, noun1, noun2, an1, an2) in tqdm.tqdm(
        enumerate(vrn_data), total=len(vrn_data)):
    anitems = [] 
    for (k,v) in an1.items():
      rid = role2int[make_tuple(k)]
      nid = noun2int[v]
      anitems.append((rid,nid))
    while len(anitems) < 6:
      anitems.append((len(role2int), len(noun2int)))
   
    anitems = sorted(anitems, key = lambda x : x[0]) 
    indexes = []
    for (k,v) in anitems: indexes += [k,v]
 
    x = list(indexes) \
        + [role2int[tuple(tRole)], noun2int[noun1], noun2int[noun2]] \
        + list(im_2_features[im1Name]) 
    y = list(im_2_features[im2Name]) + [score]

    xData.append(x)
    yData.append(y)

  dataSet = td.TensorDataset(torch.Tensor(xData), torch.Tensor(yData))
  return dataSet

def runModelTest(modelName, modelType, lr=0.0001, epochs=1, depth=2):
  makeData(
      TORCHCOMPTRAINDATATEST, TORCHCOMPDEVDATATEST, COMPFEATDIR, VRNDATATEST)
  makeData(TORCHREGTRAINDATATEST, TORCHREGDEVDATATEST, REGFEATDIR, VRNDATATEST)
  runModel(
      modelName, modelType, depth, lr, TORCHCOMPTRAINDATATEST,
      TORCHCOMPDEVDATATEST, epochs=epochs)

def runModel(
    modelName, modelType, depth=20, lr=0.25, trainLoc=TORCHCOMPTRAINDATA,
    devLoc=TORCHCOMPDEVDATA, epochs=10, weight_decay=0.01, nHidden = 1024,
    batchSize=128, decay_iter = 2):
  # Get the data
  #trainLoc = TORCHCOMPTRAINDATATEST
  #devLoc = TORCHCOMPDEVDATATEST
  print "Loading training data from %s" % str(trainLoc)
  train = torch.load(trainLoc)
  
  trainloader = td.DataLoader(
      train, batch_size=batchSize, shuffle=True, num_workers=0)

  print "Loading dev data from %s" % str(devLoc)
  dev = torch.load(devLoc)
  devloader = td.DataLoader(
      dev, batch_size=batchSize, shuffle=True, num_workers=0)

  print "found %d train datapoints" % len(train)
  print "found %d dev datapoints" % len(dev)

  x1, y1AndScore = train[0]
  y1 = y1AndScore[:-1]
  indim = len(x1) # Should be the number of dimensions.
  outdim = len(y1) # Should be the number of dimensions.
  print "Got indim as %s" % str(indim)
  print "Got outdim as %s" % str(outdim)

  printPer = 200

  role2Int = torch.load("%s_role2Int" % trainLoc)
  noun2Int = torch.load("%s_noun2Int" % trainLoc)

  nVRs = max(role2Int.values()) + 1 # TODO shouldn't add one... ?
  nWords = max(noun2Int.values()) + 1
  WESize = 256
  vrESize = 256
  print "nVRs: %d" % nVRs
  print "nWords: %d" % nWords
  print "model type {0}".format(modelType)
  if modelType == "nn":
    print "running nn"
    net = ImTransNet(
        indim - (3+12), depth, nHidden, outdim, nWords, WESize, nVRs,
        vrESize).cuda(gpu_id)
  elif modelType == "dot":
    net = MatrixDot(nWords, nVRs, int((vrESize + WESize) / 2)).cuda(gpu_id)
  elif modelType == "cross":
    net = MatrixCross(nWords, WESize, nVRs, vrESize, 1024).cuda(gpu_id)
  else:
    raise Exception("invalid modelType '%s'" % str(modelType))

  criterion = nn.MSELoss()
  optimizer = optim.SGD(
      net.parameters(), lr=lr, momentum=.9, weight_decay=.0001)
  #optimizer = optim.Adam(net.parameters(), lr, weight_decay=weight5decay)
  #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)
  def getLoss(loader, epoch, name):
    running_loss = 0.
    nIter = 0
    for i, data in enumerate(loader, 0):
      inputs, labelsAndScore = data
      scores = ag.Variable(labelsAndScore[:,-1].cuda(gpu_id))
      labels = labelsAndScore[:,:-1].cuda(gpu_id)
      inputs, labels = ag.Variable(inputs.cuda(gpu_id)), ag.Variable(labels.cuda(gpu_id))
      outputs = net(inputs,False)
      scores = scores.view(len(scores), 1).expand(outputs.size())

      loss = criterion(torch.mul(outputs, scores), torch.mul(labels, scores))
      running_loss += loss.data[0] # unsure if this is normalized.
      nIter += 1
    print('%s Set [Epoch %d] loss: %.5g' % (name, epoch+1, running_loss / nIter))

  getLoss(trainloader, -1, "Train")
  getLoss(devloader, -1, "Dev")
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs
      inputs, labelsAndScore = data

      scores = ag.Variable(labelsAndScore[:,-1].cuda(gpu_id)) # Last row is scores
      labels = labelsAndScore[:,:-1]
      
      # wrap them in Variable
      inputs, labels = ag.Variable(inputs.cuda(gpu_id)), ag.Variable(labels.cuda(gpu_id))
      
      # zero the parameter gradients
      optimizer.zero_grad()
      
      # forward + backward + optimize
      outputs = net(inputs,True)
      scores = scores.view(len(scores), 1).expand(outputs.size())
      loss = criterion(torch.mul(outputs, scores), torch.mul(labels, scores))
      loss.backward()
      optimizer.step()
      
      # print statistics
      running_loss += loss.data[0]
      if i % printPer == (printPer - 1): # print every 'printPer' mini-batches
        print('[%d, %5d] loss: %.5g' % (epoch+1, i+1, running_loss / printPer))
        running_loss = 0.0

    if epoch % decay_iter == decay_iter-1:
      #lr = args.lr * (0.1 ** (epoch // 30))
      for param_group in optimizer.param_groups:
        print "prev rate: %.4f" % param_group['lr']
        param_group['lr'] = param_group['lr'] * .5
        print "new  rate: %.4f" % param_group['lr']

    # Print this stuff after every epoch
    getLoss(trainloader, epoch, "Train")
    getLoss(devloader, epoch, "Dev")
    running_loss = 0.0
  print('Finished Training')

  torch.save(net, modelName)

def computeAllFeatures(model, dataSet, gpu_id):
  """
  model: a subclass of nn.Module, used to compute features
  dataSet: a torch.DataSet with the inputs and expected outputs.
  Since this function only really needs the inputs, the expected outputs can
  be anything (this function could also take a torch.Tensor in principle,
  but the way other the code is written, this input is convenient.

  returns a torch.Tensor() containing the features of the input
  """
  trainloader = td.DataLoader(dataSet, batch_size=128, shuffle=False, num_workers=0)
  ret = torch.Tensor()
  for i, data in enumerate(trainloader, 0):
    inputs, _ = data
    inputs = ag.Variable(inputs.cuda(gpu_id))
    outputs = model(inputs, False)
    ret = torch.cat([ret, outputs.data.cpu()])
  return ret

def computeAllFeaturesGan(model, dataSet, **kwargs):
    """
    model: a subclass of nn.Module, used to compute features
    dataSet: a torch.DataSet with the inputs and expected outputs.
    Since this function only really needs the inputs, the expected outputs can
    be anything (this function could also take a torch.Tensor in principle,
    but the way other the code is written, this input is convenient.
  
    returns a torch.Tensor() containing the features of the input
    """
    trainloader = td.DataLoader(
            dataSet, batch_size=kwargs["batchSize"], shuffle=False,
            num_workers=0)
    ret = torch.Tensor()
    noise = torch.zeros(kwargs["batchSize"], kwargs["nz"])
    for i, data in enumerate(trainloader, 0):
        inputs, _ = data
        this_batch_len = inputs.size(0)
        inputs = ag.Variable(inputs.cuda(kwargs["gpu_id"]))
        noise.normal_(0, 1)
        outputs = model(
                noise[:this_batch_len], inputs, train=True,
                ignoreCond=kwargs.get("ignoreCond", False))
        ret = torch.cat([ret, outputs.data.cpu()])
    return ret
#if __name__ == '__main__':
#  makeAllData()
# Training the image transformation logic using a neural network.
