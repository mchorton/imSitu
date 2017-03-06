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

TORCHCOMPVERBLENGTH = "data/pairLearn/comptrain.pyt_verb2Len" 
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

class ImTransNet(nn.Module):
    def __init__(self, imFeatures, verb2Len, depth, nHidden, nOutput, nWords, WESize, nVRs, vrESize):
        super(ImTransNet, self).__init__()

        nInput = imFeatures + 2 * WESize + vrESize + 6 * (WESize + vrESize)
        print nInput
        self.input_layer = nn.Linear(nInput, nHidden)
        self.hidden_layers = nn.ModuleList( [nn.Linear(nHidden, nHidden) for i in range(0,depth)])
        self.output = nn.Linear(nHidden, nOutput)
        
        # TODO try initializing these differently.
        #print "CONSIDER MODIFYING INITIALIZATION OF EMBEDDINGS"
        self.wordEmbedding = nn.Embedding(nWords, WESize)
        self.vrEmbedding = nn.Embedding(nVRs, vrESize)
       
        self.context_wordEmbedding = nn.Embedding(nWords+1, WESize)
        self.context_vrEmbedding = nn.Embedding(nVRs+1, vrESize)
 
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
        context_vectors = []
        for i in range(0, 6):
          context_vectors.append(self.context_vrEmbedding(context[:,2*i].long()).view(len(x), -1))
          context_vectors.append(self.context_wordEmbedding(context[:,1 + 2*i].long()).view(len(x),-1))

        roles = x[:,12]
        #print "shape of roles: %s" % str(roles.size())
        #print "roles is: %s" % str(roles)
        roleEmbeddings = self.vrEmbedding(roles.long()) # This is now [batchDim][1][embedding size]
        #print "shape of roleEmbeddings: %s" % str(roleEmbeddings.size())
        roleEmbeddings = roleEmbeddings.view(len(x), -1)
        #print "after reshape , shape of roleEmbeddings: %s" % str(roleEmbeddings.size())

        words = x[:,13:15]
        wordEmbeddings = self.wordEmbedding(words.long()) # This is now [batchDim][1][embedding size]
        wordEmbeddings = wordEmbeddings.view(len(x), -1)
        f = x[:, 15:]
        x = torch.cat( ( context_vectors + [roleEmbeddings, wordEmbeddings, f] ), 1)
        #x = torch.cat( ( [roleEmbeddings, wordEmbeddings, f] ), 1)
        #print "final x.shape: %s" % str(x.size())
        x = F.dropout(F.leaky_relu(self.input_layer(x)), training=train) 
        for i in range(0, len(self.hidden_layers)):
          x = F.dropout(F.leaky_relu(self.hidden_layers[i](x)), training=train)
          if i == 0: x_prev = x
          #add skip connections for depth
          if i > 0 and i % 2 == 0: 
            x = x + x_prev
            x_prev = x
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

  makeData(TORCHCOMPTRAINDATA, TORCHCOMPDEVDATA, COMPFEATDIR, VRNDATA)
  makeData(TORCHREGTRAINDATA, TORCHREGDEVDATA, REGFEATDIR, VRNDATA)
  makeData(TORCHCOMPTRAINDATATEST, TORCHCOMPDEVDATATEST, COMPFEATDIR, VRNDATATEST)
  makeData(TORCHREGTRAINDATATEST, TORCHREGDEVDATATEST, REGFEATDIR, VRNDATATEST)

def makeData(trainLoc, devLoc, featDir, vrndatafile):
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
  print "Saving img names to %s" % devImgNameFile
  json.dump(list(devImgNames), open(devImgNameFile, "w+"))

  dataSet = makeDataSet(trainLoc, featDir, vrnData, trainImgNames)
  print "Saving data to %s" % trainLoc
  torch.save(dataSet, trainLoc)

  dataSet = makeDataSet(devLoc, featDir, vrnData, devImgNames)
  print "Saving data to %s" % devLoc
  torch.save(dataSet, devLoc)

def makeDataSet(trainLoc, featureDirectory, vrnData, whitelistedImgNames):
  """
  Create a pytorch TensorDataset at 'outFileName'. It contains input suitable
  for the models trained to generate image features.
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

  print "done reading data"
  
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

  print "done getting verb lengths"

  rolesAsTuples = all_roles 
  role2Int = {role:index for index, role in enumerate(sorted(list(rolesAsTuples)))}
  allNounsUsed = all_nouns 
  noun2Int = {noun: index for index, noun in enumerate(sorted(list(allNounsUsed)))}
  verb2Int = {verb: index for index, verb in enumerate(sorted(list(all_verbs)))}
  print "done getting maps"

  verb2Len = {}
  for (k,v) in verb2Int.items(): verb2Len[verb2Int[k]] = v

  torch.save(role2Int, "%s_role2Int" % trainLoc)
  torch.save(noun2Int, "%s_noun2Int" % trainLoc)
  torch.save(verb2Int, "%s_verb2Int" % trainLoc)
  torch.save(verb2Len, "%s_verb2Len" % trainLoc)

  print "verb2Int size: %d" % len(verb2Int)
  print "role2int size: %d" % len(role2Int)
  print "noun2Int size: %d" % len(noun2Int)

  imToFeatures = { name: np.fromfile("%s%s" % (featureDirectory, name), dtype=np.float32) for name in im1Names } # Should have all names in it.
  imToFeatures = {name: np.array(values, dtype=np.float64) for name, values in imToFeatures.iteritems() }

  # Loop through, building train and dev sets.
  xData = []
  yData = []
  wasted = 0
  for i, (score, im1Name, im2Name, tRole, noun1, noun2, an1, an2) in enumerate(vrnData):
    if i % 10000 == 10000-1:
      print "iteration %d of %d" % (i, len(vrnData))
    anitems = [] 
    for (k,v) in an1.items():
      rid = role2Int[make_tuple(k)]
      nid = noun2Int[v]
      anitems.append((rid,nid))
    while len(anitems) < 6:
      anitems.append((len(role2Int), len(noun2Int)))
   
    anitems = sorted(anitems, key = lambda x : x[0]) 
    indexes = []
    for (k,v) in anitems: indexes += [k,v]
 
    x = list(indexes) + [role2Int[tuple(tRole)], noun2Int[noun1], noun2Int[noun2]] + list(imToFeatures[im1Name]) 
    y = list(imToFeatures[im2Name]) + [score]
    #print str(x[0:4]) + " , " + str(y[0]) + "," + str(y[-1])
    if im1Name in whitelistedImgNames:
      xData.append(x)
      yData.append(y)

  dataSet = td.TensorDataset(torch.Tensor(xData), torch.Tensor(yData))
  return dataSet

def runModelTest(modelName, modelType, lr=0.0001, epochs=10, depth=2):
  runModel(modelName, modelType, depth, lr, TORCHCOMPTRAINDATATEST, TORCHCOMPDEVDATATEST, epochs=epochs)

def runModel(modelName, modelType, depth=2, lr=0.0001, trainLoc=TORCHCOMPTRAINDATA, devLoc=TORCHCOMPDEVDATA, epochs=10, weight_decay=0.01, nHidden = 1024, batchSize=128):
  # Get the data
  #trainLoc = TORCHCOMPTRAINDATATEST
  #devLoc = TORCHCOMPDEVDATATEST
  print "Loading training data from %s" % str(trainLoc)
  train = torch.load(trainLoc)
  
  trainloader = td.DataLoader(train, batch_size=batchSize, shuffle=True, num_workers=4)

  print "Loading dev data from %s" % str(devLoc)
  dev = torch.load(devLoc)
  devloader = td.DataLoader(dev, batch_size=batchSize, shuffle=True, num_workers=4)

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
  verb2Len = torch.load("%s_verb2Len" % trainLoc)

  nVRs = max(role2Int.values()) + 1
  nWords = max(noun2Int.values()) + 1
  WESize = 128
  vrESize = 128
  print "nVRs: %d" % nVRs
  print "nWords: %d" % nWords

  if modelType == "nn":
    net = ImTransNet(indim - (3+12), verb2Len, depth, nHidden, outdim, nWords, WESize, nVRs, vrESize).cuda() # ?
  elif modelType == "dot":
    net = MatrixDot(nWords, nVRs, int((vrESize + WESize) / 2)).cuda()
  elif modelType == "cross":
    net = MatrixCross(nWords, WESize, nVRs, vrESize, 1024).cuda()
  else:
    raise Exception("invalid modelType '%s'" % str(modelType))

  criterion = nn.MSELoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=.9, weight_decay=.00005)
  #optimizer = optim.Adam(net.parameters(), lr, weight_decay=weight_decay)
  #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=weight_decay)
  def getLoss(loader, epoch, name):
    running_loss = 0.
    nIter = 0
    for i, data in enumerate(loader, 0):
      inputs, labelsAndScore = data
      scores = ag.Variable(labelsAndScore[:,-1].cuda())
      labels = labelsAndScore[:,:-1].cuda()
      inputs, labels = ag.Variable(inputs.cuda()), ag.Variable(labels.cuda())
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

      scores = ag.Variable(labelsAndScore[:,-1].cuda()) # Last row is scores
      labels = labelsAndScore[:,:-1]
      
      # wrap them in Variable
      inputs, labels = ag.Variable(inputs.cuda()), ag.Variable(labels.cuda())
      
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

    if epoch % 10 == 9:
      #lr = args.lr * (0.1 ** (epoch // 30))
      for param_group in optimizer.param_groups:
        print "prev rate: %.4f" % param_group['lr']
        param_group['lr'] = param_group['lr'] * .8
        print "new  rate: %.4f" % param_group['lr']

    # Print this stuff after every epoch
    getLoss(trainloader, epoch, "Train")
    getLoss(devloader, epoch, "Dev")
    running_loss = 0.0
  print('Finished Training')

  torch.save(net, modelName)

def computeAllFeatures(model, dataSet):
  """
  model: a subclass of nn.Module, used to compute features
  dataSet: a torch.DataSet with the inputs and expected outputs.
  Since this function only really needs the inputs, the expected outputs can
  be anything (this function could also take a torch.Tensor in principle,
  but the way other the code is written, this input is convenient.

  returns a torch.Tensor() containing the features of the input
  """
  trainloader = td.DataLoader(dataSet, batch_size=128, shuffle=False, num_workers=4)
  ret = torch.Tensor()
  for i, data in enumerate(trainloader, 0):
    inputs, _ = data
    inputs = ag.Variable(inputs.cuda())
    outputs = model(inputs, False)
    ret = torch.cat([ret, outputs.data.cpu()])
  return ret

if __name__ == '__main__':
  makeAllData()
# Training the image transformation logic using a neural network.
