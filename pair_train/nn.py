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
    def __init__(self, imFeatures, nHidden, nOutput, nWords, WESize, nVRs, vrESize):
        super(ImTransNet, self).__init__()

        nInput = imFeatures + 2 * WESize + vrESize
        self.hidden1 = nn.Linear(nInput, nHidden)
        self.hidden2 = nn.Linear(nHidden, nHidden)
        self.hidden3 = nn.Linear(nHidden, nHidden)
        self.output = nn.Linear(nHidden, nOutput) # TODO can I do this at "forward()" time?
        self.wordEmbedding = nn.Embedding(nWords, WESize)
        self.vrEmbedding = nn.Embedding(nVRs, vrESize)
        print "WESize: %d" % WESize
        print "vrESize: %d" % vrESize
        """
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        """

    def forward(self, x):
        """
        x looks like [role, noun1, noun2, features...]
        """
        # TODO get the embedding vector.
        # TODO this will be slow! I think? Might happen on CUDA though.
        roles = x[:,0]
        #print "shape of roles: %s" % str(roles.size())
        #print "roles is: %s" % str(roles)
        roleEmbeddings = self.vrEmbedding(roles.long()) # This is now [batchDim][1][embedding size]
        #print "shape of roleEmbeddings: %s" % str(roleEmbeddings.size())
        roleEmbeddings = roleEmbeddings.view(len(x), -1)
        #print "after reshape , shape of roleEmbeddings: %s" % str(roleEmbeddings.size())

        words = x[:,1:3]
        #print "shape of words: %s" % str(words.size())
        wordEmbeddings = self.wordEmbedding(words.long()) # This is now [batchDim][1][embedding size]
        #print "shape of wordEmbeddings: %s" % str(wordEmbeddings.size())
        wordEmbeddings = wordEmbeddings.view(len(x), -1)
        #print "after reshape , shape of wordEmbeddings: %s" % str(wordEmbeddings.size())

        x = torch.cat([roleEmbeddings, wordEmbeddings, x[:,3:]], 1)
        #print "final x.shape: %s" % str(x.size())

        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.relu(self.hidden3(x))
        x = self.output(x)
        return x

        #exit(1)

        """
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv2(x)), 2) # If the size is a square you can only specify a single number
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
        """
    
    """
    def num_flat_features(self, x):
        size = x.size()[1:] # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    """
# Our json object looks like:
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
  vrnData = json.load(open(vrndatafile))
  print "There are %d valid pairs" % len(vrnData)
  # Split it into lists of the roles, noun1, noun2, verb
  scores = [pt[0] for pt in vrnData]
  im1Names = [pt[1] for pt in vrnData]
  im2Names = [pt[2] for pt in vrnData]
  tRoles = [pt[3] for pt in vrnData] # This is a tuple btw.
  n1s = [pt[4] for pt in vrnData]
  n2s = [pt[5] for pt in vrnData]

  # TODO need to convert some of these to numbers.
  rolesAsTuples = set([tuple(e) for e in tRoles])
  role2Int = {role:index for index, role in enumerate(sorted(list(rolesAsTuples)))}
  #def nounToInt(noun):
  #  return int(noun[1:])
  allNounsUsed = set(n1s)
  noun2Int = {noun: index for index, noun in enumerate(sorted(list(allNounsUsed)))}

  torch.save(role2Int, "%s_role2Int" % trainLoc)
  torch.save(noun2Int, "%s_noun2Int" % trainLoc)

  print "role2Int size: %d" % len(role2Int)
  print "noun2Int size: %d" % len(noun2Int)

  imToFeatures = { name: np.fromfile("%s%s" % (featDir, name), dtype=np.float32) for name in im1Names } # Should have all names in it.
  imToFeatures = {name: np.array(values, dtype=np.float64) for name, values in imToFeatures.iteritems() }

  # Choose which images are part of train, and which are part of dev
  allNames = list(set(im1Names))
  random.shuffle(allNames)
  trainIm = set(allNames[:len(allNames)/2])
  devIm = set(allNames[len(allNames)/2:])

  print len(trainIm)
  print len(devIm)

  # Loop through, building train and dev sets.
  xtrain = []
  ytrain = []
  xdev = []
  ydev = []
  wasted = 0
  for i, (score, im1Name, im2Name, tRole, noun1, noun2, an1, an2) in enumerate(vrnData):
    x = [role2Int[tuple(tRole)]] + [noun2Int[noun1]] + [noun2Int[noun2]] + list(imToFeatures[im1Name])
    y = list(imToFeatures[im2Name])
    if im1Name in trainIm and im2Name in trainIm:
      xtrain.append(x)
      ytrain.append(y)
    elif im1Name in devIm and im2Name in devIm:
      xdev.append(x)
      ydev.append(y)
    else:
      wasted += 1
  print "Train set has %d points" % len(xtrain)
  print "Dev set has %d points" % len(xdev)
  print "Samples wasted: %d" % wasted

  trainData = td.TensorDataset(torch.Tensor(xtrain), torch.Tensor(ytrain))
  devData = td.TensorDataset(torch.Tensor(xdev), torch.Tensor(ydev))

  print "Saving train data to %s" % trainLoc
  torch.save(trainData, trainLoc)
  print "Saving dev data to %s" % devLoc
  torch.save(devData, devLoc)

# TODO get this to run on the GPU
def runModel(modelName, lr=0.0001, trainLoc=TORCHCOMPTRAINDATA, devLoc=TORCHCOMPDEVDATA, epochs=10):
  # Get the data
  #trainLoc = TORCHCOMPTRAINDATATEST
  #devLoc = TORCHCOMPDEVDATATEST
  print "Loading training data from %s" % str(trainLoc)
  train = torch.load(trainLoc)
  trainloader = td.DataLoader(train, batch_size=128, shuffle=True, num_workers=4)

  print "Loading dev data from %s" % str(devLoc)
  dev = torch.load(devLoc)
  devloader = td.DataLoader(dev, batch_size=128, shuffle=True, num_workers=4)

  print "found %d train datapoints" % len(train)
  print "found %d dev datapoints" % len(dev)

  x1, y1 = train[0]
  indim = len(x1) # Should be the number of dimensions.
  outdim = len(y1) # Should be the number of dimensions.
  print "Got indim as %s" % str(indim)
  print "Got outdim as %s" % str(outdim)
  nHidden = 100

  printPer = 200

  # TODO add in the nouns, and the role. Get proper data.
  #def __init__(self, nInput, nHidden, nOutput, nWords, WESize, nVRs, vrESize):

  role2Int = torch.load("%s_role2Int" % trainLoc)
  noun2Int = torch.load("%s_noun2Int" % trainLoc)

  nVRs = max(role2Int.values()) + 1
  nWords = max(noun2Int.values()) + 1
  WESize = int(nWords)
  vrESize = int(nVRs)
  print "nVRs: %d" % nVRs
  print "nWords: %d" % nWords
  net = ImTransNet(indim - 3, nHidden, outdim, nWords, WESize, nVRs, vrESize).cuda() # ?

  # TODO make a custom loss.
  criterion = nn.MSELoss()
  optimizer = optim.SGD(net.parameters(), lr)

  # TODO run on GPU
  for epoch in range(epochs):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs
      inputs, labels = data
      
      # wrap them in Variable
      inputs, labels = ag.Variable(inputs.cuda()), ag.Variable(labels.cuda())
      
      # zero the parameter gradients
      optimizer.zero_grad()
      
      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()        
      optimizer.step()
      
      # print statistics
      running_loss += loss.data[0]
      if i % printPer == (printPer - 1): # print every 'printPer' mini-batches
        print('[%d, %5d] loss: %.5g' % (epoch+1, i+1, running_loss / printPer))
        running_loss = 0.0

    # Print this stuff after every epoch
    for i, data in enumerate(trainloader, 0):
      inputs, labels = data
      inputs, labels = ag.Variable(inputs.cuda()), ag.Variable(labels.cuda())
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      running_loss += loss.data[0] # unsure if this is normalized.
    print('Train Set [Epoch %d] loss: %.5g' % (epoch+1, running_loss))
    running_loss = 0.

    # print the loss over the dev set
    for i, data in enumerate(devloader, 0):
      inputs, labels = data
      inputs, labels = ag.Variable(inputs.cuda()), ag.Variable(labels.cuda())
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      running_loss += loss.data[0] # unsure if this is normalized.
    print('Dev Set [Epoch %d] loss: %.5g' % (epoch+1, running_loss))
    running_loss = 0.0

        #loss = criterion(net(inputs)
        #trainloader = td.DataLoader(train, batch_size=128, shuffle=True, num_workers=2)
  print('Finished Training')

  torch.save(net, modelName)

if __name__ == '__main__':
  makeAllData()
