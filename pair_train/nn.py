# Training the image transformation logic using a neural network.
import json
import torch.nn as nn

class ImTransNet(nn.Module):
    def __init__(self, nInput, nHidden, nOutput):
        super(Net, self).__init__()

        self.hidden = nn.Linear(nInput, nHidden)
        self.output = nn.Linear(nHidden, nOutput) # TODO can I do this at "forward()" time?
        """
        self.conv1 = nn.Conv2d(1, 6, 5) # 1 input image channel, 6 output channels, 5x5 square convolution kernel
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120) # an affine operation: y = Wx + b
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)
        """

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.output(x)
        return x
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
def makeData():
  vrnData = json.load(open("vecStyle/vrnData.json"))
  # Split it into lists of the roles, noun1, noun2, verb
  scores = [pt[0] for pt in vrnData]
  im1Names = [pt[1] for pt in vrnData]
  im2Names = [pt[2] for pt in vrnData]
  tRoles = [pt[3] for pt in vrnData]
  n1s = [pt[4] for pt in vrnData]
  n2s = [pt[5] for pt in vrnData]

  im1Features = []
  for name in im1Names:
    im1Features.append(np.fromfile("vecStyle/%s" % name))

  im2Features = []
  for name in im1Names:
    im2Features.append(np.fromfile("vecStyle/%s" % name))
  
  # TODO save this in a way that is easily read.

# TODO get this to run on the GPU
def runModel():
  # Get the data
  data = load_data()

  nSamples = 10
  dim = 20
  nHidden = 5

  i1 = torch.rand(nSamples, dim)
  i2 = torch.rand(nSamples, dim) * 0.01 + 5 * i1

  # TODO make a custom loss.
  criterion = nn.MSELoss()

  # TODO add in the nouns, and the role. Get proper data.
  net = ImTransNet(dim, nHidden, dim)

  for epoch in range(2):
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs
      inputs, labels = data
      
      # wrap them in Variable
      inputs, labels = Variable(inputs), Variable(labels)
      
      # zero the parameter gradients
      optimizer.zero_grad()
      
      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()        
      optimizer.step()
      
      # print statistics
      running_loss += loss.data[0]
      if i % 2000 == 1999: # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch+1, i+1, running_loss / 2000))
        running_loss = 0.0
  print('Finished Training')
  
  

net
