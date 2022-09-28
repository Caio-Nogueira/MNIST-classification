from doctest import Example
from unittest import BaseTestSuite
import torch
import torchvision
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


n_epochs = 3
batch_size_train = 64
batch_size_test = 1000
learning_rate = 0.01

random_seed = 1
torch.backends.cudnn.enabled = False

# Loading the dataset
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)




class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.num_epochs = 3
        self.lr = 0.01
        self.batch_size_train = 64
        self.batch_size_test = 1000

        self.loss_history = []

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(64)	
        
        self.max_pool1 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(64)

        self.max_pool2 = nn.MaxPool2d(2)

        # self.conv2_drop = nn.Dropout2d() # perform regularization to avoid overfitting        
        self.to(self.device)
        
        self.input_dims = self.calc_input_dims()
        
        # fully connected layers
        self.fc1 = nn.Linear(self.input_dims, 64)
        self.fc2 = nn.Linear(64, 10)

        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.loss = nn.CrossEntropyLoss()
        

    def calc_input_dims(self):
        # computes the dimensions of the input to the fully connected layer
        batch_data = torch.zeros((1,1,28,28)).to(self.device)

        batch_data = self.conv1(batch_data)
        batch_data = self.conv2(batch_data)
        batch_data = self.conv3(batch_data)

        batch_data = self.max_pool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)

        batch_data = self.max_pool2(batch_data)

        return int(np.prod(batch_data.size()))
        

    def forward(self, batch_data):
        batch_data = torch.tensor(batch_data).to(self.device)    
        
        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data =  self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data =  self.bn3(batch_data)
        batch_data = F.relu(batch_data)
        
        batch_data = self.max_pool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data =  self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.max_pool2(batch_data)

        batch_data = batch_data.view(-1, self.input_dims)

        batch_data = self.fc1(batch_data)
        batch_data = F.relu(batch_data)
        batch_data = self.fc2(batch_data)

        return F.softmax(batch_data)


    def _train(self):
        self.train()

        for i in range(self.num_epochs):
            epoch_loss = 0
            epoch_accs = []

            for idx, (input, label) in enumerate(train_loader):
                self.optimizer.zero_grad() #reset the gradients
                label = label.to(self.device)
                pred = self.forward(input)
                loss = self.loss(pred, label)

                pred = F.softmax(pred, dim=1)
                classes = torch.argmax(pred, dim=1)

                wrong = torch.where(classes != label, torch.tensor([1.]).to(self.device), 
                                      torch.tensor([0.]).to(self.device))
                
                # print(wrong.get_device())
                acc = 1 - torch.sum(wrong) / self.batch_size_train

                epoch_accs.append(acc.item())
                epoch_loss += loss.item()

                loss.backward() # backpropagate the loss
                self.optimizer.step()

            print(f"Finished epoch {i} with mean accuracy {np.mean(epoch_accs)} and total loss {epoch_loss}")
            self.loss_history.append(epoch_loss)


    def _test(self, test_loader):
        self.eval()
        testing_loss = 0
        acc = []
        with torch.no_grad():

            for idx, (input, label) in enumerate(test_loader):
                label = label.to(self.device)

                pred = self.forward(input)
                loss = self.loss(pred, label)

                pred = F.softmax(pred, dim=1)
                classes = torch.argmax(pred, dim=1)

                wrong = torch.where(classes != label, torch.tensor([1]).to(self.device), 
                                        torch.tensor([0]).to(self.device))

                # build list with missclassified tensors

                
                acc = 1 - torch.sum(wrong) / self.batch_size_test

                testing_loss += loss.item()


        print(f"Finished testing epoch {idx} with mean accuracy {acc} and total loss {testing_loss}")
        self.loss_history.append(testing_loss)
        return classes

if __name__ == "__main__":

    CNN = Network()
    # CNN = Network().cuda()
    CNN._train()
    torch.save(CNN.state_dict(), 'CNN.pth')

    plt.plot(CNN.loss_history)
    plt.show()

