import torch
from training import Network
import torchvision
import numpy as np
import matplotlib.pyplot as plt

batch_size_test = 1000

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)

if __name__ == '__main__':
    
    model = Network()

    model.load_state_dict(torch.load("CNN.pth"))
    predictions = model._test(test_loader)

    for idx, (input, label) in enumerate(test_loader):

        if predictions[idx].item() != label[idx].item():
            #convert tensor to image
            print("Prediction: ", predictions[idx].item())
            img = input[idx].numpy()
            img = np.transpose(img, (1, 2, 0))
            img = img * 0.3081 + 0.1307
            plt.title("Ground Truth: " + str(label[idx].item()))
            plt.legend("Prediction: " + str(predictions[idx].item()))
            plt.imshow(img)
            plt.show()