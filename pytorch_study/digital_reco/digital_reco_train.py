import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import os

# define if use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# define network architectures
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(  # input_size=(1*28*28)
            nn.Conv2d(1, 6, 5, 1, 2),  # padding=2 guaranteed the sizes of input and output are the same
            nn.ReLU(),  # input_size=(6*28*28)
            nn.MaxPool2d(kernel_size=2, stride=2),  # output_size=(6*14*14)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),  # input_size=(16*10*10)
            nn.MaxPool2d(2, 2)  # output_size=(16*5*5)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(84, 10)

    # Define the forward propagation process, input is x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # The input and output of nn.Linear() are all values of dimension one,
        # so we should flatten the multi-dimensional tensor into one dimension.
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# So that we can manually enter the command line parameters,
# it is to make the style become similar to the Linux command line.
if not os.path.exists('./model'):
    os.makedirs('./model')
parser = argparse.ArgumentParser()
parser.add_argument('--outf', default='./model/',
                    help='folder to output images and model checkpoints')  # Model save path
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  # Model load path
opt = parser.parse_args()

# Hyperparameter setting
EPOCH = 8  # The number of traversing data sets
BATCH_SIZE = 64  #
LR = 0.001  # learning rate

# Define data preprocessing methods
transform = transforms.ToTensor()

# Define training data sets
trainset = tv.datasets.MNIST(
    root='./data/',
    train=True,
    download=True,
    transform=transform)

# Define training batch data
trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# Define test data sets
testset = tv.datasets.MNIST(
    root='./data/',
    train=False,
    download=True,
    transform=transform)

# Define test batch data
testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

# Define loss function and optimization method(use SGD)
net = Net().to(device)
criterion = nn.CrossEntropyLoss()  # Cross entropy loss function, usually used for multi-classification problems
optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)

# train
if __name__ == "__main__":

    for epoch in range(EPOCH):
        sum_loss = 0.0
        # read data
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # gradient clear
            optimizer.zero_grad()

            # forward + backward
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print an average loss for every 100 batches of training
            sum_loss += loss.item()
            if i % 100 == 99:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # Test the accuracy every time you run epoch
        with torch.no_grad():
            correct = 0
            total = 0
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # Take the class with the highest score
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('The recognition accuracy rate of %d epoch: %d%%' % (epoch + 1, (100 * correct / total)))
    torch.save(net, '%s/net_%03d.pth' % (opt.outf, epoch + 1))
