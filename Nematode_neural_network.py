# -*- coding: utf-8 -*-
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from thop import profile


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.channel = channel = 8

        self.conv = nn.Conv2d(3, channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(channel)

        self.conv1 = nn.Conv2d(channel, channel, 3, padding=1)
        self.conv2 = nn.Conv2d(channel * 2, channel, 3, padding=1)
        self.conv3 = nn.Conv2d(channel * 3, channel, 3, padding=1)
        self.conv4 = nn.Conv2d(channel * 7, channel * 2, 3, padding=1)

        self.bn1 = nn.BatchNorm2d(channel * 2)

        self.pool = nn.MaxPool2d(2, 2)
        self.dp = nn.Dropout2d(0.1)

        self.fc1 = nn.Linear(channel * 2 * 4 * 4, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        x = self.pool(self.bn(F.relu(self.conv(x))))

        s1 = self.bn(F.relu(self.conv1(x)))
        s2 = self.bn(F.relu(self.conv1(x)))
        s3 = self.bn(F.relu(self.conv1(x)))
        s4 = self.bn(F.relu(self.conv1(x)))
        s5 = self.bn(F.relu(self.conv1(x)))
        s6 = self.bn(F.relu(self.conv1(x)))
        s7 = self.bn(F.relu(self.conv1(x)))
        s8 = self.bn(F.relu(self.conv1(x)))
        s9 = self.bn(F.relu(self.conv1(x)))
        s10 = self.bn(F.relu(self.conv1(x)))

        i1 = self.bn(F.relu(self.conv2(torch.cat((s1, s2), dim=1))))
        i2 = self.bn(F.relu(self.conv3(torch.cat((s3, s4, s6), dim=1))))
        i3 = self.bn(F.relu(self.conv3(torch.cat((s5, s6, s7), dim=1))))
        i4 = self.bn(F.relu(self.conv3(torch.cat((s6, s8, s9), dim=1))))
        i5 = self.bn(F.relu(self.conv1(s10)))

        m1 = self.bn(F.relu(self.conv1(i1)))
        m2 = self.bn(F.relu(self.conv2(torch.cat((s4, i2), dim=1))))
        m3 = self.bn(F.relu(self.conv1(i3)))
        m4 = self.bn(F.relu(self.conv1(i4)))
        m5 = self.bn(F.relu(self.conv2(torch.cat((i4, i5), dim=1))))
        m6 = self.bn(F.relu(self.conv1(i5)))
        m7 = self.bn(F.relu(self.conv1(s10)))

        x = self.pool(torch.cat((m1, m2, m3, m4, m5, m6, m7), dim=1))
        x = self.dp(self.pool(self.bn1(F.relu(self.conv4(x)))))

        x = x.view(-1, self.channel * 2 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    batch_size = 16
    num_workers = 8
    file_name = './worm_1.csv'

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    net = Net()
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    data_sample = iter(trainloader)
    images_sample, _ = next(data_sample)
    images_sample = images_sample.to(device)
    flops, params = profile(model=net, inputs=(images_sample,))
    print('Parameters:%d' % params)
    print('Flops:%d' % flops)
    with open(file_name, 'a') as file_object:
        file_object.write('%d,%d\n' % (params, flops))

    for epoch in range(200):
        running_loss = 0.0
        iter_number = 0

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            iter_number = i
        print('[%d] loss: %.3f' % (epoch + 1, running_loss / (iter_number + 1)))
        with open(file_name, 'a') as file_object:
            file_object.write('%d,%.3f,' % (epoch + 1, running_loss / (iter_number + 1)))

        correct = 0
        total = 0
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        print('Accuracy: %d%%' % (100 * correct / total))
        with open(file_name, 'a') as file_object:
            file_object.write('%d%%,' % (100 * correct / total))

        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for i, data in enumerate(testloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for j in range(batch_size):
                    label = labels[j]
                    class_correct[label] += c[j].item()
                    class_total[label] += 1
        with open(file_name, 'a') as file_object:
            for i in range(10):
                if i == 9:
                    file_object.write('%d%%\n' % (100 * class_correct[i] / class_total[i]))
                else:
                    file_object.write('%d%%,' % (100 * class_correct[i] / class_total[i]))

    print('Finished Training')
