#''Author - Ekta Bhojwani''
#MSEE,UNCC
print(##########Importing_Libraries######################)
import matplotlib.pyplot as plt
import torchvision
import numpy as np
import torch
import os
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from utils import progress_bar

#Transformation on Training
print('==> Preparing data..')
transform_train = transforms.Compose([transforms.Resize((200,200)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Transformation on Testing
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

#Loading the train data
total_epoch = 300
train_data = torchvision.datasets.ImageFolder(root='./maria/labeled/train', transform=transform_train)
trainloader = torch.utils.data.DataLoader(train_data, batch_size=80, shuffle=True, num_workers=0)

#Loading the test data
test_data = torchvision.datasets.ImageFolder(root='./maria/labeled/validation',  transform=transform_test)
testloader = torch.utils.data.DataLoader(test_data, batch_size=80, shuffle=True, num_workers=0)

print(trainloader.dataset.classes)
tstart = datetime.now() #Start_calculating_time
print('start time is', tstart)

##torch.set_grad_enabled(False)
# Enabling the GPU device to mount access to the pytorch on the device, In my case Cuda available GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#defining the Resnet-50 model pretrained on Imagenet weights
model = models.resnet50(pretrained=True)

# Freeze parameters so we don't backprop through them, while freezing
for param in model.parameters():
    param.requires_grad = False #True(Whenever we want to enable backpropogation to update the gradients)
    print(model.parameters)

 

# The actual finetuning starts on adding the Fully Connected Layers with RELU, Dropout,Softmax and input features of the last
#layer and out_features corresponding to the number of classes.  
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 #nn.Dropout(0.2),# add a batch Norm if you want
                                 nn.Linear(512, 3),
                                 nn.LogSoftmax(dim=1))
#Sometimes, when only the FC layers are fine-tuned, we nackpropogate through them.
#Hence, requires_grad/gradients = True
for param in model.fc.parameters():
    param.requires_grad = True 
model = model.to(device)
print('###########################################################################')
#for name, param in model.fc.named_parameters():
    #print()
    #print('param_name_grad', param.requires_grad )
    #print('name', name)
    #print('param.shape: ', param.shape)

# Crossentropy Loss to the model outputs and predictions. 
criterion = nn.NLLLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.0003) #Optimizer optimises the gradients on backprop

#again mounting the model
model.to(device)

start_epoch = 0
train_losses, test_losses = [], []
train_accuracy, test_accuracy = [], []
# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    #train_losses, test_losses = [], []
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        #print('train')
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    train_losses.append(train_loss/(batch_idx+1))
    train_accuracy.append(100.*correct/total) 
            
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            #print(correct)
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        test_accuracy.append(100.*correct/total) 
            
        test_losses.append(test_loss/(batch_idx+1))

for epoch in range(start_epoch, start_epoch+total_epoch):
    train(epoch)
    test(epoch)
tend = datetime.now()

ep =  [i for i in range(total_epoch)]

#Plotting the Losses graph
plt.plot(ep , train_losses, label='Training loss')
plt.plot(ep , test_losses, label='Validation loss')
plt.title('loss_convfrozen_2fclayers')
plt.savefig('./graphs/maria_graphs/loss/loss_convfrozen_2layers_2ndtime')
plt.plot
plt.legend(frameon = False)
plt.show()

#Plotting the Accuracy graph
plt.plot(ep, train_accuracy , label='Training accuracy')
plt.plot(ep,test_accuracy , label='Validation accuracy')
plt.title('accuracy_convfrozen_2layers')
plt.savefig('./graphs/maria_graphs/accuracy/accuracy_convfrozen_2layers_2ndtime')
plt.plot
plt.legend(frameon = False)
plt.show()
tend = datetime.now()

delta = tend - tstart
print('training time is', delta)
torch.save(model.state_dict(), './paths/maria/Resnet_classifier_2.pth')
print('model is saved')

print(####################Freezing first 7 layers##########################)

#When freezing layers you just need to change the auto_grad function in the model architecture
'''model = models.resnet50(pretrained=True)
ct = 0
for child in model.children():
    ct += 1
    if ct < 7:
        for param in child.parameters():
            param.requires_grad = False
    else:
        for param in child.parameters():
            param.requires_grad = True

model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 #nn.Dropout(0.2),# add a batch Norm if you want
                                 nn.Linear(512, 3),
                                 nn.LogSoftmax(dim=1))
for param in model.fc.parameters():
    param.requires_grad = True
    print(model.fc.parameters) 
model = model.to(device)


