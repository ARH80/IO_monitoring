import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from autoaugment import ImageNetPolicy

torch.set_default_tensor_type(torch.cuda.FloatTensor)

data_dir = r'D:\food\images'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(), ImageNetPolicy(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + r'\train_noise', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + r'\valid', transform=test_transforms)
testdata=datasets.ImageFolder(data_dir + r'\test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=128)
test_loader=torch.utils.data.DataLoader(testdata, batch_size=64)


model =models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 101)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier
import time
for device in ['cuda']:

    criterion = nn.NLLLoss()
    # Only train the classifier parameters, feature parameters are frozen
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    model.to(device)

    for ii, (inputs, labels) in enumerate(trainloader):

        # Move input and label tensors to the GPU
        inputs, labels = inputs.to(device), labels.to(device)

        start = time.time()

        outputs = model.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if ii==3:
            break
        
    print(f"Device = {device}; Time per batch: {(time.time() - start)/3:.3f} seconds")

# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#model = models.densenet201(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024,512),nn.LeakyReLU(),nn.Linear(512,256),nn.LeakyReLU(),nn.Linear(256,101))

criterion = nn.CrossEntropyLoss()

# Only train the classifier parameters, feature parameters are frozen
#optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


#optimizer = optim.Adam(model.parameters(), lr=0.00001)
optimizer = optim.Adam(model.classifier.parameters(), lr=0.001, betas=[0.9, 0.999])   

import numpy as np
import time
def train(n_epochs,trainloader,testloader, resnet, optimizer, criterion, save_path):
    """returns trained model"""
    # initialize tracker for minimum validation loss
    valid_loss_min = np.Inf 
    running_loss=0
    
  
    for epoch in range(n_epochs):
        
        
        for inputs, labels in trainloader:
            
        # Move input and label tensors to the default device
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            start = time.time()
            logps = resnet(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
        
        resnet.eval()
        valid_loss=0
        accuracy=0
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.cuda(), labels.cuda()
                logps = resnet(inputs)
                batch_loss = criterion(logps, labels)
                valid_loss += batch_loss.item()
                    
                    # Calculate accuracy
                
                top_p, top_class = logps.topk(1, dim=1)
                equals = top_class == labels.view(*top_class.shape)
                accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
           
        
            if valid_loss <= valid_loss_min:
                print("Validation loss decreased  Saving model")
                torch.save(resnet.state_dict(),'food_classifier_densenet121_noise.pt')
                valid_loss_min=valid_loss
                
            print(f"Device = cuda; Time per batch: {(time.time() - start):.3f} seconds")       
            print(f"Epoch /{n_epochs}.. "
                  f"Train loss: {running_loss/len(trainloader):.3f}.. "
                  f"Test loss: {valid_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            resnet.train()      

train(30,trainloader,testloader, model, optimizer, criterion,'model_vowel_consonant.pt')

model.load_state_dict(torch.load('food_classifier_densenet121_noise.pt'))
torch.save(model.state_dict(),'food101.pth')
model.load_state_dict(torch.load('food101.pth'))
valid_loss=0
accuracy=0
with torch.no_grad():
  model.eval()
  for images,labels in test_loader:
    images,lables=images.cuda(),labels.cuda()
    logps = model(images)
    batch_loss = criterion(logps, labels)
    valid_loss += batch_loss.item()
    top_p, top_class = logps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(valid_loss/len(test_loader))
print(accuracy/len(test_loader))

model.load_state_dict(torch.load('food_classifier_final_1.pt'))
valid_loss=0
accuracy=0
with torch.no_grad():
  model.eval()
  for images,labels in testloader:
    images,lables=images.cuda(),labels.cuda()
    logps = model(images)
    batch_loss = criterion(logps, labels)
    valid_loss += batch_loss.item()
    top_p, top_class = logps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(valid_loss/len(testloader))
print(accuracy/len(testloader))

model.load_state_dict(torch.load('food_classifier.pt'))
valid_loss=0
accuracy=0
with torch.no_grad():
  model.eval()
  for images,labels in trainloader:
    images,lables=images.cuda(),labels.cuda()
    logps = model(images)
    batch_loss = criterion(logps, labels)
    valid_loss += batch_loss.item()
    top_p, top_class = logps.topk(1, dim=1)
    equals = top_class == labels.view(*top_class.shape)
    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
print(valid_loss/len(trainloader))
print(accuracy/len(trainloader))

import numpy as np
import cv2

class_names = [item for item in train_data.classes]
class_names[20]
from PIL import Image
# list of class names by index, i.e. a name can be accessed like class_names[0]
#class_names = [item[4:] for item in train_data.classes]

def predict_food(img_path):
    
    # load the image and return the predicted breed
    img = Image.open(img_path)
    # Resize
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).type(torch.cuda.FloatTensor) 
    img.unsqueeze_(0)
    ps=torch.exp(model(img))
    top_p, top_class = ps.topk(1, dim=1)
    
    
    return top_class.data.cpu().numpy()[0]

img_path=r'D:\Takoyaki.jpg'
img = cv2.imread(img_path)
plt.imshow(img)
plt.show()
class_names[predict_food(img_path).item()]

model=models.vgg19_bn(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
#model.classifier=nn.Sequential(nn.Linear(25088,1000),nn.ReLU(),nn.Dropout(p=0.3),nn.Linear(1000,101),nn.LogSoftmax(dim=1))
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                                       ('fc1', nn.Linear(25088, 1000)),
                                       ('relu', nn.ReLU()),
                                       ('dropout', nn.Dropout(p=0.3)),
                                       ('fc2', nn.Linear(1000, 101)),
                                       ('logsoftmax', nn.LogSoftmax(dim=1))
]))
model.classifier=classifier

model.load_state_dict(torch.load('food_classifier_george.pt'))

from PIL import Image
# list of class names by index, i.e. a name can be accessed like class_names[0]
#class_names = [item[4:] for item in train_data.classes]

def predict_food(img_path):
    
    # load the image and return the predicted breed
    img = Image.open(img_path)
    # Resize
    left_margin = (img.width-224)/2
    bottom_margin = (img.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224
    img = img.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    # Normalize
    img = np.array(img)/255
    mean = np.array([0.485, 0.456, 0.406]) #provided mean
    std = np.array([0.229, 0.224, 0.225]) #provided std
    img = (img - mean)/std
    
    # Move color channels to first dimension as expected by PyTorch
    img = img.transpose((2, 0, 1))
    img = torch.from_numpy(img).type(torch.cuda.FloatTensor) 
    img.unsqueeze_(0)
    ps=torch.exp(model(img))
    top_p, top_class = ps.topk(1, dim=1)
    
    
    return top_class

img_path=r'D:\Takoyaki.jpg'
img = cv2.imread(img_path)
predict_food(img_path)

return_label=predict_food(img_path).item()
return_label
return_label=[return_label,1]
return_label.extend([1] * 99)
print(return_label)
len(return_label)




