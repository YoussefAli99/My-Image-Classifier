import argparse
import seaborn as sb
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image


########### Step 1: Get Model h_params
parser = argparse.ArgumentParser(description = 'Pass in the Train and Predict Parameters')
parser.add_argument('--data_dir', type = str, default = 'flowers', help = 'The location of the data files')
parser.add_argument('--save_dir', type = str, default = './', help = 'The location of the saved files')
parser.add_argument('--arch', type = str, default = 'vgg16', choices=['vgg16', 'densenet121', 'alexnet'], help = 'Type in the prefererd Model Architecture')
parser.add_argument('--epochs', type = int, default = 10, help = 'Type in the number of epochs')
parser.add_argument('--lr', type = float, default = 0.001, help = 'Type in the learning rate')
parser.add_argument('--gpu', type = str,  default = 'GPU', choices=['GPU','CPU'], help = 'Type GPU or CPU with uppercase')
parser.add_argument('--input_layers', type = int, default = 25088, help = 'input layers, call multiple times to add input units')
parser.add_argument('--hidden_layers', type = int, default = 4096, help = 'hidden layers, call multiple times to add hidden units')
parser.add_argument('--output_layers', type = int, default = 102, help = 'output layers, call multiple times to add output units')
parser.add_argument('--drop_rate', type = float, default = 0.5, help = 'Type in the number of drop_rate')
parser.add_argument('--topk', type = int, default = 3, help = 'Type in the number of topk comparisons')
args = parser.parse_args()

h_params = { 'data_dir': args.data_dir,
                    'save_dir': args.save_dir,
                    'arch': args.arch,
                    'epochs': args.epochs,
                    'lr': args.lr,
                    'gpu': args.gpu,
                    'input_layers': args.input_layers,
                    'hidden_layers': args.hidden_layers,
                    'output_layers': args.output_layers,
                    'drop': args.drop_rate,
                    'topk':args.topk
}


########## Step 2: Get Data
train_dir = h_params['data_dir'] + '/train'
valid_dir = h_params['data_dir'] + '/valid'
test_dir = h_params['data_dir'] + '/test'

data_transforms = {
'train': transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])]),
'test': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]),
'valid': transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])}
#Directory Dictionary
directories = {'train': train_dir,
               'test': test_dir,
               'valid': valid_dir}

data_img = {x: datasets.ImageFolder(directories[x],transform = data_transforms[x]) 
                  for x in list(data_transforms.keys())}

dataloaders = {
'trainloader' : torch.utils.data.DataLoader(data_img['train'], batch_size=64, shuffle=True),
'testloader' : torch.utils.data.DataLoader(data_img['test'], batch_size=64, shuffle=False),
'validloader' : torch.utils.data.DataLoader(data_img['valid'], batch_size=64, shuffle=True)
}

#test image load
images, labels = next(iter(dataloaders['testloader']))
print('number of testloader data: '+ str(len(images[0,2])))

########## 3. Build the Network
def get_model(model_arch):
    #load a pretrained model
    if (model_arch == 'vgg16'):
        model = models.vgg16(pretrained = True)
    elif (model_arch == 'densenet121'):
        model = models.densenet121(pretrained = True)
    elif (model_arch == 'alexnet'):
        model = models.alexnet(pretrained = True)
    return model
#end get_model function

def build_model(model, model_arch, drop_out):
    for param in model.parameters():
        param.requires_grad = False

    if (model_arch == 'vgg16'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(25088, 4096)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('fc2', nn.Linear(4096, 102)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    elif (model_arch == 'densenet121'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(1024, 102)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    elif (model_arch == 'alexnet'):
        from collections import OrderedDict
        classifier = nn.Sequential(OrderedDict([
                                  ('fc1', nn.Linear(9216, 4096)),
                                  ('relu', nn.ReLU()),
                                  ('dropout', nn.Dropout(drop_out)),
                                  ('fc2', nn.Linear(4096, 102)),
                                  ('output', nn.LogSoftmax(dim=1))                             
                                  ]))
    else:
        print('you screwed up badly if the codes come here')
    return classifier

model = get_model(h_params['arch'].lower())
model_classifier = build_model(model, h_params['arch'].lower(), h_params['drop'])
model.classifier = model_classifier
print('\nprinting the selected architecture ' + h_params['arch'] + ' classifier = ')
print(model.classifier)

########## 4. Train the Network
def train_model(model, criterion, optimizer, epochs, load_train_data, load_valid_data, gpu):
    model.train()
    print_every,steps = 30,0
    st_gpu = False
    
    # change to cuda
    if (gpu == 'GPU' and torch.cuda.is_available()):
        st_gpu = True
        print('training moves to cuda')
        model.to('cuda')
    elif (gpu == 'CPU'):
        print('training moves to CPU')
        model.to('cpu')

    for e in range(epochs):
        running_loss = 0
    
        for inputs, labels in load_train_data:
            steps += 1
            inputs, labels = inputs.to('cuda'), labels.to('cuda')
            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()

                accuracy, test_loss = 0,0

                for images, labels in iter(load_valid_data):
                    images, labels = images.to('cuda'), labels.to('cuda')
                    output = model.forward(images)
                    test_loss += criterion(output, labels).item()
                    ps = torch.exp(output)
                    equality = (labels.data == ps.max(dim=1)[1])
                    accuracy += equality.type(torch.FloatTensor).mean()

                with torch.no_grad():
                    print("Epoch: {}/{}... ".format(e+1, epochs),
                          "Training Loss: {:.4f}".format(running_loss/print_every),
                          "Test Loss: {:.3f}.. ".format(test_loss/len(load_valid_data)),
                          "Test Accuracy: {:.3f}".format(accuracy/len(load_valid_data)))
                    running_loss = 0
                model.train()

def check_accuracy_on_test(model, testloader, gpu):    
    correct = 0
    total = 0
    
    model.eval()
    
    if (gpu == 'GPU'):
        model.to('cuda:0')
    
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            if (gpu == 'GPU'):
                images, labels = images.to('cuda'), labels.to('cuda')
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('\nAccuracy of the network on the test images: %d %%' % (100 * correct / total))

criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), h_params['lr'])
train_model(model, criterion, optimizer, h_params['epochs'], dataloaders['trainloader'], dataloaders['validloader'], h_params['gpu'])
check_accuracy_on_test(model, dataloaders['testloader'], h_params['gpu'])

########### 5. Save the model and weights
model.class_to_idx = data_img['train'].class_to_idx
checkpoint = {
    'arch': h_params['arch'],
    'class_to_idx': model.class_to_idx, 
    'state_dict': model.state_dict(),
    'optimizer': optimizer.state_dict(),
    'input_layers': h_params['input_layers'],
    'hidden_layers': h_params['hidden_layers'],
    'output_layers': h_params['output_layers'],
    'learning rate': h_params['lr'],
    'dropout': h_params['drop'],
    'epochs': h_params['epochs'],
    'topk': h_params['topk']
}
torch.save(checkpoint, 'checkpoint.pth')