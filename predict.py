import argparse
import matplotlib.pyplot as plt
import seaborn as sb
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
from PIL import img

# Get Model h_params
parser = argparse.ArgumentParser(description = 'Pass in the Train and Predict Parameters')
parser.add_argument('--img_path_and_name', type = str, default = './flowers/test/1/image_06764.jpg', help = 'The location of the img file and name')
parser.add_argument('--checkpoint_path_and_name', type = str, default = './checkpoint.pth', help = 'The location of the checkpoint file and name')
parser.add_argument('--category_file_path_and_name', type = str, default = './cat_n.json', help = 'The location of the category mapping file and name')
parser.add_argument('--topk', type = int, default = 3, help = 'Type in the number of topk comparisons')
parser.add_argument('--gpu', type = str,  default = 'GPU', choices=['GPU','CPU'], help = 'Type GPU or CPU with uppercase')
args = parser.parse_args()

h_params = { 'image_dir': args.img_path_and_name,
                    'checkpoint_dir': args.checkpoint_path_and_name,
                    'category_dir': args.category_file_path_and_name,
                    'topk': args.topk,
                    'gpu': args.gpu
}

#Label mapping
cat_file_name = h_params['category_dir']
with open(cat_file_name, 'r') as f:
    cat_n = json.load(f)
print('number of cat_n:' + str(len(cat_n)))

########### 6. load the model and weights

def load_model(checkpoint):

    output_layers = checkpoint['output_layers']
    print('\noutput_layers = ' + str(output_layers))
    hidden_layers = checkpoint['hidden_layers']
    input_layers = checkpoint['input_layers']
    
    if checkpoint['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif (model_arch == 'densenet121'):
        model = models.densenet121(pretrained = True)
    elif (model_arch == 'alexnet'):
        model = models.alexnet(pretrained = True)
        
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    model.class_to_idx = checkpoint['class_to_idx']
    
    # Create the classifier
    from collections import OrderedDict
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_layers, hidden_layers)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(hidden_layers, output_layers)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
    # Put the classifier on the pretrained network
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
#end of load_model function

checkpoint = torch.load(h_params['checkpoint_dir'])
model = load_model(checkpoint)

########### 7. Process img
def process_images(image_path):
    ''' Scales, crops, and normalizes a PIL img for a PyTorch model,
        returns an Numpy array
    '''
    #editing img
    image_transformer = transforms.Compose([  
                                       transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor()
                                       ])
                                       
    pre_img = img.open(image_path)
    pre_img = image_transformer(pre_img)
    np_image = np.array(pre_img)
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np.transpose(np_image, (1, 2, 0)) - mean)/std    
    np_image = np.transpose(np_image, (2, 0, 1))
    
    return np_image
#end process_images

image_path = h_params['image_dir']
img = process_images(image_path)

########### 8. Do a class prediction
def predict(image_path, model, topk_number, cat_n, gpu):
    ''' Predict the class (or classes) of an img using a trained deep learning model.
    '''
    model.eval()
    img = process_images(image_path)
    tensor_image = torch.from_numpy(img).type(torch.FloatTensor)
    
    gpu_mode = False
    if (gpu == 'GPU' and torch.cuda.is_available()):
        gpu_mode = True 
        print('training moves to cuda')
        model.to('cuda')
        tensor_image = tensor_image.cuda()    
       
    elif (gpu == 'CPU'):
        print('training moves to CPU')
        model.to('cpu')
        
    # add 1 to tensor img
    tensor_image = tensor_image.unsqueeze_(0)
        
    #Probability
    probability = torch.exp(model.forward(tensor_image))
    
    #pick top probabilities
    probs, classes = probability.topk(topk_number)
    
    if gpu_mode:
        probs = probs.cpu().detach().numpy().tolist()[0]
        classes = classes.cpu().detach().numpy().tolist()[0]
    else:
        probs = probs.detach().numpy().tolist()[0]
        classes = classes.detach().numpy().tolist()[0]
    
    # Convert indices to classes
    i_class = {val: key for key, val in model.class_to_idx.items()}
    
    top_labels = [i_class[classes] for classes in classes]
    top_flowers = [cat_n[i_class[classes]] for classes in classes] 
    return probs, top_flowers

probs, top_flowers = predict(image_path, model, h_params['topk'], cat_n, h_params['gpu'] )
print("\nTop flowers prediction:")
print(top_flowers)
print('\ntheir probabilities:')
print(probs)