
# coding: utf-8

# ### Developing an AI application
# 
# Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 
# 
# In this project, you'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories, you can see a few examples below. 
# 
# <img src='assets/Flowers.png' width=500px>
# 
# The project is broken down into multiple steps:
# 
# * Load and preprocess the image dataset
# * Train the image classifier on your dataset
# * Use the trained classifier to predict image content
# 
# We'll lead you through each part which you'll implement in Python.
# 
# When you've completed this project, you'll have an application that can be trained on any set of labeled images. Here your network will be learning about flowers and end up as a command line application. But, what you do with your new skills depends on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.
# 
# First up is importing the packages you'll need. It's good practice to keep all the imports at the beginning of your code. As you work through this notebook and find you need to import a package, make sure to add the import up here.

# In[1]:


# Imports here
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")

import matplotlib.pyplot as plt
import seaborn as sb

import torch
from torchvision import datasets, transforms, models

from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict
from PIL import Image

from torch.autograd import Variable


# ## Load the data
# 
# Here you'll use `torchvision` to load the data ([documentation](http://pytorch.org/docs/0.3.0/torchvision/index.html)). The data should be included alongside this notebook, otherwise you can [download it here](https://s3.amazonaws.com/content.udacity-data.com/nd089/flower_data.tar.gz). The dataset is split into three parts, training, validation, and testing. For the training, you'll want to apply transformations such as random scaling, cropping, and flipping. This will help the network generalize leading to better performance. You'll also need to make sure the input data is resized to 224x224 pixels as required by the pre-trained networks.
# 
# The validation and testing sets are used to measure the model's performance on data it hasn't seen yet. For this you don't want any scaling or rotation transformations, but you'll need to resize then crop the images to the appropriate size.
# 
# The pre-trained networks you'll use were trained on the ImageNet dataset where each color channel was normalized separately. For all three sets you'll need to normalize the means and standard deviations of the images to what the network expects. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`, calculated from the ImageNet images.  These values will shift each color channel to be centered at 0 and range from -1 to 1.
#  

# In[2]:


data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'


# In[3]:


batch_size = 64
size = (224, 224)
mean = [0.485, 0.456, 0.406]
sd = [0.229, 0.224, 0.225]

input_size = 25088
output_size = 102
hidden_layers = [12544, 1568]
drop = 0.5
epochs = 3
learning_rate = 0.001

#cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: "+str(device))
# TODO: Define your transforms for the training, validation, and testing sets
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,sd)
                                     ])
train_transforms = transforms.Compose([
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(90),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,sd)
                                     ])

# TODO: Load the datasets with ImageFolder
test_dataset = datasets.ImageFolder(test_dir, transform = test_transforms)
validation_dataset = datasets.ImageFolder(valid_dir, transform = test_transforms)
train_dataset = datasets.ImageFolder(train_dir, transform = train_transforms)
image_datasets = {'test':test_dataset, 'valid':validation_dataset, 'train':train_dataset}

# TODO: Using the image datasets and the trainforms, define the dataloaders
testloader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=batch_size,shuffle=True)
validloader = torch.utils.data.DataLoader(image_datasets['valid'], batch_size=batch_size,shuffle=True)
trainloader = torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,shuffle=True)
loaders = {'test':testloader, 'valid':validloader, 'train': trainloader}


# ### Label mapping
# 
# You'll also need to load in a mapping from category label to category name. You can find this in the file `cat_to_name.json`. It's a JSON object which you can read in with the [`json` module](https://docs.python.org/2/library/json.html). This will give you a dictionary mapping the integer encoded categories to the actual names of the flowers.

# In[4]:


import json
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
#print(cat_to_name)


# # Building and training the classifier
# 
# Now that the data is ready, it's time to build and train the classifier. As usual, you should use one of the pretrained models from `torchvision.models` to get the image features. Build and train a new feed-forward classifier using those features.
# 
# We're going to leave this part up to you. If you want to talk through it with someone, chat with your fellow students! You can also ask questions on the forums or join the instructors in office hours.
# 
# Refer to [the rubric](https://review.udacity.com/#!/rubrics/1663/view) for guidance on successfully completing this section. Things you'll need to do:
# 
# * Load a [pre-trained network](http://pytorch.org/docs/master/torchvision/models.html) (If you need a starting point, the VGG networks work great and are straightforward to use)
# * Define a new, untrained feed-forward network as a classifier, using ReLU activations and dropout
# * Train the classifier layers using backpropagation using the pre-trained network to get the features
# * Track the loss and accuracy on the validation set to determine the best hyperparameters
# 
# We've left a cell open for you below, but use as many as you need. Our advice is to break the problem up into smaller parts you can run separately. Check that each part is doing what you expect, then move on to the next. You'll likely find that as you work through each part, you'll need to go back and modify your previous code. This is totally normal!
# 
# When training make sure you're updating only the weights of the feed-forward network. You should be able to get the validation accuracy above 70% if you build everything right. Make sure to try different hyperparameters (learning rate, units in the classifier, epochs, etc) to find the best model. Save those hyperparameters to use as default values in the next part of the project.

# In[5]:


class Network(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_p=drop):
        super().__init__()
        # Add the first layer, input to a hidden layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # Add a variable number of more hidden layers
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h1, h2) for h1, h2 in layer_sizes])
        
        self.output = nn.Linear(hidden_layers[-1], output_size)
        self.dropout = nn.Dropout(p=drop_p)
        
    def forward(self, x):
        for linear in self.hidden_layers:
            x = F.relu(linear(x))
            x = self.dropout(x)
        x = self.output(x)    
        return F.log_softmax(x, dim=1) 


# In[6]:


# TODO: Build and train your network
#Pre trained network
model = models.vgg16(pretrained = True)
#model = models.vgg19(pretrained = True)
#model = models.densenet161(pretrained = True)
#Freeze params
for params in model.parameters():
    params.requires_grad = False
criterion = nn.NLLLoss()


# In[7]:


def validation(loader, model):
    test_loss = 0
    accuracy = 0
    for images, labels in loader:
        images, labels = Variable(images.to(device)), Variable(labels.to(device))
        output = model.forward(images)
        test_loss += criterion(output, labels).item()
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy


# In[8]:


classifier = Network(input_size, output_size, hidden_layers, drop_p=drop)
classifier.to(device)
model.classifier = classifier
model.to(device)

optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
steps = 0
running_loss = 0
print_every = 40
for e in range(epochs):
    for images, labels in loaders['train']:
        model.train()
        images, labels = Variable(images.to(device)), Variable(labels.to(device))
        steps += 1
        optimizer.zero_grad()
        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            # Make sure network is in eval mode for inference
            model.eval()
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validation(loaders['valid'], model)
            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(loaders['valid'])),
                  "Test Accuracy: {:.3f}".format(accuracy/len(loaders['valid'])))
            running_loss = 0
            model.train()


# ## Testing your network
# 
# It's good practice to test your trained network on test data, images the network has never seen either in training or validation. This will give you a good estimate for the model's performance on completely new images. Run the test images through the network and measure the accuracy, the same way you did validation. You should be able to reach around 70% accuracy on the test set if the model has been trained well.

# In[ ]:


# TODO: Do validation on the test set
model.eval()
with torch.no_grad():
    test_loss, accuracy = validation(loaders['valid'], model)
print("Test Loss: {:.3f}.. ".format(test_loss/len(loaders['valid'])),
      "Test Accuracy: {:.3f}".format(accuracy/len(loaders['valid']))) 


# In[9]:


def check_accuracy_on_test(loader): 
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = Variable(images.to(device)), Variable(labels.to(device))
            outputs = model.forward(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
    
check_accuracy_on_test(loaders['test'])


# ## Save the checkpoint
# 
# Now that your network is trained, save the model so you can load it later for making predictions. You probably want to save other things such as the mapping of classes to indices which you get from one of the image datasets: `image_datasets['train'].v`. You can attach this to the model as an attribute which makes inference easier later on.
# 
# ```model.class_to_idx = image_datasets['train'].class_to_idx```
# 
# Remember that you'll want to completely rebuild the model later so you can use it for inference. Make sure to include any information you need in the checkpoint. If you want to load the model and keep training, you'll want to save the number of epochs as well as the optimizer state, `optimizer.state_dict`. You'll likely want to use this trained model in the next part of the project, so best to save it now.

# In[ ]:


#print(train.model)
print(image_datasets['train'].class_to_idx)


# In[ ]:


# TODO: Save the checkpoint 
def makeCheckPoint(trainer, name):
    checkpoint = {'input_size': input_size,
              'output_size': output_size,
              'batch_size':batch_size,
              'epochs': epochs,
              'optimizer': trainer.optimizer.state_dict,
              'drop': drop,
              'learning_rate': learning_rate,
              'class_to_idx': image_datasets['train'].class_to_idx,
              'hidden_layers': [each.out_features for each in trainer.model.classifier.hidden_layers],
              'state_dict': trainer.model.state_dict()}
    torch.save(checkpoint, name+'.pth')
    
makeCheckPoint(trainer, 'checkpoint2')


# ## Loading the checkpoint
# 
# At this point it's good to write a function that can load a checkpoint and rebuild the model. That way you can come back to this project and keep working on it without having to retrain the network.

# In[ ]:


# TODO: Write a function that loads a checkpoint and rebuilds the model
def loadCheckPoint(name):
    checkpoint = torch.load(name+'.pth', map_location=lambda storage, loc: storage)
    #checkpoint = torch.load(name+'.pth')
    classifier = Network(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], drop_p=checkpoint['drop'])
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    return {'input_size': checkpoint['input_size'],
            'output_size': checkpoint['output_size'],
            'batch_size': checkpoint['batch_size'],
            'epochs': checkpoint['epochs'],
            'optimizer': checkpoint['optimizer'],
            'drop': checkpoint['drop'],
            'learning_rate': checkpoint['learning_rate'],
            'class_to_idx': checkpoint['class_to_idx'],
            'hidden_layers': checkpoint['hidden_layers']}
    
checkpoint = loadCheckPoint('checkpoint2')


# # Inference for classification
# 
# Now you'll write a function to use a trained network for inference. That is, you'll pass an image into the network and predict the class of the flower in the image. Write a function called `predict` that takes an image and a model, then returns the top $K$ most likely classes along with the probabilities. It should look like 
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```
# 
# First you'll need to handle processing the input image such that it can be used in your network. 
# 
# ## Image Preprocessing
# 
# You'll want to use `PIL` to load the image ([documentation](https://pillow.readthedocs.io/en/latest/reference/Image.html)). It's best to write a function that preprocesses the image so it can be used as input for the model. This function should process the images in the same manner used for training. 
# 
# First, resize the images where the shortest side is 256 pixels, keeping the aspect ratio. This can be done with the [`thumbnail`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) or [`resize`](http://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail) methods. Then you'll need to crop out the center 224x224 portion of the image.
# 
# Color channels of images are typically encoded as integers 0-255, but the model expected floats 0-1. You'll need to convert the values. It's easiest with a Numpy array, which you can get from a PIL image like so `np_image = np.array(pil_image)`.
# 
# As before, the network expects the images to be normalized in a specific way. For the means, it's `[0.485, 0.456, 0.406]` and for the standard deviations `[0.229, 0.224, 0.225]`. You'll want to subtract the means from each color channel, then divide by the standard deviation. 
# 
# And finally, PyTorch expects the color channel to be the first dimension but it's the third dimension in the PIL image and Numpy array. You can reorder dimensions using [`ndarray.transpose`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.ndarray.transpose.html). The color channel needs to be first and retain the order of the other two dimensions.

# In[ ]:


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    image.thumbnail([256,256],Image.ANTIALIAS)
    width, height = image.size
    print(image.size)
    leading = (width - 224)/2
    if leading <= 0:
        leading = 0
        trailing = height
    else:
        trailing = leading + 224
        
    top = (height - 224)/2
    if top <= 0:
        top = 0
        bottom = 224
    else:
        bottom = top + 224
    
    print("Leading: "+str(leading)+" Top: "+str(top)+" Trailing: "+str(trailing)+" Bottom: "+str(bottom))
    image = image.crop(box=(leading,top,trailing,bottom))
    np_image = np.array(image)/255
    
    np_mean = np.array(mean)
    np_sd = np.array(sd)
    
    np_image = ((np_image-np_mean)/np_sd)
    np_image = np_image.transpose((2,0,1))
    return torch.from_numpy(np_image)


# To check your work, the function below converts a PyTorch tensor and displays it in the notebook. If your `process_image` function works, running the output through this function should return the original image (except for the cropped out portions).

# In[ ]:


def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax


# ## Class Prediction
# 
# Once you can get images in the correct format, it's time to write a function for making predictions with your model. A common practice is to predict the top 5 or so (usually called top-$K$) most probable classes. You'll want to calculate the class probabilities then find the $K$ largest values.
# 
# To get the top $K$ largest values in a tensor use [`x.topk(k)`](http://pytorch.org/docs/master/torch.html#torch.topk). This method returns both the highest `k` probabilities and the indices of those probabilities corresponding to the classes. You need to convert from these indices to the actual class labels using `class_to_idx` which hopefully you added to the model or from an `ImageFolder` you used to load the data ([see here](#Save-the-checkpoint)). Make sure to invert the dictionary so you get a mapping from index to class as well.
# 
# Again, this method should take a path to an image and a model checkpoint, then return the probabilities and classes.
# 
# ```python
# probs, classes = predict(image_path, model)
# print(probs)
# print(classes)
# > [ 0.01558163  0.01541934  0.01452626  0.01443549  0.01407339]
# > ['70', '3', '45', '62', '55']
# ```

# In[ ]:


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    model.eval()
    
    image = process_image(Image.open(image_path))
    
    image = image.unsqueeze(0)
    image = image.float()    
    output = model.forward(Variable(image))
    ps = torch.exp(output)
    
    probs, indices = torch.topk(ps, 5)
    
    probs = probs.cpu()
    indices = indices.cpu()
    probs = probs.squeeze()
    indices = indices.squeeze()
    indices = indices.numpy()
    probs = probs.detach().numpy()
    
    # invert class_to_idx dict
    idx_to_class = {i:c for c,i in image_datasets['test'].class_to_idx.items() }
    classes = [ idx_to_class[i] for i in indices ]
    return probs, classes

# TODO: Implement the code to predict the class from an image file


# ## Sanity Checking
# 
# Now that you can use a trained model for predictions, check to make sure it makes sense. Even if the testing accuracy is high, it's always good to check that there aren't obvious bugs. Use `matplotlib` to plot the probabilities for the top 5 classes as a bar graph, along with the input image. It should look like this:
# 
# <img src='assets/inference_example.png' width=300px>
# 
# You can convert from the class integer encoding to actual flower names with the `cat_to_name.json` file (should have been loaded earlier in the notebook). To show a PyTorch tensor as an image, use the `imshow` function defined above.

# In[ ]:


# TODO: Display an image along with the top 5 classes
testPath = "flowers/test/38/image_05806.jpg"
probs, classes = predict(testPath, model)
categories = [ cat_to_name[i] for i in classes ]

def imshowc(image_path, ps, classes, title=None):
    """Imshow for Tensor."""
    image = process_image(Image.open(image_path))
    
    fig, (ax1, ax2) = plt.subplots(figsize=(10,7), ncols=2)

    image = image.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax1.axis('off')
    ax1.imshow(image)
    
    ax2.barh(classes, ps)
    ax2.set_aspect(0.1)
    ax2.set_yticks(classes)
    ax2.set_yticklabels(classes)
    ax2.set_xlim(0, 1.1)
    
    plt.tight_layout()

print(categories)
print(classes)
print(probs)
imshowc(testPath,probs,categories)

