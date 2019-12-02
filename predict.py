import argparse
import torch
from torchvision import datasets, transforms, models
import numpy as np
from PIL import Image
import json
import myhelper

class Predict:
    mean = [0.485, 0.456, 0.406]
    sd = [0.229, 0.224, 0.225]
    
    def __init__(self):
        print("Welcome to the Predictor")
        self.in_arg = self.get_input_args()
        print("Will run predictor using: Checkpoint: "+str(self.in_arg.checkpoint)+", Image: "+str(self.in_arg.image) +", Top_K: "+str(self.in_arg.top_k))
        with open(self.in_arg.category_names, 'r') as f:
            self.cat_to_name = json.load(f)
        #Set device
        self.device = myhelper.device(self.in_arg.gpu)
        if self.device is None:
            print('You selected GPU, but GPU is not available!, closing')
            return
        print("Using Device: "+str(self.device))
        print("Loading checkpoint")
        self.checkpoint = self.loadCheckPoint()
        print("Predicting")
        probs, classes = self.predict(self.in_arg.image, self.model, self.in_arg.top_k)
        #categories = [ self.in_arg.cat_to_name[i] for i in classes ]
        for key,value in enumerate(probs):
            print("Class: "+classes[key]+", Flower: "+str(self.cat_to_name[classes[key]])+", Probability: "+str(value))

    def predict(self, image_path, model, topk):
        self.model.eval()
        image = self.process_image(Image.open(image_path))
        image = image.unsqueeze(0)
        image = image.float()    
        image = image.to(self.device)
        print(image.shape)
        output = model.forward(image)
        ps = torch.exp(output)
        
        probs, indices = torch.topk(ps, topk)
        
        probs = probs.cpu()
        indices = indices.cpu()
        if topk > 1:
            probs = probs.squeeze()
            indices = indices.squeeze()
        else:    
            probs = probs.squeeze(1)
            indices = indices.squeeze(1)

        indices = indices.numpy()
        probs = probs.detach().numpy()
        
        # invert class_to_idx dict
        idx_to_class = {i:c for c,i in self.checkpoint['class_to_idx'].items() }
        classes = [ idx_to_class[i] for i in indices ]
        return probs, classes

    def process_image(self,image):
        image.thumbnail([256,256],Image.ANTIALIAS)
        width, height = image.size
        #print(image.size)
        leading = (width - 224)/2
        if leading <= 0:
            leading = 0
            trailing = 224
        else:
            trailing = leading + 224
            
        top = (height - 224)/2
        if top <= 0:
            top = 0
            bottom = 224
        else:
            bottom = top + 224
        
        #print("Leading: "+str(leading)+" Top: "+str(top)+" Trailing: "+str(trailing)+" Bottom: "+str(bottom))
        image = image.crop(box=(leading,top,trailing,bottom))
        np_image = np.array(image)/255
        
        np_mean = np.array(self.mean)
        np_sd = np.array(self.sd)
        
        np_image = ((np_image-np_mean)/np_sd)
        np_image = np_image.transpose((2,0,1))
        return torch.from_numpy(np_image)

    def loadCheckPoint(self):
        checkpoint = torch.load(self.in_arg.checkpoint, map_location=lambda storage, loc: storage)
        #checkpoint = torch.load(name+'.pth')
        classifier = myhelper.Network(checkpoint['input_size'], checkpoint['output_size'], checkpoint['hidden_layers'], drop_p=checkpoint['drop'])
        classifier.load_state_dict(checkpoint['state_dict'])
        
        print("Loading model "+str(checkpoint['model']))
        if checkpoint['model'] == 'vgg16':
            self.model = models.vgg16(pretrained = True)
        elif checkpoint['model'] == 'densenet161':
            self.model = models.densenet161(pretrained = True)
            
        for params in self.model.parameters():
            params.requires_grad = False

        self.model.classifier = classifier
        self.model.to(self.device)

        return {'input_size': checkpoint['input_size'],
                'output_size': checkpoint['output_size'],
                'batch_size': checkpoint['batch_size'],
                'epochs': checkpoint['epochs'],
                'drop': checkpoint['drop'],
                'model': checkpoint['model'],
                'learning_rate': checkpoint['learning_rate'],
                'class_to_idx': checkpoint['class_to_idx'],
                'hidden_layers': checkpoint['hidden_layers']}

    def get_input_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint',type=str, help='<Required>Set path to checkpoint', required=True)
        parser.add_argument('--image',type=str, help='<Required> Set path to image', required=True)
        parser.add_argument('--top_k',type=int,default=1, help='Set the number of top predictions wanted, default 1')
        parser.add_argument('--category_names',type=str,default='cat_to_name.json', help='Path to category names file, default cat_to_name.json')
        parser.add_argument('--gpu',type=str, default="no", help='Set yes to use GPU, default CPU')
        return parser.parse_args() 

Predict()