
import numpy as np
import torch
from torchvision import datasets, transforms, models
from torch import nn
from torch import optim
import argparse
import os
import myhelper


class Train:
    
    def __init__(self):
        print("Welcome to the Trainer")
        #Set input arguments
        self.in_arg = self.get_input_args()
        print("Will run trainer using: Path: "+str(self.in_arg.dir)+", Learning Rate: "+str(self.in_arg.learning_rate)+", Hidden Units: "+str(self.in_arg.hidden_units)+", Epochs: "+str(self.in_arg.epochs))
        #Checks
        if os.path.isdir(self.in_arg.dir) == False:
            print("ERROR Cannot find directory for data at: "+self.in_arg.dir)
            return
        if self.in_arg.save_dir != "" and os.path.isdir(self.in_arg.save_dir) == False:
            print("ERROR Cannot find directory to save checkpoint at: "+self.in_arg.save_dir)
            return
        #Set directories
        self.train_dir = self.in_arg.dir + '/train'
        self.valid_dir = self.in_arg.dir + '/valid'
        self.test_dir = self.in_arg.dir + '/test'
        #Set device
        print(self.in_arg.gpu)
        self.device = myhelper.device(self.in_arg.gpu)
        if self.device is None:
            print('You selected GPU, but GPU is not available!, closing')
            return
        print("Using Device: "+str(self.device))
        #Set data
        self.setupData()
        #Load model
        print("Loading model "+self.in_arg.arch)
        if self.in_arg.arch == 'vgg16':
            self.model = models.vgg16(pretrained = True)
        elif self.in_arg.arch == 'densenet161':
            self.model = models.densenet161(pretrained = True)
        for params in self.model.parameters():
            params.requires_grad = False
        #Setup classifier
        print("Setting up classifier")
        self.classifier = myhelper.Network(self.input_size, self.output_size, self.hidden_layers, drop_p=self.drop)
        self.classifier.to(self.device)
        self.model.classifier = self.classifier
        self.model.class_to_idx = self.image_datasets['train'].class_to_idx
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.classifier.parameters(), lr= self.learning_rate)
        self.criterion = nn.NLLLoss()
        #Run training, and make checkpoint
        self.runTraining(self.makeCheckpoint)


    def runTraining(self, completion):
        print("Started training, will print validation every 40")
        steps = 0
        running_loss = 0
        print_every = 40
        for e in range(self.epochs):
            for images, labels in self.loaders['train']:
                self.model.train()
                images, labels = images.to(self.device), labels.to(self.device)
                steps += 1
                self.optimizer.zero_grad()
                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                
                if steps % print_every == 0:
                    self.model.eval()
                    with torch.no_grad():
                        test_loss, accuracy = self.validation(self.loaders['valid'], self.model)
                    print("Epoch: {}/{}.. ".format(e+1, self.epochs),
                        "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                        "Validation Loss: {:.3f}.. ".format(test_loss/len(self.loaders['valid'])),
                        "Validation Accuracy: {:.3f}".format(accuracy/len(self.loaders['valid'])))
                    running_loss = 0
                    self.model.train()
            if e == (self.epochs - 1):
                self.model.eval()
                with torch.no_grad():
                    test_loss, accuracy = self.validation(self.loaders['valid'], self.model)
                print("Final",
                    "Validation Loss: {:.3f}.. ".format(test_loss/len(self.loaders['valid'])),
                    "Validation Accuracy: {:.3f}".format(accuracy/len(self.loaders['valid'])))
                print("Training complete")
                completion()

    def setupData(self):
        mean = [0.485, 0.456, 0.406]
        sd = [0.229, 0.224, 0.225]

        if self.in_arg.arch == 'vgg16' :
            self.input_size = 25088
        elif self.in_arg.arch == 'densenet161':
            self.input_size = 2208
            
        self.output_size = 102
        self.hidden_layers = self.in_arg.hidden_units
        self.drop = 0.5
        self.epochs = self.in_arg.epochs
        self.learning_rate = self.in_arg.learning_rate

        #Transforms for the training, validation, and testing sets
        test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,sd)
                                     ])
        train_transforms = transforms.Compose([
                                      transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomRotation(30),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean,sd)
                                     ])

        #Load Datasets with ImageFolder
        test_dataset = datasets.ImageFolder(self.test_dir, transform = test_transforms)
        validation_dataset = datasets.ImageFolder(self.valid_dir, transform = test_transforms)
        train_dataset = datasets.ImageFolder(self.train_dir, transform = train_transforms)
        self.image_datasets = {'test':test_dataset, 'valid':validation_dataset, 'train':train_dataset}

        #Dataloaders
        testloader = torch.utils.data.DataLoader(self.image_datasets['test'], batch_size=32,shuffle=True)
        validloader = torch.utils.data.DataLoader(self.image_datasets['valid'], batch_size=32,shuffle=True)
        trainloader = torch.utils.data.DataLoader(self.image_datasets['train'], batch_size=64,shuffle=True)
        self.loaders = {'test':testloader, 'valid':validloader, 'train': trainloader}



    def validation(self, loader, model):
        test_loss = 0
        accuracy = 0
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            output = model.forward(images)
            test_loss += self.criterion(output, labels).item()
            ps = torch.exp(output)
            equality = (labels.data == ps.max(dim=1)[1])
            accuracy += equality.type(torch.FloatTensor).mean()
        return test_loss, accuracy
        
    def makeCheckpoint(self):
        print("Making checkpoint")
        checkpoint = {'input_size': self.input_size,
              'output_size': self.output_size,
              'batch_size':64,
              'epochs': self.epochs,
              #'optimizer': trainer.optimizer.state_dict,
              'drop': self.drop,
              'learning_rate': self.learning_rate,
              'class_to_idx': self.model.class_to_idx,
              'hidden_layers': [each.out_features for each in self.model.classifier.hidden_layers],
              'model': self.in_arg.arch,
              'state_dict': self.model.classifier.state_dict()}
        torch.save(checkpoint, self.in_arg.save_dir+'checkpoint.pth')
        print("All done")

    def get_input_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_units','--list', type=int, nargs='+',default=[12544,1568], help='Set hidden units, seperated by space, default 12544,1568 for vgg16')
        parser.add_argument('--dir',type=str,default='flowers', help='Path to images folder')
        parser.add_argument('--save_dir',type=str,default='', help='Set directory to save checkpoints eg. "yourdirectory/", will save in this directory file named checkpoint.pht')
        parser.add_argument('--arch',type=str,default='vgg16',help='Set pre trained model vgg16 or densenet161, default vgg16')
        parser.add_argument('--learning_rate',type=float,default='0.001',help='Set learning rate, default 0.001')
        parser.add_argument('--epochs',type=int,default='5',help='Set epochs, default 5')
        parser.add_argument('--gpu',type=str, default="no", help='Set yes to use GPU, default CPU')
        return parser.parse_args() 

Train()