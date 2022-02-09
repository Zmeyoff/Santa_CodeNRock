import torch
from torch import nn
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

from sklearn.model_selection import train_test_split

import time 
from tqdm import tqdm

def set_requires_grad(model, value=False):
    for param in model.parameters():
        param.requires_grad = value

def train_model(model, dataloaders, criterion, optimizer,
                phases, num_epochs=3):
    start_time = time.time()

    acc_history = {k: list() for k in phases}
    loss_history = {k: list() for k in phases}

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = len(dataloaders[phase])
            for inputs, labels in tqdm(dataloaders[phase], total=n_batches):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double()
            epoch_acc /= len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss,
                                                       epoch_acc))
            loss_history[phase].append(epoch_loss)
            acc_history[phase].append(epoch_acc)

        print()

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60,
                                                        time_elapsed % 60))

    return model, acc_history

def init_model_ef4(device):
    model = torchvision.models.efficientnet_b4()    
    set_requires_grad(model, False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(device)
    return model

def init_model_ef7(device):
    model = torchvision.models.efficientnet_b7()    
    set_requires_grad(model, False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 3)
    model = model.to(device)
    return model

def init_model_regy8(device):
    model = torchvision.models.regnet_y_8gf()
    set_requires_grad(model, False)
    model.fc = torch.nn.Linear(model.fc.in_features, 3)
    model = model.to(device)
    return model

class DedyDataset(Dataset):
    def __init__(self, root_dir, csv_path=None, transform=None):
        
        self.transform = transform
        self.files = [os.path.join(root_dir, fname) for fname in os.listdir(root_dir)]
        self.targets = None
        if csv_path:
            df = pd.read_csv(csv_path, sep="\t")
            self.targets = df["class_id"].tolist()
            self.files = [os.path.join(root_dir, fname) for fname in df["image_name"].tolist()]

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        target = self.targets[idx] if self.targets else -1
        if self.transform:
            image = self.transform(image)
        return image, target

# hardcode
MODEL_WEIGHTS = "./data/weight/baseline.pt"
MODEL_WEIGHTS_EF4 = "./data/weight/baseline_ef4_last.pt"
MODEL_WEIGHTS_EF7 = "./data/weight/baseline_ef7_last.pt"
MODEL_WEIGHTS_RY8 = "./data/weight/baseline_ry8_last.pt"
TRAIN_DATASET = "./data/train/"
TEST_DATASET = "./data/test/"
TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/out/submission.csv"
if __name__ == "__main__":
    img_size = 380
    # make slight augmentation and normalization on ImageNet statistics
    
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        #transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    batch_size = 1
    num_workers = 1
    dset = DedyDataset(TEST_DATASET, transform=trans)
    #testset = torch.utils.data.Subset(dset, ind_test)
    
    testloader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                            shuffle=False, num_workers=num_workers)
    
    #iloaders = {'train': trainloader, 'val': testloader}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model4 = init_model_ef4(device)
    model7 = init_model_ef7(device)
    model8 = init_model_regy8(device)
    model4.load_state_dict(torch.load(MODEL_WEIGHTS_EF4))
    model7.load_state_dict(torch.load(MODEL_WEIGHTS_EF7))
    model8.load_state_dict(torch.load(MODEL_WEIGHTS_RY8))
    model4.eval()    
    model7.eval()    
    model8.eval()    
    n_batches = len(testloader)
    #for inputs, labels in tqdm(testloader, total=n_batches):
    arr = []
    mu = 0.8
    for i , (inputs, labels) in enumerate(testloader, 0):
        inputs = inputs.to(device)
        labels = labels.to(device)
        model4_out = model4(inputs)
        model7_out = model7(inputs)
        model8_out = model8(inputs)
        split_out = 0.4 * model4_out + 0.4 * model8_out + 0.2 * model7_out
        _, predicted = torch.max(split_out, 1)
        #_, predicted = torch.max(model(inputs), 1)
        sample_fname = dset.files[i]
        #print(predicted.cpu().)
        #print(os.path.basename(sample_fname), predicted.item())
        arr.append([os.path.basename(sample_fname), predicted.item()])

    #print(arr)

    df = pd.DataFrame(arr, columns=['image_name', 'class_id'])
    #print(df.head())
    df.to_csv('./data/out/submission.csv', index=False, sep='\t')
