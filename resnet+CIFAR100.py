import torch 
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os 
import matplotlib.pyplot as plt 
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"  # Arrange GPU devices starting from 0
os.environ["CUDA_VISIBLE_DEVICES"]= "0"  

class InConv(nn.Module):
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
    def forward(self, x):
        return self.conv(x)
    

class BottleNeck(nn.Module):
    def __init__(self, in_channels=64, mid_channels = 64, out_channels=256, downsample=None, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=mid_channels, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        
        self.conv2 = nn.Conv2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=3, padding=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(in_channels=mid_channels, out_channels=out_channels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        #print('Input Size : {}'.format(x.size()))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        #print(out.size())
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(identity) #skip connection channel from 256 -> 512
        #print(identity.size(), out.size())
        out +=identity
        out = self.relu(out)
        return out

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1)) 
        self.fc = nn.Linear(in_channels, out_features=num_classes) 
    def forward(self,x):
        x = self.gap(x) # input = batchsize, 2048, H, W # output = batchsize, 2048, 1, 1
        x = x.view(x.size(0), -1) # flatten #output = batchsize, 2048
        x = self.fc(x) # output = batchsize, num_classes
        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.inconv = InConv(3, 64)
        self.blockgroup1 = self.make_group(3, 64, 256, 1)
        self.blockgroup2 = self.make_group(4, 256, 512, 2)
        self.blockgroup3 = self.make_group(6, 512, 1024, 2)
        self.blockgroup4 = self.make_group(3, 1024, 2048, 2)
        self.outconv = OutConv(2048, num_classes)
    def make_group(self, num_of_bottleneck, in_channels, out_channels, stride):
        #block_downsample = nn.Conv2d(in_channels*2, out_channels, 1, st)
        downsample = None
        if stride!=1 or in_channels != out_channels:    
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(BottleNeck(in_channels, out_channels//4, out_channels, downsample=downsample, stride=stride)) # first bottleneck block with downsample+shortcut
        
        for _ in range(1, num_of_bottleneck):
            layers.append(BottleNeck(out_channels, out_channels//4, out_channels))
        
        return nn.Sequential(*layers)
    def forward(self, x):
        out = self.inconv(x)
        out = self.blockgroup1(out)
        #print(out.size())
        out = self.blockgroup2(out)
        #print(out.size())
        out = self.blockgroup3(out)
        #print(out.size())
        out = self.blockgroup4(out)
        out = self.outconv(out)
        return out
    
transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

data_dir = "./datasets"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

save_dir = "./checkpoints"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
train_data = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=transforms)
val_data = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=transforms)

train_loader =DataLoader(train_data, batch_size=128, shuffle=True)
val_loader =DataLoader(val_data, batch_size=128, shuffle=True)

print('Dataset Load Completed')

def process_epoch(model, criterion, loader, optimizer=None, trainmode=True):
    if trainmode:
        model.train()
    else:
        model.eval()
    
    closs = 0
    correct = 0
    total = 0
    with tqdm(loader, unit='batch') as tepoch:
        for images, labels in tepoch:
            if torch.cuda.is_available():
                #print("Moving to CUDA")
                images = images.cuda()
                labels = labels.cuda()
            if trainmode:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
            else:
                with torch.no_grad():
                    outputs = model(images)
                    loss = criterion(outputs, labels) # average loss from all the samples
        _, predicted = torch.max(outputs.data, 1) # value, index of second dimension(prob. of classes)
        
        closs   += loss.item() * labels.size(0) # 배치 샘플 수 x 배치의 loss
        total   += labels.size(0) #배치 크기
        correct += (predicted == labels).sum().item() # 배치 길이 * boolean 중에 맞는 거 개수 
        tepoch.set_postfix(loss=closs/total, acc_pct=correct/total*100)

    return (closs/total), (correct/total)
                
                
                
num_classes=10
model = ResNet50(num_classes)
if torch.cuda.is_available():
    model.cuda()

learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

criterion = nn.CrossEntropyLoss()
max_epoch = 100
##Train Model##
for epoch in range(max_epoch):
    tloss, tacc = process_epoch(model, criterion, train_loader, optimizer, trainmode=True)
    vloss, vacc = process_epoch(model, criterion, val_loader, optimizer, trainmode=False)
    print('Epoch {:d} completed. Train loss {:.3f} Val loss {:.3f} Train accuracy {:.1f}% Test accuracy {:.1f}%'.format(epoch,tloss,vloss,tacc*100,vacc*100))
    scheduler.step()
    if(epoch+1)%5 == 0:
        save_path = os.path.join(save_dir,f'resnet50_epoch_{epoch+1}.pth')
        torch.save(model.state_dict(), save_path)
        print(f'Model saved at {save_path}')

