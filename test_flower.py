import torch
import os
import glob
import torch.nn.functional as F
import torch.optim as optim
import PIL
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from matplotlib import pyplot as plt

class flower_dataset(Dataset):
    def __init__(self, data_dir, transforms=None):
        """
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform，数据预处理
        """
        # data_info存储所有图片路径和标签，在DataLoader中通过index读取样本
        self.data_info = self.get_img_info(data_dir)
        self.transforms = transforms

    def __getitem__(self, index):
        # 通过 index 读取样本
        path_img, label = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     # 0~255
        if self.transforms:
            data = self.transforms(img)   # 在这里做transform，转为tensor等等
        # 返回是样本和标签
        return data, label

    # 返回所有样本的数量
    def __len__(self):
        return len(self.data_info)

    def dtr(self):
        image=[]
        labels=[]
        for i in range(len(self.data_info)):
            path_img, label = self.data_info[i]
            img = Image.open(path_img).convert('RGB')
            image.append(img)
            labels.append(label)
        return image,labels


    @staticmethod
    def get_img_info(data_dir):
        data_info = list()
        for i in range(17):
            paths = glob.glob(os.path.join(data_dir+str(i)+'/', '*.jpg'))
            for path_img in paths:
                data_info.append((path_img,i))
        return data_info

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 3, kernel_size=5, stride=5)    #50*50
        self.conv2 = torch.nn.Conv2d(3, 32, kernel_size=5)             #23*23
        self.conv3 = torch.nn.Conv2d(32, 64, kernel_size=4)            #10*10
        self.conv4 = torch.nn.Conv2d(64, 128, kernel_size=3)           #4*4
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc1 = torch.nn.Linear(2048, 512)
        self.fc2 = torch.nn.Linear(512, 17)
 
    def forward(self, x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = F.relu(self.pooling(self.conv3(x)))
        x = F.relu(self.pooling(self.conv4(x)))
        x = x.view(batch_size, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


batch_size = 64
transform = transforms.Compose([
    transforms.Resize((500,500)),
    transforms.ToTensor(),
    transforms.Normalize((0.42047042, 0.42260465, 0.28259334), (0.24752693, 0.22665608, 0.20883438))
])
model=Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
state_dict=torch.load('net_params.pth')
model.load_state_dict(state_dict)
#path=input("请输入图片路径:\n")
path='/Users/xuchengtian/Desktop/code/AI/17flowers/test/flower0/image_0022.jpg'
image_PIL = PIL.Image.open(path)
image_tensor = transform(image_PIL)
image_tensor.unsqueeze_(0)
image_tensor = image_tensor.to(device)
out = model(image_tensor)
_,predicted = torch.max(out, 1)
image_PIL.show()
print('It is flower'+str(int(predicted)))