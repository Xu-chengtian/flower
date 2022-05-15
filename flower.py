import torch
import os
import glob
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.autograd import Variable
from matplotlib import pyplot as plt

from PIL import Image

def get_conf_matrix(dataset,model):
    data,target=dataset.dtr()
    matrix = torch.zeros((17, 17), dtype=torch.int32)
    result = model(data.unsqueeze(1).float().to('cpu'))
    result = torch.argmax(result, 1)
    for i, j in zip(result, target):
        matrix[i, j] += 1
    return matrix

def show_conf_matrix(conf_matrix):
    Emotion = 10
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)
    thresh = conf_matrix.max() / 2
    for x in range(Emotion):
        for y in range(Emotion):
            info = int(conf_matrix[y, x])
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")
    plt.tight_layout()
    plt.yticks(range(Emotion), labels)
    plt.xticks(range(Emotion), labels)
    plt.show()
    plt.close()

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
        a,b=self.data_info[891]
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
        '''
        # data_dir 是训练集、验证集或者测试集的路径
        for root, dirs, _ in os.walk(data_dir):
            # 遍历类别
            # dirs ['1', '100']
            for sub_dir in dirs:
                # 文件列表
                img_names = os.listdir(os.path.join(root, sub_dir))
                # 取出 jpg 结尾的文件
                img_names = list(filter(lambda x: x.endswith('.jpg'), img_names))
                # 遍历图片
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    # 图片的绝对路径
                    path_img = os.path.join(root, sub_dir, img_name)
                    # 标签，这里需要映射为 0、1 两个类别
                    label = rmb_label[sub_dir]
                    # 保存在 data_info 变量中
                    data_info.append((path_img, int(label)))
        return data_info
        '''
'''
def read_img(path):
    #os.listdir(path)表示在path路径下的所有文件和和文件夹列表
    #用cate记录五种花的文件路径
    #path='/Users/xuchengtian/Desktop/code/AI/17flowers/train/flower'
    imgs=[]  #存放所有的图片
    labels=[]  #图片的类别标签
    for i in range(17):
        paths = glob.glob(os.path.join(path+str(i)+'/', '*.jpg'))
        for pic in paths:
            img=cv2.imread(pic,cv2.IMREAD_COLOR)
            img=transform.resize(img,(500,500))
            img = cv2.normalize(img, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC3)
            imgs.append(img)
            labels.append(i)
    return np.array(imgs,np.float32),np.array(labels,np.int32)

def disturb(data,label):
    num_example=data.shape[0]
    arr=np.arange(num_example)
    np.random.shuffle(arr)
    img=data[arr]
    labels=label[arr]
    return img,labels

def allocate(data,label):
    inter = 0.8
    num_example = data.shape[0]
    s1 = np.int(num_example*inter)
    x_train = data[:s1]
    y_train = label[:s1]
    x_val = data[s1:]
    y_val = label[s1:]
    return x_train,y_train,x_val,y_val
'''
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

def train():
    for batch_idx, data in enumerate(train_loader, 0):
        inputs,target = data
        inputs,target = inputs.to(device),target.to(device)
        optimizer.zero_grad()
 
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        #if batch_idx % 20 == 19:
        print('[%d] loss:%.3f' % (batch_idx + 1, loss.item()))

def eval():
    correct=0
    total=0
    for data in test_loader:
        images, labels = data
        outputs = model(Variable(images))
        _,predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted==labels).sum()
    return correct/total



def getStat(train_data):
    '''
    Compute mean and variance for training data
    :param train_data: 自定义类Dataset(或ImageFolder即可)
    :return: (mean, std)
    '''
    print('Compute mean and variance for training data.')
    print(len(train_data))
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=1, shuffle=False, num_workers=0,
        pin_memory=True)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for X, _ in train_loader:
        for d in range(3):
            mean[d] += X[:, d, :, :].mean()
            std[d] += X[:, d, :, :].std()
    mean.div_(len(train_data))
    std.div_(len(train_data))
    print(list(mean.numpy()), list(std.numpy()))

'''
train_imgs,train_labels=read_img('/Users/xuchengtian/Desktop/code/AI/17flowers/train/flower')
test_imgs,test_labels=read_img('/Users/xuchengtian/Desktop/code/AI/17flowers/test/flower')
train_imgs,train_labels=disturb(train_imgs,train_labels)
train_imgs,train_labels,val_imgs,val_labels=allocate(train_imgs,train_labels)
model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
'''
flower_label={
    'flower0':0,
    'flower1':1,
    'flower2':2,
    'flower3':3,
    'flower4':4,
    'flower5':5,
    'flower6':6,
    'flower7':7,
    'flower8':8,
    'flower9':9,
    'flower10':10,
    'flower11':11,
    'flower12':12,
    'flower13':13,
    'flower14':14,
    'flower15':15,
    'flower16':16
}
batch_size = 64
transform = transforms.Compose([
    transforms.Resize((500,500)),
    transforms.ToTensor(),
    transforms.Normalize((0.42047042, 0.42260465, 0.28259334), (0.24752693, 0.22665608, 0.20883438))
])
train_dataset = flower_dataset(data_dir='/Users/xuchengtian/Desktop/code/AI/17flowers/train/flower', transforms=transform)
test_dataset = flower_dataset(data_dir='/Users/xuchengtian/Desktop/code/AI/17flowers/train/flower', transforms=transform)
train_loader = DataLoader(train_dataset,shuffle=True,batch_size=batch_size)
test_loader = DataLoader(test_dataset,shuffle=False,batch_size=batch_size)
#getStat(train_dataset)

model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
for i in range(20):
    train()
print(eval())