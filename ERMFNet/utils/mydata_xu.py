'''
#6300
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2,os
from scipy import signal
from torchvision.transforms import Compose, Normalize, ToTensor
import scipy
import codecs
class  Mydata(Dataset):


    def __init__(self, img_speed_path=r"E:\python\pythonProject5\GLMDrivenet-master\total\train.txt\\",
    img_path=r"E:\python\pythonProject5\GLMDrivenet-master\dataset_total\train\img\\",
    speed_path=r"E:\python\pythonProject5\GLMDrivenet-master\dataset_total\train\speed\\",
    transforms=None,test=False):

        self.img_speed_path=img_speed_path
        self.test=test
        self.img_path = img_path
        self.speed_path = speed_path
        self.transforms = transforms
        self.img_list=[]
        self.speed_list=[]
        self.label_list=[]

        with codecs.open(self.img_speed_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                list1=i.split()
                self.img_list.append(list1[0])
                self.speed_list.append(list1[1])
                self.label_list.append(list1[2])
        # print(self.img_list)#['0.jpg', '1.jpg']
        # print(self.speed_list)#['0.txt', '1.txt']
        # print(self.label_list)#['0', '1']

        self._vid_transform, self._speed_transform = self._get_normalization_transform()


    def _get_normalization_transform(self):
        _vid_transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _speed_transform = Compose([Normalize(mean=[0.0], std=[12.0])])
        return _vid_transform, _speed_transform

    def __len__(self):

        return len(open(self.img_speed_path,'r').readlines()) #dui  2

    def __getitem__(self, idx):

        image_path=os.path.join(self.img_path,self.img_list[idx])
        # print("idx image_path",idx,image_path) trainimg/1.jpg
        image = cv2.imread(image_path) #cv默认bgr hwc 我们正常读取图片是的通道顺序是h,w,c，但是通过pytorch中的ToTensor()处理之后，读出来的图片数据通道顺序就变成了c,h,w
        #cv2.imshow('imag',image)
        #image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224,224))
        image = image/255.0
        image = image.transpose(2, 0, 1)

        #print("image_shape",image.shape)
        txt_path=os.path.join(self.speed_path,self.speed_list[idx])
        # print("txt_path",txt_path)
        samples = []
        with codecs.open(txt_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                samples.append(i)
        samples=list(map(float, samples))
        samples=np.array(samples)

        frequencies, times, spectrogram =signal.spectrogram(samples, 1260, nperseg=512, noverlap=483)
        #print("specshape",spectrogram.shape)
        if spectrogram.shape != (257, 200):
            return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(np.random.rand(1, 257, 200)), torch.LongTensor([3])
        spectrogram = np.log(spectrogram + 1e-7)
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        image = self._vid_transform(torch.Tensor(image))
        speed = torch.Tensor(spectrogram.reshape(spec_shape))
        speed = self._speed_transform(speed)

        result=[int(self.label_list[idx])]
        #print("result",result)
        return image, speed, torch.LongTensor(result)'''

'''if __name__ == "__main__":

    train_datasets = Mydata()

    train_loader = DataLoader(train_datasets, batch_size=2, shuffle=True, num_workers=0)

    for subepoch, (img, speed, label) in enumerate(train_loader):

        print('label.shape')
        #print(label.shape)
        print(label)
        print('img.shape')
        print(img.shape)
        print('speed.shape')
        print(speed.shape)'''
#         label = label.squeeze(1)
#         idx = (label != 3).numpy().astype(bool)
#         print(idx.sum())
#         break

from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import cv2, os
from scipy import signal
from torchvision.transforms import Compose, Normalize, ToTensor
import scipy
import codecs
import pywt


class Mydata(Dataset):


    def __init__(self, img_speed_path=r"/root/1/GLMDrivenet-master/total/train.txt",
                     img_path=r"/root/1/GLMDrivenet-master/dataset_total/train/img",
                     speed_path=r"/root/1/GLMDrivenet-master/dataset_total/train/speed",
                     transforms=None, test=False):






        self.img_speed_path = img_speed_path
        self.test = test
        self.img_path = img_path
        self.speed_path = speed_path
        self.transforms = transforms
        self.img_list = []
        self.speed_list = []
        self.label_list = []

        with codecs.open(self.img_speed_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                list1 = i.split()
                self.img_list.append(list1[0])
                self.speed_list.append(list1[1])
                self.label_list.append(list1[2])

        self._vid_transform, self._speed_transform = self._get_normalization_transform()

    def _get_normalization_transform(self):
        _vid_transform = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        _speed_transform = Compose([Normalize(mean=[0.0], std=[12.0])])
        return _vid_transform, _speed_transform

    def __len__(self):
        return len(open(self.img_speed_path, 'r').readlines())

    def _wavelet_transform(self, samples):

        # 进行小波包分解
        wp = pywt.WaveletPacket(samples, 'db4', mode='symmetric', maxlevel=4)

        # 获取第4层所有节点的小波系数
        level4_nodes = wp.get_level(4)

        # 将小波系数重塑为目标形状 (257, 200)
        # 如果系数数量不够，进行填充或截断
        all_coeffs = []
        for node in level4_nodes:
            coeffs = node.data
            # 对每个节点的系数进行适当长度的处理
            target_length = 200
            if len(coeffs) > target_length:
                coeffs = coeffs[:target_length]
            else:
                # 填充到目标长度
                coeffs = np.pad(coeffs, (0, target_length - len(coeffs)), mode='constant')
            all_coeffs.append(coeffs)

        # 组合所有节点的系数
        wavelet_matrix = np.array(all_coeffs)

        # 如果节点数量不够257，进行填充
        if wavelet_matrix.shape[0] < 257:
            padding = np.zeros((257 - wavelet_matrix.shape[0], 200))
            wavelet_matrix = np.vstack([wavelet_matrix, padding])
        elif wavelet_matrix.shape[0] > 257:
            wavelet_matrix = wavelet_matrix[:257, :]

        return wavelet_matrix

    def __getitem__(self, idx):

        image_path = os.path.join(self.img_path, self.img_list[idx])
        image = cv2.imread(image_path)
        image = cv2.resize(image, (224, 224))
        image = image / 255.0
        image = image.transpose(2, 0, 1)

        txt_path = os.path.join(self.speed_path, self.speed_list[idx])
        samples = []
        with codecs.open(txt_path, 'r', 'ascii') as infile:
            for i in infile.readlines():
                i = i.strip('\n')
                samples.append(i)
        samples = list(map(float, samples))
        samples = np.array(samples)

        # 将原来的STFT替换为小波变换
        # frequencies, times, spectrogram = signal.spectrogram(samples, 1260, nperseg=512, noverlap=483)
        spectrogram = self._wavelet_transform(samples)

        # 保持原有的形状检查
        if spectrogram.shape != (257, 200):
            return torch.Tensor(np.random.rand(3, 224, 224)), torch.Tensor(
                np.random.rand(1, 257, 200)), torch.LongTensor([3])

        # 保持原有的对数变换
        spectrogram = np.log(np.abs(spectrogram) + 1e-7)  # 取绝对值避免对数负数
        spec_shape = list(spectrogram.shape)
        spec_shape = tuple([1] + spec_shape)

        image = self._vid_transform(torch.Tensor(image))
        speed = torch.Tensor(spectrogram.reshape(spec_shape))
        speed = self._speed_transform(speed)

        result = [int(self.label_list[idx])]
        return image, speed, torch.LongTensor(result)


if __name__ == "__main__":

    train_datasets = Mydata()

    train_loader = DataLoader(train_datasets, batch_size=2, shuffle=True, num_workers=0)

    for subepoch, (img, speed, label) in enumerate(train_loader):
        print('label.shape')
        print(label.shape)
        print(label)
        print('img.shape')
        print(img.shape)
        print('speed.shape')
        print(speed.shape)


