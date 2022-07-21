from cProfile import label
import os
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from PIL import Image


class CELEBA_Customized(Dataset):

    def __init__(self, path):
        super().__init__()
        self.image_path = path + '/img_align_celeba/'
        self.label_path = path + '/list_landmarks_align_celeba.txt'

    def __len__(self):
        return len(os.listdir(self.image_path))

    def __getitem__(self, index):

        if index + 1 < 10:
            filename = '00000' + str(index + 1) + '.jpg'
        elif index  + 1 < 100:
            filename = '0000' + str(index + 1) + '.jpg'
        elif index + 1 < 1000:
            filename = '000' + str(index + 1) + '.jpg'
        elif index + 1 == 1000:
            filename = '00' + str(index + 1) + '.jpg'
        else:
            pass

        image = Image.open(self.image_path + filename)
        file = open(self.label_path)

        labels = file.read().splitlines()
        label_info = labels[index + 2]
        temp = label_info.split(" ")
        res = [int(i) for i in temp[1:] if i != '']
        label = torch.tensor(res)
        
        # Define a transform to convert PIL
        # image to a Torch tensor
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        img_tensor = transform(image)

        return img_tensor, label
