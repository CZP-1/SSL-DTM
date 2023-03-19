import cv2
from PIL import Image
import copy
import torch
from .randaugment import RandAugment

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
        self.strong_transfrom = copy.deepcopy(transform)
        self.strong_transfrom.transforms.insert(0, RandAugment(3,5))

    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        out3 = self.strong_transfrom(inp)
        return out1, out2, out3

def get_affwild(train_file_list,val_file_list,unlabel_file_list,  transform_train=None, transform_val=None):
    
    train_labeled_dataset = Dataset_Affwild(train_file_list, transform=transform_train)
    train_unlabeled_dataset = Dataset_Affwild(unlabel_file_list,  transform=TransformTwice(transform_train))
    test_dataset = Dataset_Affwild(val_file_list,transform=transform_val)

    print (f"#Labeled: {len(train_labeled_dataset)} #Unlabeled: {len(train_unlabeled_dataset)}")
    return train_labeled_dataset, train_unlabeled_dataset, test_dataset


def img_loader(path):
    try:
        with open(path, 'rb') as f:
            img = cv2.imread(path)
            img = Image.fromarray(img)
            return img
    except IOError:
        print('Cannot load image ' + path)

class Dataset_Affwild(torch.utils.data.Dataset):
    def __init__(self, file_list, transform=None, loader=img_loader):

        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        import pandas as pd
        data = pd.read_csv(file_list)
        image_list = data.path.values.tolist()
        label_list = data.label.values.tolist()

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = 8

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(img_path)
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.image_list)