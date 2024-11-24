import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import ImageDraw


# several data augumentation strategies
def randomFlip(img, label, edge_label):
    # left right flip
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
        edge_label = edge_label.transpose(Image.FLIP_LEFT_RIGHT)
    return img, label, edge_label


def randomCrop(image, label, edge_label):
    border = 40
    image_width = image.size[0]
    image_height = image.size[1]
    crop_win_width = np.random.randint(image_width - border, image_width)
    crop_win_height = np.random.randint(image_height - border, image_height)
    random_region = (
        (image_width - crop_win_width) >> 1, (image_height - crop_win_height) >> 1, (image_width + crop_win_width) >> 1,
        (image_height + crop_win_height) >> 1)
    return image.crop(random_region), label.crop(random_region), edge_label.crop(random_region)


def randomRotation(image, label, edge_label):
    mode = Image.BICUBIC
    if random.random() > 0.6:
        random_angle = np.random.randint(-90, 90)
        image = image.rotate(random_angle, mode)
        label = label.rotate(random_angle, mode)
        edge_label = edge_label.rotate(random_angle, mode)
    return image, label, edge_label


# dataset for training
class InpaintingDataset(data.Dataset):
    def __init__(self, image_root, gt_root, edge_gt_root, trainsize, dataset_name='IID'):
        self.trainsize = trainsize
        
        if dataset_name == 'IID':
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png') or f.endswith('.jpg')]
            self.edge_gts = [edge_gt_root + f for f in os.listdir(edge_gt_root) if f.endswith('.png') or f.endswith('.jpg')]
        else:
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg')
                    or f.endswith('.png')]
            self.gts = [gt_root + f.split('/', -1)[-1] for f in self.images]
            self.edge_gts = [edge_gt_root + f.split('/', -1)[-1] for f in self.images]
        
        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.edge_gts = sorted(self.edge_gts)
        self.load_filesNum()
        
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.448, 0.433, 0.397], [0.230, 0.226, 0.235])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        self.edge_gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])
        # get size of dataset
        self.size = len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        H, W = image.height, image.width
        gt = self.binary_loader(self.gts[index], H, W)
        edge_gt = self.binary_loader(self.edge_gts[index], H, W)
        
        # data augumentation
        image, gt, edge_gt = randomFlip(image, gt, edge_gt)
        image, gt, edge_gt = randomCrop(image, gt, edge_gt)
        image, gt, edge_gt = randomRotation(image, gt, edge_gt)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        edge_gt = self.edge_gt_transform(edge_gt)

        return image, gt, edge_gt
        
    def load_filesNum(self):
        print(f'loading {len(self.images)} images')

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path, h, w):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('L')
                if img.size != (w, h):
                    img = img.resize(w, h)
            return img
        except:
            img = Image.new('L', (w, h))
            draw = ImageDraw.Draw(img)
            draw.point((random.randint(5, w-5), random.randint(5, h-5)), fill='white')
            return img


    def __len__(self):
        return self.size



# dataloader for training
def get_loader(image_root, gt_root, edge_root, dataset_name, batchsize, trainsize,
               shuffle=True, num_workers=12, pin_memory=True):
    dataset = InpaintingDataset(image_root, gt_root, edge_root, trainsize, dataset_name)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader


# test dataset and loader
class test_dataset:
    def __init__(self, image_root, gt_root, testsize, dataset_name='IID'):
        self.testsize = testsize
        
        if dataset_name == 'IID':
            # get filenames IID data
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.png') or f.endswith('.jpg')]
            self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.png')]
        else:
            # get filenames DEFACTO data
            self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
            self.gts = [gt_root + f.split('/', -1)[-1] for f in self.images]
            
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.430, 0.387], [0.239, 0.225, 0.230])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        H, W = image.height, image.width
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index], H, W)

        name = self.images[self.index].split('/')[-1]


        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'

        self.index += 1
        self.index = self.index % self.size

        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')
    
    def binary_loader(self, path, h, w):
        try:
            with open(path, 'rb') as f:
                img = Image.open(f).convert('L')
                if img.size != (w, h):
                    img = img.resize(w, h)
            return img
        except:
            img = Image.new('L', (w, h))
            draw = ImageDraw.Draw(img)
            draw.point((random.randint(5, w-5), random.randint(5, h-5)), fill='white')
            return img

    def __len__(self):
        return self.size
    
    
    
    


