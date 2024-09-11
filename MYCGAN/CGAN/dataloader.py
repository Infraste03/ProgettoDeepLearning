"""
The  code defines a CelebA dataset class and a data loader function for loading and
preprocessing images with specified attributes.

:param image_dir: The `image_dir` parameter refers to the directory where the images are stored for
the CelebA dataset. This directory contains the image files that will be used for training or
testing the model
:param attr_path: The `attr_path` parameter in the code refers to the path to a file containing
attribute information for the CelebA dataset. This file contains information about various
attributes associated with each image in the dataset, such as whether the person in the image is
wearing sunglasses, smiling, etc
:param selected_attrs: Selected_attrs is a list of attributes that you want to use for filtering the
CelebA dataset. These attributes could be characteristics like "smiling", "wavy hair", "eyeglasses",
etc. The selected_attrs list is used to identify and filter images based on these attributes during
dataset initialization
:param crop_size: The `crop_size` parameter in the `get_loader` function is used to specify the size
of the center crop to be taken from the images. In this case, it is set to 178. The `CenterCrop`
transformation is applied to the images with this specified crop size before resizing them, defaults
to 178 (optional)
:param image_size: The `image_size` parameter in the code snippet you provided is used to specify
the size of the images in the dataset. In this case, it is set to 64, which means that the images
will be resized to a square shape with dimensions 64x64 pixels during preprocessing, defaults to 128
(optional)
:param batch_size: Batch size refers to the number of samples that will be propagated through the
neural network at once during training. It is a hyperparameter that can affect the training process
and model performance. A larger batch size can lead to faster training times but may require more
memory, while a smaller batch size can provide more, defaults to 16 (optional)
:param dataset: The dataset parameter specifies the name of the dataset being used. In this case, it
is set to 'CelebA', which is a popular dataset of celebrity faces, defaults to CelebA (optional)
:param mode: The `mode` parameter in the code refers to whether the dataset is being used for
training or testing. It can take on two values: 'train' or 'test'. When 'train' is specified, the
dataset is used for training the model, while 'test' indicates that the dataset is, defaults to
train (optional)
:param num_workers: The `num_workers` parameter in the `get_loader` function specifies the number of
subprocesses to use for data loading. It controls the number of parallel data loading processes to
speed up data retrieval. Increasing the `num_workers` can potentially speed up the data loading
process by utilizing multiple CPU cores for, defaults to 1 (optional)
:return: The `get_loader` function returns a data loader object that can be used to iterate over
batches of images and their corresponding labels from the CelebA dataset.
"""
import os
import random
import torch
import torch.utils.data
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


image_size = 64

# Define the CelebA Dataset
class CelebA(data.Dataset):
    def __init__(self, image_dir, attr_path, selected_attrs, transform, mode):
        self.image_dir = image_dir
        self.attr_path = attr_path
        self.selected_attrs = selected_attrs
        self.transform = transform
        self.mode = mode
        self.train_dataset = []
        self.test_dataset = []
        self.attr2idx = {}
        self.idx2attr = {}
        self.preprocess()
        self.num_images = len(self.train_dataset) if mode == 'train' else len(self.test_dataset)
        print(f"Initialized {mode} dataset with {self.num_images} samples.")
        self.check_dataset()

    def preprocess(self):
        lines = [line.rstrip() for line in open(self.attr_path, 'r')]
        all_attr_names = lines[1].split()
        for i, attr_name in enumerate(all_attr_names):
            self.attr2idx[attr_name] = i
            self.idx2attr[i] = attr_name

        lines = lines[2:]
        random.seed(1234)
        random.shuffle(lines)
        for i, line in enumerate(lines):
            split = line.split()
            filename = split[0]
            values = split[1:]

            label = [values[self.attr2idx[attr_name]] == '1' for attr_name in self.selected_attrs]

            if (i+1) < 30050:
                self.test_dataset.append([filename, label])
            else:
                self.train_dataset.append([filename, label])

        print(f"Train dataset size: {len(self.train_dataset)}")
        print(f"Test dataset size: {len(self.test_dataset)}")

    def check_dataset(self):
        missing_files = []
        for dataset in [self.train_dataset, self.test_dataset]:
            for filename, _ in dataset:
                image_path = os.path.join(self.image_dir, filename)
                if not os.path.isfile(image_path):
                    missing_files.append(image_path)
        if missing_files:
            print(f"Missing files: {len(missing_files)}")
            for path in missing_files:
                print(path)

    def __getitem__(self, index):
        dataset = self.train_dataset if self.mode == 'train' else self.test_dataset
        filename, label = dataset[index]
        image_path = os.path.join(self.image_dir, filename)
        if not os.path.isfile(image_path):
            print(f"File not found: {image_path}")
            # Return a blank image and the label
            return torch.zeros((3, image_size, image_size)), torch.FloatTensor(label)
        image = Image.open(image_path)
        return self.transform(image), torch.FloatTensor(label)

    def __len__(self):
        return self.num_images

def get_loader(image_dir, attr_path, selected_attrs, crop_size=178, image_size=128, 
               batch_size=16, dataset='CelebA', mode='train', num_workers=1):
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'CelebA':
        dataset = CelebA(image_dir, attr_path, selected_attrs, transform, mode)

    print(f"Number of images in {mode} dataset: {len(dataset)}")  # Debugging line

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader