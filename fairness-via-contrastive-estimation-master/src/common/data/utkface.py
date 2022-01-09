import torch
import cv2
import numpy as np
import torch.nn as nn
import imageio
import random
from torch.utils.data import Dataset
import os
import pandas as pd
import torch.optim as optim
from skimage import transform
from tensorflow.keras.utils import to_categorical

def groupAge(age):
#     [0, 5, 18, 24, 26, 27, 30, 34, 38, 46, 55, 65, len(ages)])
    if age>=0 and age<5:
        return 0
    elif age>=5 and age<18:
        return 1
    elif age>=18 and age<24:
        return 2
    elif age>=24 and age<26:
        return 3
    elif age>=26 and age<27:
        return 4
    elif age>=27 and age<30:
        return 5
    elif age>=30 and age<34:
        return 6
    elif age>=34 and age<38:
        return 7
    elif age>=38 and age<46:
        return 8
    elif age>=46 and age<55:
        return 9
    elif age>=55 and age<65:
        return 10
    else:
        return 11

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        landmarks = landmarks * [new_w / w, new_h / h]

        return {'image': img, 'landmarks': landmarks}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]

        landmarks = landmarks - [left, top]

        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}



class PrivacyDataLoader:
    def __init__(self, train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset):
        self.train_data = train_data
        self.train_label = train_label
        self.train_sensitive_label = train_sensitive_label
        self.test_data = test_data, test_label
        self.test_label = test_label
        self.test_sensitive_label = test_sensitive_label
        self.trainset = trainset
        self.testset = testset

class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, sensitive_tensor):
        Dataset.__init__(self)
        assert data_tensor.size(0) == target_tensor.size(0)
        assert target_tensor.size(0) == sensitive_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.sensitive_tensor = sensitive_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.sensitive_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)

def _xavier_init_(m: nn.Module):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight)

class FolderDataset(Dataset):
    def __init__(self, image_names, labels, rootpath, pub_att='age', sen_att='gender', transform=None):
        Dataset.__init__(self)
        self.labels = labels
        self.image_names = image_names
        self.rootpath = rootpath + '/images'
        self.transform = transform
        self.sensitive_attribute = sen_att
        self.public_attribute = pub_att
        if self.sensitive_attribute not in ['gender', 'age', 'ethnicity']: raise Exception (f'age, gender, or ethnicity only.')

    def __getitem__(self, idx):
        try:
            img = imageio.imread(self.rootpath+'/'+ self.image_names[idx])

            img = cv2.resize(np.array(img), (50,50))
            img = np.array(img)/255.0
            img = img.transpose((2, 0, 1))

            data = self.labels.loc[self.labels['image_id'] == self.image_names[idx][:-4]].values
            # age = to_categorical(groupAge(data[0][1]), num_classes=12, dtype='float32')
            # ethnicity = to_categorical(data[0][3], num_classes=5, dtype='float32')

            pub_label = self.attribute_choice(data, self.public_attribute)
            sen_label = self.attribute_choice(data, self.sensitive_attribute)

            return torch.from_numpy(img), pub_label, sen_label

        except IndexError:
            print(idx)
            print("[ERROR]", self.image_names[idx])

    def __len__(self):
        return len(self.image_names)

    def attribute_choice(self, data, choice):

        if choice == 'age':
            return groupAge(data[0][1])
        elif choice == 'gender':
            return data[0][2]
        else:
            return data[0][3]

def formatdata(labels, train_count, validation_count, images, eth_choice, verbose=False):

    partitions = {'train': [],
                    'validation': [],
                    'test': []}
    labels_dict = {'train_age': [], 'train_gender': [], 'train_ethnicity': [],
                    'validation_age': [], 'validation_gender': [], 'validation_ethnicity': [],
                    'test_age': [], 'test_gender': [], 'test_ethnicity': []}
    random.seed(1)

    print(f"[INFO] Preparing train and test data....")

    ethnicity = np.zeros(5)
    gender = np.zeros(2)
    age = np.zeros(12)

    test_ethnicity = np.zeros(5)
    test_gender = np.zeros(2)
    test_age = np.zeros(12)

    for cnt, image in enumerate(images):
        try:
            data = labels.loc[labels['image_id'] == image[:-4]].values
        except IndexError:
                    print("[ERROR]", image)

        if data[0][3] != 4:
            if cnt < train_count and ethnicity[data[0][3]] < 3000: # and ethnicity[data[0][3]] < 6000:
                partitions['train'].append(image)
                labels_dict['train_age'].append(to_categorical(groupAge(data[0][1]), num_classes=12, dtype='float32'))
                labels_dict['train_gender'].append(data[0][2])
                labels_dict['train_ethnicity'].append(to_categorical(data[0][3], num_classes=5, dtype='float32'))
                ethnicity[data[0][3]] += 1
                age[groupAge(data[0][1])] += 1
                gender[data[0][2]] += 1
            else:
                partitions['test'].append(image)
                labels_dict['test_age'].append(to_categorical(groupAge(data[0][1]), num_classes=12, dtype='float32'))
                labels_dict['test_gender'].append(data[0][2])
                labels_dict['test_ethnicity'].append(to_categorical(data[0][3], num_classes=5, dtype='float32'))

    print(f"[INFO] Done. Train={len(partitions['train'])}. Test={len(partitions['test'])}")

    if verbose:
        print("[INFO] Training Data")
        print("Size of train data: ", len(partitions['train']))
        print("Size of age as label: ", len(labels_dict['train_age']))
        print("Size of gender as label: ", len(labels_dict['train_gender']))
        print("Size of ethnicity as label: ", len(labels_dict['train_ethnicity']))
        print("\n")
        print("[INFO] Validation Data")
        print("Size of validation data: ", len(partitions['validation']))
        print("Size of age as label: ", len(labels_dict['validation_age']))
        print("Size of gender as label: ", len(labels_dict['validation_gender']))
        print("Size of ethnicity as label: ", len(labels_dict['validation_ethnicity']))
        print("\n")
        print("[INFO] Test Data")
        print("Size of test data: ", len(partitions['test']))
        print("Size of age as label: ", len(labels_dict['test_age']))
        print("Size of gender as label: ", len(labels_dict['test_gender']))
        print("Size of ethnicity as label: ", len(labels_dict['test_ethnicity']))

    return partitions, labels_dict

class UTKFaceDataLoader(PrivacyDataLoader):
    def __init__(self, train_count, validation_count, test_count, n_sensitive_class, n_target_class, sensitive_attribute, public_attribute, dataset_path,
                 train_batch_size = 128, test_batch_size = 1000,
                 train_data=None, train_label=None, train_sensitive_label=None, test_data=None, test_label=None,
                 test_sensitive_label=None, trainset=None, testset=None,
                 embed_length=512, atk_layer=0,
                 adv_decay=1e-3, dis_choice = 3, adv_choice=3, tar_choice = 3,
                 adv_lr=0.01, net_lr=1e-3, dis_lr=1e-3, tar_lr=1e-3, eth_choice = 0):

        self.name = "UTKFace"
        super().__init__(train_data, train_label, train_sensitive_label, test_data, test_label, test_sensitive_label, trainset, testset)

        # adversary
        self.embed_length = embed_length
        self.atk_layer = atk_layer
        self.adv_lr = adv_lr
        self.adv_decay = adv_decay
        self.adv_choice = adv_choice

        # original model
        self.net_lr = net_lr
        self.tar_lr = tar_lr
        self.tar_choice = tar_choice
        self.eth_choice = eth_choice


        # discriminator
        self.dis_lr = dis_lr
        self.dis_choice = dis_choice

        self.dataset_path = dataset_path
        # self.label_path = 'datasets/face_dataset.csv'
        self.train_count = train_count
        self.validation_count = validation_count
        self.test_count = test_count

        # gender = 2, ethnicity = 5, age = 12
        self.public_attribute=public_attribute
        self.sensitive_attribute = sensitive_attribute
        self.n_target_class = n_target_class
        self.n_sensitive_class = n_sensitive_class
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.max_epoch_encoder = 200
        self.max_epoch_discriminator = 200

    def load(self):

        # Obtain labels
        self.labels = pd.read_csv(self.dataset_path + '/datasets/face_dataset.csv')
        self.labels = self.labels.sample(frac=1).reset_index(drop=True)
        images = os.listdir(self.dataset_path + '/images')
        print('[INFO] Total number of images: ', len(images))

        # Splitting labels. Data is note read now but at Dataloader to be memory efficient.
        # train:validation:test = 70:20:10 = 16596:4742:2370
        self.partitions, labels_dict = formatdata(self.labels, self.train_count, self.validation_count, images, self.eth_choice)

        # train/test dataset and private label parameters
        self.train_size = self.train_count
        self.test_size = len(self.labels) - self.validation_count - self.train_count

        # create train/test datasets
        print(f"[INFO] (UTKFace) Public label={self.public_attribute}, Sensitive label={self.sensitive_attribute}")
        self.trainset = FolderDataset(self.partitions['train'], self.labels, self.dataset_path, pub_att = self.public_attribute, sen_att=self.sensitive_attribute )
        self.validset = FolderDataset(self.partitions['validation'], self.labels, self.dataset_path, pub_att = self.public_attribute, sen_att=self.sensitive_attribute )
        self.testset  = FolderDataset(self.partitions['test'], self.labels, self.dataset_path, pub_att = self.public_attribute, sen_att=self.sensitive_attribute)

        # print sets stats
        train_public_count, train_sen_count = self.set_stats('train')
        print(f"[INFO] Stats | Train public attribute:{train_public_count} | Train sensitive attribute {train_sen_count} ")
        valid_public_count, valid_sen_count = self.set_stats('validation')
        print(f"[INFO] Stats | Valid public attribute:{valid_public_count} | Valid sensitive attribute {valid_sen_count} ")
        test_public_count, test_sen_count = self.set_stats('test')
        print(f"[INFO] Stats | Test public attribute:{test_public_count} | Test sensitive attribute {test_sen_count} ")


        # if self.public_attribute == 'ethnicity':
            # weight = torch.tensor(train_public_count)/torch.sum(train_public_count)

        print('[INFO] Initialize dataloaders')
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=self.train_batch_size, shuffle=True, num_workers=1, pin_memory=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=self.test_batch_size, shuffle=False, num_workers=1)

    def set_stats(self, setname):
        public_count = np.zeros(self.n_target_class)
        sensitive_count = np.zeros(self.n_sensitive_class)

        for image_name in self.partitions[setname]:
            data = self.labels.loc[self.labels['image_id'] == image_name[:-4]].values
            # sensitive_count[groupAge(data[0][1])] += 1
            # public_count[data[0][2]] += 1

            if self.public_attribute == 'age':
                public_count[groupAge(data[0][1])] += 1
            elif self.public_attribute == 'gender':
                public_count[data[0][2]] += 1
            else:
                public_count[data[0][3]] += 1

            if self.sensitive_attribute == 'age':
                sensitive_count[groupAge(data[0][1])] += 1
            elif self.sensitive_attribute == 'gender':
                sensitive_count[data[0][2]] += 1
            else:
                sensitive_count[data[0][3]] += 1

        return public_count, sensitive_count

    def attribute_choice(self, data, choice):

        if choice == 'age':
            return groupAge(data[0][1])
        elif choice == 'gender':
            return data[0][2]
        else:
            return data[0][3]

def load_utkface(val_size=0.0):
    dataset_path = '/home/hjchris/data/UTKFace'

    utk_dl=UTKFaceDataLoader(dataset_path = dataset_path,
                                            train_count = 18964,                     # Total image is 23705. Randomly choose [train_count].
                                            validation_count = 0,                    # There is no validation in the training.
                                            test_count = 4741,                       # Redundent parameter.
                                            train_batch_size = 128,
                                            test_batch_size = 400,
                                            public_attribute = 'gender',
                                            n_target_class = 2,
                                            sensitive_attribute = 'ethnicity',             # gender or ethnicity.
                                            n_sensitive_class = 4,                  # gender = 2, ethnicity = 5. (for output size of target)
                                            )
    return {"train": utk_dl.trainloader,
            "valid": None,
            "test": utk_dl.testloader
    }
