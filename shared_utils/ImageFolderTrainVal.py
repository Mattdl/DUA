""" Pytorch ImageFolder overwrites. """
import bisect
import os
import os.path

import torch
import torch.utils.data as data
from PIL import Image
from torchvision import datasets
from itertools import accumulate

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)


def make_dataset(dir, class_to_idx, file_list):
    images = []
    # print('here')
    dir = os.path.expanduser(dir)
    set_files = [line.rstrip('\n') for line in open(file_list)]
    for target in sorted(os.listdir(dir)):
        # print(target)
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if is_image_file(fname):
                    dir_file = target + '/' + fname
                    # print(dir_file)
                    if dir_file in set_files:
                        path = os.path.join(root, fname)
                        item = (path, class_to_idx[target])
                        images.append(item)

    return images


class ImageFolderTrainVal(datasets.ImageFolder):
    """ Overwrite PyTorch ImageFolder for train/val split. """

    def __init__(self, root, files_list, transform=None, target_transform=None,
                 loader=default_loader, classes=None):
        if classes is None:
            classes, class_to_idx = find_classes(root)
        else:
            class_to_idx = {classes[i]: i for i in range(len(classes))}
        print(root)
        imgs = make_dataset(root, class_to_idx, files_list)
        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                                                                             "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.root = root
        self.samples = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader


class ImagePathlist(data.Dataset):
    """
    Adapted from: https://github.com/pytorch/vision/issues/81
    Load images and labels from lists containing paths.
    """

    def __init__(self, imlist, classes, targetlist=None, root='', transform=None, loader=default_loader):
        """
        :param imlist: list of paths (str) to the images
        :param targetlist: list of labels (str) of the images
        """
        super().__init__()
        self.imlist = imlist
        self.targetlist = targetlist
        self.root = root
        self.transform = transform
        self.loader = loader

        self.classes = sorted(classes)
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}

    def __getitem__(self, index):
        impath = self.imlist[index]

        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        if self.targetlist is not None:
            target = self.targetlist[index]
        else:
            # extract target from img path
            try:
                classname = os.path.basename(os.path.dirname(impath))
                target = self.class_to_idx[classname]
            except:
                pass

        return img, target

    def __len__(self):
        return len(self.imlist)


class ConcatDatasetDynamicLabels(torch.utils.data.ConcatDataset):
    """
    Dataset to concatenate multiple datasets.
    Purpose: useful to assemble different existing datasets, possibly
    large-scale datasets as the concatenation operation is done in an
    on-the-fly manner.
    Arguments:
        datasets (sequence): List of datasets to be concatenated
        the output labels are shifted by the dataset index which differs from the pytorch implementation that return the original labels
    """

    def __init__(self, datasets, classes_len):
        """
        :param datasets: List of Imagefolders
        :param classes_len: List of class lengths for each imagefolder
        """
        super(ConcatDatasetDynamicLabels, self).__init__(datasets)
        self.cumulative_classes_len = list(accumulate(classes_len))

    def __getitem__(self, idx):
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
            img, label = self.datasets[dataset_idx][sample_idx]
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            img, label = self.datasets[dataset_idx][sample_idx]
            label = label + self.cumulative_classes_len[dataset_idx - 1]  # Shift Labels
        return img, label
