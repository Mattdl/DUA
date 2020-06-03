import os
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import numpy as np
import random
import tqdm

import torch
from torchvision import transforms, datasets
import torch.utils.data as data

import shared_utils.transforms
import shared_utils.utils as utils
from shared_utils.ImageFolderTrainVal import ImagePathlist

# Global vars
dataset_names = ["MITusertransform", "MITindoorscenes", "tinyimagenet", "numbers", "numbers_nb", "numbers_nb_bal"]


def parse(ds_name, usercount=None, user_cat_pref=None, user_cat_prefimgs=None, transform_severeness=None,
          save_imgfolders=False):
    """Parses string ds name to actual Training and Eval ds objects."""
    if ds_name == 'MITindoorscenes':
        eval_ds = IndoorScenesDataset(usercount=usercount,
                                      user_cat_pref=user_cat_pref,
                                      user_cat_prefimgs=user_cat_prefimgs,
                                      save_imgfolders=save_imgfolders)
        tr_ds = IndoorScenesDataset(save_imgfolders=True)
    elif ds_name == 'MITusertransform':
        eval_ds = IndoorScenesTransformDataset(usercount=usercount,
                                               user_cat_prefimgs=user_cat_prefimgs,
                                               save_imgfolders=save_imgfolders,
                                               severeness=transform_severeness)
        tr_ds = IndoorScenesDataset(save_imgfolders=True)
    elif ds_name == 'numbers':
        eval_ds = NumbersDataset(mode='ds_tasks_incl', save_imgfolders=save_imgfolders)
        tr_ds = NumbersDataset(mode='ds_tasks_incl', save_imgfolders=True)
    elif ds_name == 'numbers_nb':
        eval_ds = NumbersDataset(mode='nb_tasks', save_imgfolders=save_imgfolders)
        tr_ds = NumbersDataset(mode='nb_tasks', save_imgfolders=True)
    elif ds_name == 'numbers_nb_bal':
        eval_ds = NumbersDataset(mode='nb_tasks_bal', save_imgfolders=save_imgfolders)
        tr_ds = NumbersDataset(mode='nb_tasks_bal', save_imgfolders=True)
    else:
        raise NotImplementedError("DS not implemented: ", ds_name)
    return tr_ds, eval_ds


def get_nc_per_task(dataset):
    return [len(classes_for_task) for classes_for_task in dataset.classes_per_task.values()]


class CustomDataset(metaclass=ABCMeta):
    """
    Abstract properties/methods that can be used regardless of which subclass the instance is.
    """

    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def task_count(self): pass

    @property
    @abstractmethod
    def input_size(self): pass

    @property
    @abstractmethod
    def imgfolders(self): pass

    @abstractmethod
    def get_imgfolder_path(self, task_name, notransform):
        pass

    @abstractmethod
    def get_taskname(self, task_index):
        pass


class NumbersDataset(CustomDataset):
    """
    3 Tasks: MNIST, SVHN, CMNIST

    3 users, each has samples from a different dataset.
    with subclasses only
    e.g. USER 1: classes 0,1 from MNIST
        USER 2: classes 2,3, from SVHN
        USER 3: classes 4,5 from CMNIST

    The server trains only on the remaining classes of all tasks
    e.g. 2 to 9 from MNIST; 1,2 and 4 to 9 SVHN;...
    """
    usercount = None
    task_count = None
    base_name = 'NumbersDataset'
    name = base_name
    classes_per_task = OrderedDict()
    input_size = (28, 28)
    # task_sequence = ['MNIST', 'SVHN', 'CMNIST']
    task_sequence = None
    modes = ['nb_tasks', 'nb_tasks_bal', 'ds_tasks_incl']
    imgfolders = {}

    def __init__(self, mode=None, save_imgfolders=False, create_imgfolders=True):
        utils.set_random(seed=27)
        config = utils.get_parsed_config()
        self.root_path = os.path.join(utils.read_from_config(config, 'ds_root_path'))

        # Mode
        self.mode = self.modes[0] if mode is None else mode
        self.name = '_'.join([self.base_name, self.mode])

        # Task Sequence
        if 'ds_tasks' in self.mode:  # Each dataset is a separate task
            self.task_sequence = ['MNIST', 'SVHN']
            self.task_count = len(self.task_sequence)
            self.usercount = 2
        elif 'nb_tasks' in self.mode:  # MNIST+SVHN combined constitute
            self.task_sequence = ["numbers={},{}".format(i, i + 1) for i in range(0, 10, 2)]
            self.task_count = 5
            # self.usercount = 10 # Each user 1 task only
            self.usercount = 2
        else:
            raise NotImplementedError("Mode is not implemented for ds: {}".format(self.mode))

        # Imgfolders
        self.usercount = 2
        self.save_imgfolders = save_imgfolders
        self.imgfolders = {}  # In Memory
        self.imgfolder_paths = {'train': OrderedDict(),  # task_name gives imfolder_path
                                'user': {u: {t: None for t in self.task_sequence}
                                         for u in
                                         range(1, self.usercount + 1)}}  # Useridx gives according imgfolder_path

        # Init MNIST
        self.mnist_user_path = None
        self.mnist_tr_path = None

        self.tr_MNIST_targets = None
        self.user_MNIST_targets = [0, 1]

        # Init SVHN
        self.svhn_user_path = None
        self.svhn_tr_path = None

        self.user_SVHN_targets = [0, 1]
        self.tr_SVHN_targets = None

        # IMGFOLDERS
        if create_imgfolders:
            if mode == 'ds_tasks_incl':
                self.create_custom_MNIST_include_usertargets(self.user_MNIST_targets, overwrite=False)
                self.create_custom_SVHN(self.user_SVHN_targets, overwrite=False)
            # elif mode == 'ds_tasks_excl':
            #     raise Exception("ds_tasks_excl mode not used.")
            #     self.create_custom_MNIST_exclude_usertargets(self.user_MNIST_targets)
            elif 'nb_tasks' in self.mode:  # MNIST+SVHN combined constitute
                self.create_custom_MNIST_SVHN_number_tasks(overwrite=False, balanced='bal' in self.mode)
            else:
                raise NotImplementedError("Mode is not implemented for ds: {}".format(self.mode))
            self.cat_per_task = {task: sorted([i for i in range(0, 10)]) for task in self.task_sequence}

    def create_custom_MNIST_SVHN_number_tasks(self, bs=1000, overwrite=False, balanced=False):
        """
        Classes = different
        Creates custom Dataset objects from original MNIST. User categories are also included in the training set.
        User samples are not available for the server set: they are extracted from the original MNIST test set.
        :param bs: batch size to iterate over original MNIST.
        :return:
        """

        tr_balanced_cnt = 10000

        # Init
        self.targets = [[i, i + 1] for i in range(0, 10, 2)]  # both eval/tr
        print("targets:{}".format(self.targets))

        svhn_root = os.path.join(self.root_path, 'SVHN')
        mnist_root = os.path.join(self.root_path, 'MNIST')

        seq_root = os.path.join(self.root_path, 'MNIST+SVHN')
        cust_seq_root = os.path.join(seq_root, 'split')
        utils.create_dir(cust_seq_root)

        # SVHN dataloader
        orig_svhn_tr_ds = datasets.SVHN(svhn_root, split='train', download=True,
                                        transform=transforms.Compose([
                                            transforms.CenterCrop(self.input_size[0]),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                        ]))
        orig_svhn_test_ds = datasets.SVHN(svhn_root, split='test', download=True,
                                          transform=transforms.Compose([
                                              transforms.CenterCrop(self.input_size[0]),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ]))
        orig_svhn_tr_dsloader = torch.utils.data.DataLoader(orig_svhn_tr_ds,
                                                            batch_size=bs, shuffle=True, num_workers=4)
        orig_svhn_test_dsloader = torch.utils.data.DataLoader(orig_svhn_test_ds,
                                                              batch_size=bs, shuffle=False, num_workers=4)
        # MNIST dataloaders
        orig_mnist_tr_ds = datasets.MNIST(mnist_root, train=True, download=True,
                                          transform=transforms.Compose([
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.1307,), (0.3081,))
                                          ]))
        orig_mnist_test_ds = datasets.MNIST(mnist_root, train=False,
                                            transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.1307,), (0.3081,))
                                            ]))
        orig_mnist_tr_dsloader = torch.utils.data.DataLoader(orig_mnist_tr_ds,
                                                             batch_size=bs, shuffle=True, num_workers=4)
        orig_mnist_test_dsloader = torch.utils.data.DataLoader(orig_mnist_test_ds,
                                                               batch_size=bs, shuffle=False, num_workers=4)

        # Iterate tasks
        for task_idx, task_targets in enumerate(self.targets):
            task_name = self.task_sequence[task_idx]
            print("*** TASK {} ***".format(task_name))
            print("task_targets={}".format(task_targets))

            # Training: Both MNIST+SVHN
            print("* TASK {}: training".format(task_name))
            tr_ds_path = os.path.join(seq_root,
                                      '{}_trainsubset_SubsetMNIST+SVHN_targets{}.pth'.format('tr', task_name))
            if not os.path.isfile(tr_ds_path) or overwrite or not self.save_imgfolders:
                tr_dataset = None
                for tr_dsloader in [orig_svhn_tr_dsloader, orig_mnist_tr_dsloader]:
                    tr_data, tr_targets = self._subset_data_on_labels(tr_dsloader, task_targets)
                    if tr_balanced_cnt is not None:
                        tr_data = tr_data[:tr_balanced_cnt]
                        tr_targets = tr_targets[:tr_balanced_cnt]
                    if tr_dataset is None:
                        tr_dataset = SubsetDataset(tr_data, tr_targets, task_targets)
                    else:
                        tr_dataset.append(tr_data, tr_targets)

                self.store_trset(tr_dataset, tr_ds_path)
            self.imgfolder_paths['train'][task_name] = tr_ds_path

            # Eval: Separate MNIST/SVHN
            # SVHN
            svhn_user = 1
            print("* TASK {}: SVHN user".format(task_name))
            user_ds_path = os.path.join(seq_root,
                                        '{}_testsubset_SubsetSVHN_targets{}.pth'.format('user', task_targets))
            if not os.path.isfile(tr_ds_path) or overwrite or not self.save_imgfolders:
                data, targets = self._subset_data_on_labels(orig_svhn_test_dsloader, task_targets)
                user_dataset = SubsetDataset(data, targets, task_targets)

                self.store_evalset_user(user_dataset, user_ds_path)

            self.imgfolder_paths['user'][svhn_user][task_name] = user_ds_path

            # MNIST
            mnist_user = 2
            print("* TASK {}: MNIST user".format(task_name))
            user_ds_path = os.path.join(seq_root,
                                        '{}_testsubset_SubsetMNIST_targets{}.pth'.format('user', task_targets))
            if not os.path.isfile(tr_ds_path) or overwrite or not self.save_imgfolders:
                data, targets = self._subset_data_on_labels(orig_mnist_test_dsloader, task_targets)
                user_dataset = SubsetDataset(data, targets, task_targets)

                self.store_evalset_user(user_dataset, user_ds_path)
            self.imgfolder_paths['user'][mnist_user][task_name] = user_ds_path

        print("Custom MNIST+SVHN preprocessing finished")

    def store_evalset_user(self, user_dataset, user_ds_path):
        """ Save eval set for a user."""
        eval_ds, iw_ds = self.split_ds(user_dataset, tr_ratio=0.5)  # Split & Format
        ds = {'eval': eval_ds, 'iw': iw_ds}

        if self.save_imgfolders:  # Store
            torch.save(ds, user_ds_path)
            print("Saved custom user dataset: {}".format(user_ds_path))
        else:  # In Memory
            self.imgfolders[user_ds_path] = ds
            print("In memory user dataset: {}".format(user_ds_path))

    def store_trset(self, tr_dataset, tr_ds_path):
        """ Save tr set."""
        train_ds, val_ds = self.split_ds(tr_dataset, tr_ratio=0.8)  # Split & Format
        ds = {'train': train_ds, 'val': val_ds}

        if self.save_imgfolders:  # Store
            torch.save(ds, tr_ds_path)
            print("Saved custom user dataset: {}".format(tr_ds_path))
        else:  # In Memory
            self.imgfolders[tr_ds_path] = ds
            print("In memory user dataset: {}".format(tr_ds_path))

    # SVHN
    def create_custom_SVHN(self, user_SVHN_targets, bs=1000, overwrite=False):
        """
        Creates custom Dataset objects from original cropped 32x32 SVHN.
        User categories are also included in the training set.
        User samples are not available for the server set: they are extracted from the original SVHN test set.
        Server samples comprise all the training data.
        :return:
        """

        # Init
        self.tr_SVHN_targets = [i for i in range(0, 10)]
        svhn_root = os.path.join(self.root_path, 'SVHN')
        cust_SVHN_root = os.path.join(svhn_root, 'split')
        utils.create_dir(cust_SVHN_root)

        user_ds_path = os.path.join(cust_SVHN_root,
                                    '{}_testsubset_SubsetSVHN_targets{}.pth'.format('user', user_SVHN_targets))
        tr_ds_path = os.path.join(cust_SVHN_root,
                                  '{}_fulltrain_SubsetSVHN.pth'.format('tr'))
        self.svhn_user_path = user_ds_path
        self.svhn_tr_path = tr_ds_path

        # Training Data
        if not os.path.isfile(tr_ds_path) or overwrite:
            # Original Data
            orig_tr_ds = datasets.SVHN(svhn_root, split='train', download=True,
                                       transform=transforms.Compose([
                                           transforms.ToPILImage(),
                                           transforms.CenterCrop(self.input_size[0]),
                                           transforms.ToTensor(),
                                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                       ]))
            tr_dataset = SubsetDataset(torch.tensor(orig_tr_ds.data),
                                       torch.tensor(orig_tr_ds.labels),
                                       self.tr_SVHN_targets,
                                       transform=orig_tr_ds.transform,
                                       target_transform=orig_tr_ds.target_transform)

            # Format
            train_ds, val_ds = self.split_ds(tr_dataset, tr_ratio=0.8)
            torch.save({'train': train_ds, 'val': val_ds}, tr_ds_path)
            print("Saved custom SVHN dataset: {}".format(tr_ds_path))

        # User Data
        if not os.path.isfile(user_ds_path) or overwrite:
            orig_test_ds = datasets.SVHN(svhn_root, split='test', download=True,
                                         transform=transforms.Compose([
                                             transforms.CenterCrop(self.input_size[0]),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                         ]))
            orig_eval_dsloader = torch.utils.data.DataLoader(
                orig_test_ds, batch_size=bs, shuffle=False, num_workers=4)
            print("Original SVHN testset has {} samples".format(len(orig_test_ds)))
            user_data, user_labels = self._subset_data_on_labels(orig_eval_dsloader, user_SVHN_targets)
            user_dataset = SubsetDataset(user_data, user_labels, user_SVHN_targets)

            # Split & Format
            eval_ds, iw_ds = self.split_ds(user_dataset, tr_ratio=0.5)
            torch.save({'eval': eval_ds, 'iw': iw_ds}, user_ds_path)
            print("Saved custom SVHN dataset: {}".format(user_ds_path))
        print("Custom SVHN preprocessing finished")

    def create_custom_MNIST_include_usertargets(self, user_MNIST_targets, bs=1000, overwrite=False):
        """
        Creates custom Dataset objects from original MNIST. User categories are also included in the training set.
        User samples are not available for the server set: they are extracted from the original MNIST test set.
        :param bs: batch size to iterate over original MNIST.
        :param exclude_targets: exclude the categories from the
        :return:
        """

        # Init
        self.tr_MNIST_targets = [i for i in range(0, 10)]
        mnist_root = os.path.join(self.root_path, 'MNIST')
        cust_mnist_root = os.path.join(mnist_root, 'split')
        utils.create_dir(cust_mnist_root)

        user_ds_path = os.path.join(cust_mnist_root,
                                    '{}_testsubset_SubsetMNIST_targets{}.pth'.format('user', user_MNIST_targets))
        tr_ds_path = os.path.join(cust_mnist_root,
                                  '{}_fulltrain_SubsetMNIST.pth'.format('tr'))
        self.mnist_user_path = user_ds_path
        self.mnist_tr_path = tr_ds_path

        # Training Data
        if not os.path.isfile(tr_ds_path) or overwrite:
            # Original Data
            orig_tr_ds = datasets.MNIST(mnist_root, train=True, download=True,
                                        transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.1307,), (0.3081,))
                                        ]))
            tr_dataset = SubsetDataset(orig_tr_ds.train_data, orig_tr_ds.train_labels, self.tr_MNIST_targets)

            # Format
            train_ds, val_ds = self.split_ds(tr_dataset, tr_ratio=0.8)
            torch.save({'train': train_ds, 'val': val_ds}, tr_ds_path)
            print("Saved custom MNIST dataset: {}".format(tr_ds_path))

        # User Data
        if not os.path.isfile(user_ds_path) or overwrite:
            orig_mnist_test = datasets.MNIST(mnist_root, train=False,
                                             transform=transforms.Compose([
                                                 transforms.ToTensor(),
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))
            orig_eval_dsloader = torch.utils.data.DataLoader(
                orig_mnist_test, batch_size=bs, shuffle=False, num_workers=4)
            print("Original MNIST testset has {} samples".format(len(orig_mnist_test)))

            # Select based on categories
            user_data, user_labels = self._subset_data_on_labels(orig_eval_dsloader, user_MNIST_targets)
            user_dataset = SubsetDataset(user_data, user_labels, user_MNIST_targets)

            # Split & Format
            eval_ds, iw_ds = self.split_ds(user_dataset, tr_ratio=0.5)
            torch.save({'eval': eval_ds, 'iw': iw_ds}, user_ds_path)
            print("Saved custom MNIST dataset: {}".format(user_ds_path))
        print("Custom MNIST preprocessing finished")

    def create_custom_MNIST_exclude_usertargets(self, user_MNIST_targets, bs=1000, overwrite=False):
        """
        Creates custom Dataset objects from original MNIST.
        :param bs: batch size to iterate over original MNIST.
        :param exclude_targets: exclude the categories from the
        :return:
        """

        # Init
        self.tr_MNIST_targets = [i for i in range(0, 10) if i not in user_MNIST_targets]
        mnist_root = os.path.join(self.root_path, 'MNIST')
        cust_mnist_root = os.path.join(mnist_root, 'split')
        utils.create_dir(cust_mnist_root)

        ds_template = '{}_SubsetMNIST_targets{}.pth'
        user_ds_path = os.path.join(cust_mnist_root, ds_template.format('user', user_MNIST_targets))
        tr_ds_path = os.path.join(cust_mnist_root, ds_template.format('tr', self.tr_MNIST_targets))
        self.mnist_user_path = user_ds_path
        self.mnist_tr_path = tr_ds_path

        if overwrite or (not os.path.isfile(user_ds_path) and not os.path.isfile(tr_ds_path)):
            # Original Data
            orig_tr_dsloader = torch.utils.data.DataLoader(
                datasets.MNIST(mnist_root, train=True, download=True,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])), batch_size=bs, shuffle=False, num_workers=4)
            orig_eval_dsloader = torch.utils.data.DataLoader(
                datasets.MNIST(mnist_root, train=False,
                               transform=transforms.Compose([
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.1307,), (0.3081,))
                               ])), batch_size=bs, shuffle=False, num_workers=4)

            # Select based on categories
            user_data = None
            user_labels = None
            tr_data = None
            tr_labels = None
            for dataloader in [orig_tr_dsloader, orig_eval_dsloader]:
                for batch_idx, (imgs, targets) in tqdm.tqdm(enumerate(dataloader)):
                    user_idx = None
                    for user_MNIST_target in user_MNIST_targets:
                        user_idx_single = targets.eq(user_MNIST_target)
                        user_idx = user_idx + user_idx_single if user_idx is not None else user_idx_single

                    # Add user data if any
                    if user_idx.any:
                        user_data = torch.cat((user_data, imgs[user_idx])) \
                            if user_data is not None else imgs[user_idx]
                        user_labels = torch.cat((user_labels, targets[user_idx])) \
                            if user_labels is not None else targets[user_idx]

                    # Add tr data
                    tr_idx = ~user_idx
                    tr_data = torch.cat((tr_data, imgs[tr_idx])) \
                        if tr_data is not None else imgs[tr_idx]
                    tr_labels = torch.cat((tr_labels, targets[tr_idx])) \
                        if tr_labels is not None else targets[tr_idx]
            print("Got userdata ({} samples), tr-data ({} samples)".format(user_data.shape, tr_data.shape))
            user_dataset = SubsetDataset(user_data, user_labels, user_MNIST_targets)
            tr_dataset = SubsetDataset(tr_data, tr_labels, self.tr_MNIST_targets)

            # Format
            eval_ds, iw_ds = self.split_ds(user_dataset, tr_ratio=0.5)
            train_ds, val_ds = self.split_ds(tr_dataset, tr_ratio=0.8)

            torch.save({'eval': eval_ds, 'iw': iw_ds}, user_ds_path)
            torch.save({'train': train_ds, 'val': val_ds}, tr_ds_path)
            print("Saved custom MNIST dataset: {}".format(user_ds_path))
            print("Saved custom MNIST dataset: {}".format(tr_ds_path))
        print("Custom MNIST preprocessing finished")

    # STATIC METHODS
    @staticmethod
    def _subset_data_on_labels(orig_dsloader, label_targets):
        """ Iterate through original dsloader and extract data/labels with labels in label_targets."""
        # Select based on categories
        data_subset = None
        label_subset = None
        for (imgs, targets) in orig_dsloader:
            user_idx = None
            for label_target in label_targets:
                user_idx_single = targets.eq(label_target)
                user_idx = user_idx + user_idx_single if user_idx is not None else user_idx_single

            # Add user data if any
            if user_idx.any:
                data_subset = torch.cat((data_subset, imgs[user_idx])) \
                    if data_subset is not None else imgs[user_idx]
                label_subset = torch.cat((label_subset, targets[user_idx])) \
                    if label_subset is not None else targets[user_idx]
        print("Selected userdata ({} samples)".format(data_subset.shape))

        return data_subset, label_subset

    @staticmethod
    def split_ds(ds, tr_ratio=0.8):
        train_size = int(tr_ratio * len(ds))
        val_size = len(ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
        # val_ds.dataset.transform = self.get_val_transform()
        return train_ds, val_ds

    # GETTERS
    def get_test_imagefolder(self, imglist, classes, test_transform):
        return ImagePathlist(imlist=imglist, classes=classes, transform=test_transform)

    def get_taskname(self, task_count):
        """e.g. Translation of 'Task 1' to the actual name of the first task."""
        if task_count < 1 or task_count > len(self.task_sequence):
            raise ValueError('[ERROR] Task count out of bounds: ', task_count)
        return self.task_sequence[task_count - 1]

    def get_imgfolder_path(self, task_name, train=True, user=None):
        """
        When user is specified, only return user data, regardless task_name, as user only has data for one task.

        :param task_name: str with task name, must be in self.task_sequence
        :param train: imgfolder for training (and validation)
        :param user: id number of user (from 1 up to and including length of taskseq)
        :return:
        """
        assert train == (user is None), "Can't get imgfolder both for trainmode and specific user. Specify only one."

        if 'ds_tasks' in self.mode:  # Each dataset is a separate task
            return self._get_ds_tasks_imgfolder_path(task_name, train, user)
        elif 'nb_tasks' in self.mode:  # MNIST+SVHN combined constitute
            return self._get_nb_tasks_imgfolder_path(task_name, train, user)

    def _get_nb_tasks_imgfolder_path(self, task_name, train, user):
        if user is not None:
            assert 1 <= user <= self.usercount, "User ID is invalid: {}".format(user)
            return self.imgfolder_paths['user'][user][task_name]

        if task_name is None:
            return None
        assert task_name in self.task_sequence, "Invalid task name: {}".format(task_name)
        return self.imgfolder_paths['train'][task_name]

    def _get_ds_tasks_imgfolder_path(self, task_name, train, user):
        if user is not None:
            assert 1 <= user <= len(self.task_sequence), "User ID is invalid: {}".format(user)
            user_idx = user - 1
            if user_idx == self.task_sequence.index("MNIST"):
                if train:
                    return self.mnist_tr_path
                else:
                    return self.mnist_user_path
            elif user_idx == self.task_sequence.index("SVHN"):
                if train:
                    return self.svhn_tr_path
                else:
                    return self.svhn_user_path
            else:
                raise NotImplementedError("USER IS NOT IMPLEMENTED: ", user)

        if task_name is None:
            return None
        assert task_name in self.task_sequence, "Invalid task name: {}".format(task_name)
        if task_name == "MNIST":
            if train:
                return self.mnist_tr_path
            else:
                return self.mnist_user_path
        elif task_name == "SVHN":
            if train:
                return self.svhn_tr_path
            else:
                return self.svhn_user_path
        else:
            raise NotImplementedError("TASK IS NOT IMPLEMENTED: ", task_name)


class SubsetDataset(data.Dataset):
    """Wrapper MNIST dataset, all data loaded in memory."""

    def __init__(self, data, labels, classes, root=None, transform=None, target_transform=None):
        """
        :param data: tensor of imgs, assumed already transformed to tensor and normalized.
        :param labels: tensor of labels
        :param classes: list of all different classes the labels can be from.
        :param transform: additional transforms on runtime applied.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.data = data
        self.labels = labels
        self.classes = sorted(classes)

        self.data = self._format_std_rgb(data)

        # Checks
        self.check_only_contains(self.labels, self.classes)
        assert len(self.data) == len(self.labels)
        print("Initialized custom Dataset of length {}".format(len(self.labels)))

        # Original label mapping to idxs
        self.class_to_idx = {label: idx for idx, label in enumerate(self.classes)}

    def append(self, new_data, new_labels):
        # Checks
        self.check_only_contains(new_labels, self.classes)
        assert len(new_data) == len(new_labels)

        new_data = self._format_std_rgb(new_data)

        # Append
        self.data = torch.cat((self.data, new_data), dim=0)
        self.labels = torch.cat((self.labels, new_labels), dim=0)
        print("Appended custom Dataset of length {}".format(len(self.labels)))

    @staticmethod
    def check_only_contains(inp_labels, labellist):
        """Check all labels only contain from a predefined list of categories."""
        labels_eq = torch.zeros_like(inp_labels).byte()
        for label in labellist:
            labels_eq += inp_labels.eq(label)
        assert labels_eq.all(), "{} labels not in: {}".format(torch.sum(~labels_eq), labellist)
        print("CHECK PASSED, only containing labels in ", labellist)

    @staticmethod
    def _format_std_rgb(data):
        """ Reshape to standard RGB format."""
        # MNIST binary data
        if isinstance(data, torch.ByteTensor):
            data = data.float()

        # Colour channels
        if len(data.shape) == 3:
            print("data shape={}".format(data.shape))
            data = data.unsqueeze(1)  # Add channel dim
            data = data.repeat(1, 3, 1, 1)
            print("Updated data shape={}".format(data.shape))
        elif len(data.shape) == 4 and data.shape[1] == 1:
            data = data.repeat(1, 3, 1, 1)
            print("Updated data shape={}".format(data.shape))
        elif not len(data.shape) == 4 and data.shape[1] == 3:
            raise Exception("Data shape will cause errors, need 3 RGB channels, shape = {}".format(data.shape))

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], torch.tensor(self.class_to_idx[self.labels[index].item()])

        # Check
        # plot.plot.imshow_tensor(img, title="Label = " + str(target))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target


class IndoorScenesDataset(CustomDataset):
    """ CatPrior dataset."""
    name = 'IndoorScenesMIT'
    classes_per_task = OrderedDict()
    input_size = (224, 224)
    task_sequence = ['home', 'leisure', 'public', 'store', 'work']
    imgfolders = {}

    def_task_count = 4  # Max 5 (length of task sequence)
    def_usercount = 5  # How many users
    def_user_cat_pref = 3  # Nb of task-specific categories to allocate extra samples to user
    def_user_cat_prefimgs = 70  # Nb of extra samples per preferred category

    task_count = None
    usercount = None
    user_cat_pref = None
    user_cat_prefimgs = None

    def __init__(self, seed=27, task_sequence=None, input_size=(224, 224), transforms=None, demo=True, usercount=None,
                 user_cat_pref=None, user_cat_prefimgs=None, imgfolder_suffix=None, task_count=None,
                 save_imgfolders=True, equal_userdata=False):
        """
        Randomness in:
            selecting preference categories
            selecting samples from a category
        :param task_sequence: List of ordered supercategories
        :param input_size: img input size
        :param transforms: Dict with key 'train'/'val'/'test' and value a list of transforms (1 per task)
        :param equal_userdata: All users get same data allocated, (e.g. with each user a different transform).
        """
        print("\n===> INIT DATASET {} <===".format(self.name))

        utils.set_random(seed=seed)
        config = utils.get_parsed_config()
        self.init_dsparams(usercount, user_cat_pref, user_cat_prefimgs, task_count)
        self.imgfolders = {}  # <path>,<imgfolder> pairs, keeping random in mem (no unique ids for parallel jobs required)
        self.equal_userdata = equal_userdata

        self.root_path = os.path.join(utils.read_from_config(config, 'ds_root_path'), "MIT_indoor_scenes", 'Tasks')
        self.input_size = input_size
        self.save_imgfolders = save_imgfolders

        self.extra_img = 'Images'
        self.tr_img = 'TrainImages'
        self.test_img = 'TestImages'

        self.task_sequence = self.task_sequence if task_sequence is None else task_sequence
        self.cat_per_task = {task: [] for task in self.task_sequence}
        self.cat_imgs_cnt = {}  # Nb extra imgs available per category
        self.user_cat_imgs_cnt = {user: {} for user in range(1, self.usercount + 1)}  # Total user imgs count per cat

        # Imgfolders
        tr_imgfolder_parts = ['tr_imgfolder']
        if imgfolder_suffix is not None:
            tr_imgfolder_parts.append(imgfolder_suffix)
        tr_imgfolder_parts.append('{}.pth'.format(self.name))
        self.tr_imgfolder = '_'.join(tr_imgfolder_parts)

        test_imgfolder_parts = ['user{}_imgfolder']
        if imgfolder_suffix is not None:
            test_imgfolder_parts.append(imgfolder_suffix)
        test_imgfolder_parts.append('{}.pth'.format(self.name))
        self.test_imgfolder = '_'.join(test_imgfolder_parts)

        # Init user prefs
        restore_path = os.path.join(utils.get_root_src_path(), 'data', 'user_prefs.pth')
        if os.path.exists(restore_path):
            self.user_prefs = torch.load(restore_path)
            reuse_prefs = True
        else:
            self.user_prefs = {}
            reuse_prefs = False
            for user in range(1, self.usercount + 1):
                self.user_prefs[user] = {}
                for task in self.task_sequence:
                    self.user_prefs[user][task] = []

        # Init transforms
        self.transforms = self.init_transforms() if transforms is None else transforms

        # Divide the data over the users in imagefolders
        self.eval_imgpaths, self.iw_imgpaths = None, None
        self.init_user_paths(reuse_prefs=reuse_prefs, equal_userdata=self.equal_userdata)  # Divide paths over users
        self.init_imagefolders(self.transforms, save=save_imgfolders)  # Save total imgfolders for users

        self.analyze_user_cat_counts()

        # View the data
        if demo:
            # self.show_userimges()
            self.analyze_extra_data_counts()

    def analyze_extra_data_counts(self):
        """Print per task how many extra imgs we have for all categories."""
        # Analyze
        import operator
        for task, cats in self.cat_per_task.items():
            print("\nTASK {}: extra imgs ".format(task))
            task_img_cnt_dict = {}
            for cat in cats:
                task_img_cnt_dict[cat] = self.cat_imgs_cnt[cat]
            sorted_task_img_cnt = sorted(task_img_cnt_dict.items(), key=operator.itemgetter(1))
            sorted_task_img_cnt.reverse()

            for (cat, cat_imgs_cnt) in sorted_task_img_cnt:
                print("{} = {}".format(cat, cat_imgs_cnt))

    def analyze_user_cat_counts(self, mode='eval'):
        print("*** Summary Category Counts Per User: {} set ***".format(mode))
        paths = self.eval_imgpaths if mode == 'eval' else self.iw_imgpaths

        for user in range(1, self.usercount + 1):
            for task, task_paths in paths[user].items():
                cat_counts = {}
                for imgpath in task_paths:
                    img_cat = os.path.basename(os.path.dirname(imgpath))
                    if img_cat not in cat_counts:
                        cat_counts[img_cat] = 0
                    cat_counts[img_cat] += 1
                print("USER {}, TASK {}: cat counts(total={}) ={}, prefs={}".format(
                    user, task, len(task_paths), cat_counts, self.user_prefs[user][task]))

    def init_dsparams(self, usercount=None, user_cat_pref=None, user_cat_prefimgs=None, task_count=None,
                      legacy_mode=True):
        """ Params that define the dataset constitution."""
        ext_name = []
        if task_count is None:
            task_count = self.def_task_count
        if self.task_count is None:  # If not intialized yet
            assert 1 <= task_count <= len(self.task_sequence)
            self.task_count = task_count
            self.task_sequence = self.task_sequence[:task_count]
            ext_name.append("tc=" + str(task_count))

        if usercount is None:
            usercount = self.def_usercount
        if self.usercount is None:  # If not intialized yet
            self.usercount = usercount
            ext_name.append("uc=" + str(usercount))

        if user_cat_pref is None:
            user_cat_pref = self.def_user_cat_pref
        if self.user_cat_pref is None:  # If not intialized yet
            self.user_cat_pref = user_cat_pref
            ext_name.append("upref=" + str(user_cat_pref))

            if self.user_cat_pref == 0:
                print("[DS-MODE] No priors, user_cat_prefimgs selects images over all categories in task.")

        if user_cat_prefimgs is None:
            user_cat_prefimgs = self.def_user_cat_prefimgs
        if self.user_cat_prefimgs is None:  # If not intialized yet
            self.user_cat_prefimgs = user_cat_prefimgs
            ext_name.append("prefc=" + str(user_cat_prefimgs))

        ext_name = [] if legacy_mode else ext_name  # Legacy exps not using extended names
        self.name = '_'.join([self.name] + ext_name)

    # IMGFOLDER PATHS
    def get_imgfolder_path(self, task, train=True, user=None):
        if task is None:
            return None
        if train:
            return os.path.join(self.root_path, self.tr_img, task, self.tr_imgfolder)
        else:
            assert user is not None, "Define user id for test imgfolders"
            return os.path.join(self.root_path, self.test_img, task, self.test_imgfolder.format(user))

    # TRANSFORMS
    def init_transforms(self):
        """Creates default train,val,user-test Transform sequences, 1 per task."""
        trainval_transforms = [self.get_train_transform()] * len(self.task_sequence)
        user_transforms = [self.get_test_transform()] * len(self.task_sequence)  # applies both on eval/iw sets
        test_transforms = {user: user_transforms for user in range(1, self.usercount + 1)}
        return {'trainval': trainval_transforms, 'test': test_transforms}

    # IMAGEFOLDERS
    def init_imagefolders(self, transforms, overwrite=True, save=False):
        """ Creates train,val,user-test Imagefolders when not existing"""
        print("INIT IMAGEFOLDERS")
        self.imgfolders = {}
        for task_idx, task in enumerate(self.task_sequence):
            tr_path = os.path.join(self.root_path, self.tr_img, task, self.tr_imgfolder)
            train_ds, val_ds = self.get_trainval_imagefolders(task, transforms['trainval'][task_idx])

            out = {'train': train_ds, 'val': val_ds}
            self.imgfolders[tr_path] = out
            if save and (not os.path.exists(tr_path) or overwrite):
                torch.save(out, tr_path)
            print("Created train(size {})/val (size {}) imgfolders for task '{}'"
                  .format(len(train_ds), len(val_ds), task))

            test_path = os.path.join(self.root_path, self.test_img, task, self.test_imgfolder)
            for user in range(1, self.usercount + 1):  # divide imgpaths_user into 2 sets
                # Cleanup existing
                user_transform = transforms['test'][user][task_idx]
                user_eval_ds = self.get_test_imagefolder(
                    self.eval_imgpaths[user][task], self.cat_per_task[task], user_transform)
                user_iw_ds = self.get_test_imagefolder(
                    self.iw_imgpaths[user][task], self.cat_per_task[task], user_transform)

                out = {'eval': user_eval_ds, 'iw': user_iw_ds}
                self.imgfolders[test_path.format(user)] = out
                if save and (not os.path.exists(test_path.format(self.usercount)) or overwrite):
                    torch.save(out, test_path.format(user))
                print("Created eval(size {})/IW (size {}) imgfolders for task '{}', path={}"
                      .format(len(user_eval_ds), len(user_iw_ds), task, test_path.format(user)))
        print("initiated imgfolders")

    def get_test_path(self, task, user=None):
        test_path = os.path.join(self.root_path, self.test_img, task, self.test_imgfolder)
        if user is not None:
            test_path.format(user)
        return test_path

    def get_trainval_imagefolders(self, task, trainval_transform, tr_ratio=0.8):
        full_ds = datasets.ImageFolder(transform=trainval_transform,
                                       root=os.path.join(self.root_path, self.tr_img, task))

        train_size = int(tr_ratio * len(full_ds))
        val_size = len(full_ds) - train_size
        train_ds, val_ds = torch.utils.data.random_split(full_ds, [train_size, val_size])
        val_ds.dataset.transform = self.get_val_transform()

        return train_ds, val_ds

    def get_test_imagefolder(self, imglist, classes, test_transform):
        return ImagePathlist(imlist=imglist, classes=classes, transform=test_transform)

    def _assign_test_data(self, debug=False, equal_userdata=False):
        """ Divide test data with 20 imgs/cat equally over all users."""
        print("Dividing original test data over users into actual eval and IW data.")
        eval_test_imgpaths = {user: {} for user in range(1, self.usercount + 1)}
        iw_test_imgpaths = {user: {} for user in range(1, self.usercount + 1)}
        for user in range(1, self.usercount + 1):
            for task in self.task_sequence:
                eval_test_imgpaths[user][task] = []
                iw_test_imgpaths[user][task] = []

        add_cats = len(self.cat_per_task[task]) == 0
        for task in self.task_sequence:
            if debug:
                print("SUPERCAT TASK {}".format(task))
            test_path = os.path.join(self.root_path, self.test_img, task)
            for category in os.listdir(test_path):
                if not os.path.isfile(os.path.join(test_path, category)):
                    if add_cats:
                        self.cat_per_task[task].append(category)
                    self.user_cat_imgs_cnt[category] = {user: 0 for user in range(1, self.usercount + 1)}
                    cat_path = os.path.join(test_path, category)

                    # Collect all imgpaths in category
                    cat_imgpaths = []
                    for img in os.listdir(cat_path):
                        cat_imgpaths.append(os.path.join(cat_path, img))
                    if debug:
                        print("CAT {}: {}".format(category, len(cat_imgpaths)))

                    # Img counts for each user
                    if equal_userdata:  # Assign all to each user
                        users = [u for u in range(1, self.usercount + 1)]  # One to copy
                        eval_cnt = int(len(cat_imgpaths) / 2)
                        for user in users:
                            eval_test_imgpaths[user][task].extend(cat_imgpaths[:eval_cnt])
                            iw_test_imgpaths[user][task].extend(cat_imgpaths[eval_cnt:])

                    else:  # Assign exclusive subset to each user
                        random.shuffle(cat_imgpaths)  # Introduce randomness

                        fixed_userimgs = int(len(cat_imgpaths) / self.usercount)
                        rnd_userimgs = len(cat_imgpaths) % self.usercount
                        rnd_users = [x + 1 for x in np.random.permutation(self.usercount)]
                        usercounts = {user: fixed_userimgs for user in range(1, self.usercount + 1)}
                        for rnd_user in rnd_users[:rnd_userimgs]:
                            usercounts[rnd_user] += 1

                        # Assign paths to both IW and eval sets
                        imgpath_cnt = 0
                        for user in rnd_users:
                            usercount = usercounts[user]
                            selected_paths = cat_imgpaths[imgpath_cnt:imgpath_cnt + usercount]
                            eval_cnt = int(len(selected_paths) / 2)
                            eval_test_imgpaths[user][task].extend(selected_paths[:eval_cnt])
                            iw_test_imgpaths[user][task].extend(selected_paths[eval_cnt:])
                            imgpath_cnt += usercount
            sorted(self.cat_per_task[task])
            if debug:
                for user in range(1, self.usercount + 1):
                    print("USER {}: Eval samples = {}, IW samples = {}".format(
                        user, len(eval_test_imgpaths[user][task]), len(iw_test_imgpaths[user][task])))
        return eval_test_imgpaths, iw_test_imgpaths

    def _assign_extra_data(self, debug=False, reuse_prefs=False, equal_userdata=False):
        """ Assign extra data for user preferenced categories per task.
            self.user_cat_pref == 0, implies no pref categories, but select from all in task.
        """
        # Init
        eval_extra_imgpaths = {user: {} for user in range(1, self.usercount + 1)}
        iw_extra_imgpaths = {user: {} for user in range(1, self.usercount + 1)}
        for user in range(1, self.usercount + 1):
            for task in self.task_sequence:
                eval_extra_imgpaths[user][task] = []
                iw_extra_imgpaths[user][task] = []

        print("Appending {} categories per task per user".format(self.user_cat_pref))
        self.cat_imgs_cnt = {}  # init
        for task in self.task_sequence:
            if debug:
                print("SUPERCAT TASK {}".format(task))
            test_path = os.path.join(self.root_path, self.extra_img, task)
            cat_users_cnt = OrderedDict({cat: 0 for cat in self.cat_per_task[task]})
            cat_imgs_cnt = OrderedDict({cat: len([img for img in os.listdir(os.path.join(test_path, cat))])
                                        for cat in cat_users_cnt.keys()})
            self.cat_imgs_cnt = {**self.cat_imgs_cnt, **cat_imgs_cnt}

            # COPY OR EXCLUSIVE
            if equal_userdata:  # Select one user, and copy to other users
                users = [1]  # Ref user

            else:  # All users have exclusive data
                users = [u for u in range(1, self.usercount + 1)]

            print("TASK {}: EXLUSIVE DATA FOR {} users".format(task, len(users)))
            # MODE 1: Select user subsets over all categories, each exclusive
            if self.user_cat_pref == 0:

                # Collect all imgpaths over all categories
                all_imgs_free_paths = []  # All paths in one list
                for cat in cat_users_cnt.keys():
                    cat_path = os.path.join(test_path, cat)
                    for img in os.listdir(cat_path):
                        path = os.path.join(cat_path, img)
                        all_imgs_free_paths.append(path)

                # Allocate to users
                for user in users:
                    all_imgs_free_paths = self._allocate_extra_free_paths(all_imgs_free_paths, user, task,
                                                                          eval_extra_imgpaths, iw_extra_imgpaths)

            # MODE 2: Add for each user the preferences and corresponding imgs
            else:
                # Collect all imgpaths per category
                cat_imgs_free_paths = OrderedDict()  # Paths per category
                for cat in cat_users_cnt.keys():
                    cat_imgs_free_paths[cat] = []
                    cat_path = os.path.join(test_path, cat)
                    for img in os.listdir(cat_path):
                        path = os.path.join(cat_path, img)
                        cat_imgs_free_paths[cat].append(path)

                # To fit in more data for last task (only for 5 task sequence)
                force_cnt = 0
                min_force_cnt = force_cnt
                if task == 'work' and self.user_cat_prefimgs >= 30:
                    force_cat = 'warehouse'  # Cat with most data has to be used by most of the users
                    min_force_cnt = 9

                # Allocate to users
                for user in users:
                    rnd_cnt = 0
                    user_force_cnt = 0
                    limit = 1 if self.user_cat_pref == 0 else self.user_cat_pref  # Only 1 to select from all merged cats
                    cnt = 0
                    while rnd_cnt < limit:  # Add new preference and corresponding imgs
                        cnt += 1
                        if cnt % 100 == 0:
                            print("Random allocation failed, can't find combination satisfying constraints.")

                        # Preference constraints
                        if reuse_prefs:  # Reuse existing preferences (e.g. reallocate for multiple iterations)
                            cat = self.user_prefs[user][task][rnd_cnt]
                        else:  # New random pref
                            if force_cnt < min_force_cnt and user_force_cnt == 0:  # Force users to take a task
                                idx = self.cat_per_task[task].index(force_cat)
                                user_force_cnt += 1
                            else:
                                idx = random.randint(0, len(self.cat_per_task[task]) - 1)
                            cat = list(cat_users_cnt.keys())[idx]

                        # Don't exceed available nb category imgs
                        constr_len = len(cat_imgs_free_paths[cat]) >= self.user_cat_prefimgs
                        # no duplicate categories
                        constr_dupl = cat not in self.user_prefs[user][task] or reuse_prefs
                        # no duplicate inter-user prefs
                        constr_userdupl = not (len(self.user_prefs[user][task]) == self.user_cat_pref
                                               and sum(
                                    [set(self.user_prefs[user][task]) != set(self.user_prefs[other_user][task])
                                     for other_user in range(1, self.usercount + 1) if user != other_user]) == 0
                                               )

                        if constr_len and constr_dupl and constr_userdupl:
                            if not reuse_prefs:
                                self.user_prefs[user][task].append(cat)  # Save prefs

                            # To select from
                            cat_imgs_free_paths[cat] = self._allocate_extra_free_paths(
                                cat_imgs_free_paths[cat], user, task, eval_extra_imgpaths, iw_extra_imgpaths
                            )

                            # Update counters
                            cat_users_cnt[cat] += 1
                            rnd_cnt += 1
                    if debug:
                        print("cat_users_cnt={}".format(cat_users_cnt))

            if equal_userdata:  # Copy to other users
                assert len(users) == 1, "More than 1 user as reference to copy..."
                ref_user = users[0]
                other_users = [u for u in range(1, self.usercount + 1) if u != ref_user]
                for user in other_users:
                    eval_extra_imgpaths[user][task].extend(eval_extra_imgpaths[ref_user][task])
                    iw_extra_imgpaths[user][task].extend(iw_extra_imgpaths[ref_user][task])

        return eval_extra_imgpaths, iw_extra_imgpaths

    def _allocate_extra_free_paths(self, free_paths, user, task, eval_extra_imgpaths, iw_extra_imgpaths):
        assert len(free_paths) >= self.user_cat_prefimgs, \
            "Not enough images, lower user_cat_prefimgs! Requested {}, got {} for task {}" \
                .format(self.user_cat_prefimgs, len(free_paths), task)

        # Generate indexes for the free paths
        idxs = np.random.permutation(len(free_paths) - 1)
        selected_idx = idxs[:self.user_cat_prefimgs]
        free_idx = idxs[self.user_cat_prefimgs:]

        # Update free paths and Imgs selected for user
        selected_paths = [free_paths[idx] for idx in selected_idx]  # Extract
        free_paths = [free_paths[idx] for idx in free_idx]  # Update

        # Add to users imgs
        eval_cnt = int(self.user_cat_prefimgs / 2)
        eval_extra_imgpaths[user][task].extend(selected_paths[:eval_cnt])
        iw_extra_imgpaths[user][task].extend(selected_paths[eval_cnt:])

        return free_paths

    def reassign_user_paths(self, seed):
        """Reassign all user data, while retaining the preferenced categories."""
        # New seed
        utils.set_random(seed=seed)

        # Check categories already assigned
        if self.user_cat_pref != 0:
            for user in range(1, self.usercount + 1):
                for task in self.task_sequence:
                    assert len(self.user_prefs[user][task]) == self.user_cat_pref

        # Reassign
        self.init_user_paths(reuse_prefs=True, equal_userdata=self.equal_userdata)  # Redivide userdata
        self.init_imagefolders(self.transforms, save=self.save_imgfolders)  # Store imgfolders with the paths

        self.analyze_user_cat_counts()

    def init_user_paths(self, debug=False, reuse_prefs=False, equal_userdata=False):
        """ Create list with img paths for each user, both for eval and importance weight estimation."""

        # Assign imgs
        print("\n*** ASSIGNING TEST DATA ***")
        eval_test_imgpaths, iw_test_imgpaths = self._assign_test_data(debug, equal_userdata=equal_userdata)

        print("\n*** ASSIGNING EXTRA DATA ***")
        eval_extra_imgpaths, iw_extra_imgpaths = self._assign_extra_data(debug, reuse_prefs=reuse_prefs,
                                                                         equal_userdata=equal_userdata)

        # MERGE
        self.eval_imgpaths = {user: {} for user in range(1, self.usercount + 1)}
        self.iw_imgpaths = {user: {} for user in range(1, self.usercount + 1)}
        for user in range(1, self.usercount + 1):
            for task in self.task_sequence:
                self.eval_imgpaths[user][task] = eval_test_imgpaths[user][task] + eval_extra_imgpaths[user][task]
                self.iw_imgpaths[user][task] = iw_test_imgpaths[user][task] + iw_extra_imgpaths[user][task]

        self._print_users_summary()

    def _print_users_summary(self):
        for user in range(1, self.usercount + 1):
            print("USER {}: SUMMARY".format(user))
            for task in self.task_sequence:
                print(" => Task {}: Eval samples = {}, IW samples = {}, prefs = {}"
                      .format(task,
                              len(self.eval_imgpaths[user][task]),
                              len(self.iw_imgpaths[user][task]),
                              self.user_prefs[user][task]))

    # TRANSFORMS
    def get_train_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(self.input_size[0]),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def get_val_transform(self):
        return self.get_test_transform()

    def get_test_transform(self):
        return transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(self.input_size[0]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    # UTILS
    def get_taskname(self, task_count):
        """e.g. Translation of 'Task 1' to the actual name of the first task."""
        if task_count < 1 or task_count > len(self.task_sequence):
            raise ValueError('[ERROR] Task count out of bounds: ', task_count)
        return self.task_sequence[task_count - 1]


class IndoorScenesTransformDataset(IndoorScenesDataset):
    """ TransPrior dataset. """
    name = 'IndoorScenesMIT_usertransform'

    # Params
    severeness = 3
    prob = 1

    transform_pool = [
        shared_utils.transforms.DeformTransform('spatter', severeness),
        shared_utils.transforms.DeformTransform('elastic', severeness),
        shared_utils.transforms.ColorTransform('sat', severeness),
        shared_utils.transforms.BlurTransform('defocus', severeness),
        shared_utils.transforms.NoiseTransform('gaus', severeness),
        shared_utils.transforms.ColorTransform('bright', severeness),
        shared_utils.transforms.BlurTransform('gaus', severeness),
        shared_utils.transforms.NoiseTransform('jpeg', severeness),
        shared_utils.transforms.ColorTransform('contrast', severeness),
        shared_utils.transforms.NoiseTransform('impulse', severeness),
    ]

    def_task_count = 4  # Max 5 (length of task sequence)
    def_usercount = 10  # How many users
    def_user_cat_pref = 0  # Force categories to zero: to disentangle transform/preference priors
    def_user_cat_prefimgs = 1000  # Nb of extra samples over all categoryies in a task
    equal_userdata = True  # Reduce variance only to transforms between users

    def __init__(self, demo=False, usercount=None, user_cat_prefimgs=None, task_count=None,
                 save_imgfolders=False, severeness=None):
        assert self.def_user_cat_pref == 0, "When using transforms, we use no prior on preferred classes."
        usercount = self.def_usercount if usercount is None else usercount
        user_cat_prefimgs = self.def_user_cat_prefimgs if user_cat_prefimgs is None else user_cat_prefimgs
        task_count = self.def_task_count if task_count is None else task_count
        self.severeness = self.severeness if severeness is None else self.severeness

        self.transform_pool = [transforms.RandomApply([transf], p=self.prob) for transf in self.transform_pool]

        super().init_dsparams(usercount, self.def_user_cat_pref, user_cat_prefimgs, task_count)

        id = "transform_sev={},p={},eu={}".format(self.severeness, self.prob, self.equal_userdata)
        init_transforms = self.init_transforms()
        super().__init__(imgfolder_suffix=id,
                         transforms=init_transforms,
                         usercount=usercount,
                         user_cat_pref=self.def_user_cat_pref,
                         user_cat_prefimgs=user_cat_prefimgs,
                         task_count=task_count,
                         save_imgfolders=save_imgfolders,
                         equal_userdata=self.equal_userdata)

        if demo:
            self.showcase_transforms()

    # TRANSFORMS
    def init_transforms(self):
        """Creates default train,val,user-test Transform sequences, 1 per task."""
        trainval_transforms = [self.get_train_transform()] * len(self.task_sequence)

        test_transforms = {user: [transforms.Compose([
            self.transform_pool[user - 1],
            transforms.CenterCrop(self.input_size[0]),  # For rotations
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
        )] * len(self.task_sequence)
                           for user in range(1, self.usercount + 1)}

        return {'trainval': trainval_transforms, 'test': test_transforms}

    def showcase_transforms(self, ex_per_user=3):
        """ Check how the transforms look on actual dataset. """
        import plot.plot as plot

        for user in range(1, self.usercount + 1):
            pics = {}
            print("PLOTTING USER {}".format(user))
            imgfolder_path = self.get_imgfolder_path(self.task_sequence[0], train=False, user=user)
            dsets = torch.load(imgfolder_path)
            dset_loader = torch.utils.data.DataLoader(dsets['eval'], batch_size=1, shuffle=False, num_workers=0)
            for idx, data in enumerate(dset_loader):
                if idx == ex_per_user:
                    break
                image, label = data
                image = image.cuda().squeeze()
                image = image.squeeze()
                img = plot.tensor_to_img(image)
                pics["Img {}".format(idx)] = img

            plot.plot_figures(pics, nrows=1, ncols=ex_per_user, title="USER {}".format(user))
