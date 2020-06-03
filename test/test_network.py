import pdb
import numpy as np

import torch
from torch.autograd import Variable
import collections
from train.LwF.ModelWrapperLwF import ModelWrapperLwF


def get_head_outputs(images, model, heads=None, current_head_idx=None, final_layer_idx=None):
    """Get output given the head."""
    if heads is not None:
        head = heads[current_head_idx]
        model.classifier._modules[final_layer_idx] = head  # Change head
        model.eval()
        outputs = model(Variable(images))
        return outputs
    elif isinstance(model, ModelWrapperLwF):
        outputs = model(Variable(images))
        return outputs[current_head_idx].data
    else:
        raise Exception("SHOULD SPECIFY HEADS TO USE")


def get_dataloader(imgfolders, batch_size):
    return {mode: torch.utils.data.DataLoader(imgfolders[mode], batch_size, shuffle=True, num_workers=4)
            for mode in ['eval', 'iw']}


def test_model(model, ds, target_task_head_idx, ds_classes, target_head=None, batch_size=200, final_layer_idx=None):
    """

    :param method:
    :param model:
    :param ds: Imagefolder/Dataset object; or path to dataset (str)
    :param target_task_head_idx: for EBLL,LWF which have all heads in model itself
    :param target_head: Actual head in list, so idx should be 0
    :param batch_size:
    :param subset:
    :param per_class_stats:
    :param final_layer_idx:
    :return:
    """
    if target_head is not None:
        if not isinstance(target_head, list):
            target_head = [target_head]
        assert target_task_head_idx == 0, "Only EBLL, LWF have heads in model itself, here head idx indicates target_headlist idx"

    # Init current model
    if hasattr(model, 'classifier') and final_layer_idx is None:
        final_layer_idx = str(len(model.classifier._modules) - 1)
    model.eval()
    model = model.cuda()

    # Init dataset
    if isinstance(ds, str):
        imgfolders = torch.load(ds)
    else:
        imgfolders = ds
    ds_loader = get_dataloader(imgfolders, batch_size)['eval']

    # Init stat counters
    class_correct = [0. for _ in range(len(ds_classes))]
    class_total = [0. for _ in range(len(ds_classes))]
    batch_count = 0

    # Iterate data
    for data in ds_loader:
        batch_count += 1
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        outputs = get_head_outputs(images, model, target_head, target_task_head_idx, final_layer_idx)
        _, target_head_pred = torch.max(outputs.data, 1)
        c = (target_head_pred == labels).squeeze()

        # Class specific stats
        for i in range(len(target_head_pred)):
            gt = labels[i].item()
            correct_cnt = c.item() if len(c.shape) == 0 else c[i].item()
            class_total[gt] += 1
            class_correct[gt] += correct_cnt

        del images, labels, outputs, data

    # OVERALL ACC
    accuracy = np.sum(class_correct) * 100 / (np.sum(class_total))
    print('Overall Accuracy: ' + str(accuracy))

    return accuracy


def test_task_joint_model(model, ds, task_idx, cumulative_classes_len, ds_classes, batch_size=200):
    """
    1 model to test them all.
    Test the performance of a given task in a model that is trained jointly on a set of tasks.
    Shared output layer, but masks out other task outputs.

    :param task_idx: the tested task ordered idx in the task lengths
    :param task_lengths: number of classes in each task
    :param batch_size:
    :param tasks_idxes: array of lists, with each list a set of integers that correspond to the FC outputs for this task_idx
    :return:
    """
    # Init current model
    model.eval()
    model = model.cuda()

    # Init dataset
    if isinstance(ds, str):
        imgfolders = torch.load(ds)
    else:
        imgfolders = ds
    ds_loader = get_dataloader(imgfolders, batch_size)['eval']

    # Init Task mask
    cumulative_classes_len = [0] + cumulative_classes_len
    this_task_class_mask = torch.tensor(
        list(range(cumulative_classes_len[task_idx], cumulative_classes_len[task_idx + 1])))

    # Init stat counters
    class_correct = [0. for _ in range(len(ds_classes))]
    class_total = [0. for _ in range(len(ds_classes))]
    batch_count = 0

    # Iterate data
    for data in ds_loader:
        batch_count += 1
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(Variable(images))

        # Filter out the ouput of other tasks
        this_tasks_outputs = outputs.data[:, this_task_class_mask]

        # Predict Max
        _, predicted = torch.max(this_tasks_outputs.data, 1)
        c = (predicted == labels).squeeze()

        # Class specific stats
        for i in range(len(predicted)):
            gt = labels[i].item()
            correct_cnt = c.item() if len(c.shape) == 0 else c[i].item()
            class_total[gt] += 1
            class_correct[gt] += correct_cnt

        del images, labels, outputs, data

    # OVERALL ACC
    accuracy = np.sum(class_correct) * 100 / (np.sum(class_total))
    print('Overall Accuracy: ' + str(accuracy))

    return accuracy
