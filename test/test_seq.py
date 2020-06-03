"""
Testing for a sequence of tasks.
Each task is tested on the trained model when it was newly added and all subsequently trained models.
"""
import traceback
import copy

from shared_utils.methods import Joint
from test.test_network import *
import shared_utils.utils as utils

import shared_utils.methods as methods
import torch.nn as nn
import train


class EvalMetrics(object):
    def __init__(self):
        self.seq_acc = []
        self.seq_forgetting = []
        self.seq_head_acc = []


def get_prev_heads(prev_head_model_paths, head_layer_idx):
    """
    :param prev_head_model_paths:   Previous Models to extract head from.
    """
    if not isinstance(prev_head_model_paths, list):
        prev_head_model_paths = [prev_head_model_paths]

    if len(prev_head_model_paths) == 0:
        return []

    heads = []
    # Add prev model heads
    for head_model_path in prev_head_model_paths:
        if isinstance(head_model_path, str):
            previous_model_ft = torch.load(head_model_path)
        else:
            previous_model_ft = head_model_path

        if isinstance(previous_model_ft, dict):
            previous_model_ft = previous_model_ft['model']

        head = previous_model_ft.classifier._modules[head_layer_idx]
        assert isinstance(head, torch.nn.Linear) or isinstance(head, torch.nn.Conv2d), type(head)
        heads.append(copy.deepcopy(head.cuda()))
        del previous_model_ft

    return heads


def eval_task_modelseq(models_path, datasets, task_save_idx, test_task_ds_idx, method, ds_classes, batch_size=200,
                       BN_model_paths=None, debug=False, task_lengths=None):
    """
    Performance on the ref task dataset (dataset_index), for all models that trained on this set (model_idx >= dataset_index)

    :param task_save_idx: the task number for which the results are stored
    :param test_task_ds_idx:   (task# - 1), used to find corresponding reference task dataset/model
    :param models_path:     list with sequence of model paths that used this task for training ( 'task_dataset_idx + 1')
    :param datasets:   list of paths to the dataset imgfolders; or dataset object
    :param batch_size:
    :return:
    """
    seq_acc = {}
    seq_forgetting = {}
    seq_acc[task_save_idx] = []
    seq_forgetting[task_save_idx] = []
    head_accuracy = None
    seq_head_acc = []

    if method.name == Joint.name and task_lengths is None:
        raise Exception()

    print("TESTING ON TASK ", test_task_ds_idx + 1)

    prev_head_model_paths = models_path[test_task_ds_idx]  # Model of ref task
    dataset_path = datasets[test_task_ds_idx]  # Data of ref task

    BN_stat_model = None
    if BN_model_paths is not None:
        if isinstance(BN_model_paths[test_task_ds_idx], str):
            BN_stat_model = torch.load(BN_model_paths[test_task_ds_idx])
        else:
            BN_stat_model = BN_model_paths[test_task_ds_idx]

        print("EXTRACTING BN STATS from {}".format(BN_model_paths[test_task_ds_idx]))

    if debug:
        print("Testing Dataset = ", dataset_path)

    # Iterate Models
    for trained_model_idx in range(test_task_ds_idx, len(datasets)):
        model_under_test_path = models_path[trained_model_idx]  # Model for current task

        print('=> Testing model trained up to and including TASK ', str(trained_model_idx + 1))

        # Heads from which models are needed
        n_heads = 1
        if debug:
            print("Testing on model = ", model_under_test_path)
        try:
            if trained_model_idx > 0 and isinstance(method, methods.LWF):  # LWF stacked heads
                if isinstance(model_under_test_path, str):
                    model = torch.load(model_under_test_path)
                else:
                    model = copy.deepcopy(model_under_test_path)
                target_head_idx = test_task_ds_idx
                print("EVAL on prev head idx: ", target_head_idx)

                # Patch
                model = LWF_patch(model)
                BN_stat_model = LWF_patch(BN_stat_model)

                # BN
                if BN_stat_model is not None:
                    replace_BN_params(BN_stat_model, model)

                accuracy = test_model(model, dataset_path, target_head_idx, ds_classes, batch_size=batch_size)

            else:  # Other methods
                if isinstance(model_under_test_path, str):
                    model = torch.load(model_under_test_path)
                else:
                    model = copy.deepcopy(model_under_test_path)
                if isinstance(model, dict):
                    model = model['model']

                head_layer_idx, _ = utils.get_last_neural_layer(model)  # Last head layer of prev model
                head_layer_idx = head_layer_idx if head_layer_idx in model.classifier._modules \
                    else str(head_layer_idx)
                current_head = model.classifier._modules[head_layer_idx]
                assert isinstance(current_head, torch.nn.Linear) \
                       or isinstance(current_head, torch.nn.Conv2d), "NO VALID HEAD IDX"

                heads = get_prev_heads(prev_head_model_paths, head_layer_idx)
                print("EVAL on prev heads: ", prev_head_model_paths)

                # BN
                if BN_stat_model is not None:
                    replace_BN_params(BN_stat_model, model)

                target_head_idx = 0
                assert n_heads == len(heads) == 1
                accuracy = test_model(model, dataset_path, target_head_idx, ds_classes,
                                      target_head=heads,
                                      batch_size=batch_size, final_layer_idx=head_layer_idx)

            # Append to sequences
            seq_acc[task_save_idx].append(accuracy)
            if trained_model_idx > test_task_ds_idx:
                first_task_acc = seq_acc[task_save_idx][0]
                seq_forgetting[task_save_idx].append(first_task_acc - accuracy)
            if head_accuracy is not None:
                seq_head_acc.append(head_accuracy)

            del model

        except Exception as e:
            print("ERROR in Testing model, trained until TASK ", str(trained_model_idx + 1))
            print("Aborting testing on further models")
            traceback.print_exc(5)
            break
    return seq_acc, seq_forgetting, seq_head_acc


def replace_BN_params(extract_model, out_model):
    """ Replace BatchNorm params in out_model, extracted from extract_model."""
    for extract_module, out_module in zip(extract_model.features.children(), out_model.features.children()):
        if isinstance(out_module, nn.BatchNorm1d) \
                or isinstance(out_module, nn.BatchNorm2d) \
                or isinstance(out_module, nn.BatchNorm3d):
            for extract_param, out_param in zip(extract_module.parameters(), out_module.parameters()):
                out_param.data.copy_(extract_param.data)
    print("Replaced BatchNorm stats.")


def LWF_patch(model):
    if isinstance(model, train.LwF.ModelWrapperLwF.ModelWrapperLwF):
        model_patch = train.LwF.ModelWrapperLwF.ModelWrapperLwF(model.model, model.last_layer_name)  # Patch
        assert hasattr(model_patch, 'model')
        if hasattr(model.model, 'features'):
            model_patch.features = model.model.features
        model_patch.classifier = model.model.classifier
        model = model_patch
    return model
