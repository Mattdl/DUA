import os
import configparser
import sys
import random
import numpy
import warnings
import datetime
import uuid

import torch.nn as nn
import torch


########################################
# CONFIG PARSING
########################################
def get_root_src_path():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '..'))


def get_parsed_config():
    """ Using this file as reference to get src root: src/<dir>/utils.py"""
    config = configparser.ConfigParser()
    src_path = get_root_src_path()
    config.read(os.path.join(src_path, 'config.init'))

    # Replace "./" with abs src root path
    for key, path in config['DEFAULT'].items():
        if '.' == read_from_config(config, key).split(os.path.sep)[0]:
            config['DEFAULT'][key] = os.path.join(src_path, read_from_config(config, key)[2:])

    return config


def read_from_config(config, key_value):
    return os.path.normpath(config['DEFAULT'][key_value]).replace("'", "").replace('"', "")


########################################
# DETERMINISTIC
########################################
def set_random(seed=None):
    if seed is None:
        seed = 7
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    numpy.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


########################################
# PYTORCH UTILS
########################################
def replace_last_classifier_layer(model, out_dim):
    """Replace last linear/conv2d layer in the classifier of the model, to have out_dim output dimensions."""
    last_classlayer_index, type = get_last_neural_layer(model)

    if type == 'linear':
        num_ftrs = model.classifier._modules[last_classlayer_index].in_features
        model.classifier._modules[last_classlayer_index] = nn.Linear(num_ftrs, out_dim).cuda()
    elif type == 'conv2d':
        model.num_classes = out_dim
        num_ftrs = model.classifier._modules[last_classlayer_index].in_channels
        kernel_size = model.classifier._modules[last_classlayer_index].kernel_size
        stride = model.classifier._modules[last_classlayer_index].stride
        model.classifier._modules[last_classlayer_index] = nn.Conv2d(num_ftrs, out_dim, kernel_size, stride).cuda()

    print("NEW {} CLASSIFIER HEAD with {} units".format(type, out_dim))
    return model


def get_last_neural_layer(model):
    """Searches classifier from last to first index, returning first encountered Linear/conv2d layer."""
    last_classlayer_index = None
    type = None
    for idx, mod in reversed(list(enumerate(model.classifier.children()))):
        if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d):
            type = 'linear' if isinstance(mod, nn.Linear) else 'conv2d'
            last_classlayer_index = idx
            break

    # Pytorch inconsistency fix
    last_classlayer_index = last_classlayer_index if last_classlayer_index in model.classifier._modules \
        else str(last_classlayer_index)

    return last_classlayer_index, type


def reset_BN_layers(model):
    if hasattr(model, 'features'):
        for idx, mod in enumerate(model.features.children()):
            if isinstance(mod, nn.BatchNorm1d) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm3d):
                mod.reset_parameters()
                torch.nn.init.ones_(mod.weight)  # In Pytroch v1.0.1 uniform weight reset --> v1.3.0 all to 1
    else:
        print("[WARN] Model has no attribute 'features'")


def save_cuda_mem_req(out_dir, out_filename='cuda_mem_req.pth.tar'):
    """

    :param out_dir: /path/to/best_model.pth.tar
    :param out_filename:
    :return:
    """
    out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)

    mem_req = {}
    mem_req['cuda_memory_allocated'] = torch.cuda.memory_allocated(device=None)
    mem_req['cuda_memory_cached'] = torch.cuda.memory_cached(device=None)

    torch.save(mem_req, out_path)
    print("SAVED CUDA MEM REQ {} to path: {}".format(mem_req, out_path))


def save_preprocessing_time(out_dir, time, out_filename='preprocess_time.pth.tar'):
    if os.path.isfile(out_dir):
        out_dir = os.path.dirname(out_dir)
    out_path = os.path.join(out_dir, out_filename)
    torch.save(time, out_path)
    print("SAVED CUDA PREPROCESSING TIME {} to path: {}".format(time, out_path))


########################################
# PATH UTILS
########################################

def get_now():
    return str(datetime.datetime.now().date()) + "_" + ':'.join(str(datetime.datetime.now().time()).split(':')[:-1])


def get_unique_id():
    return uuid.uuid1()


def get_test_results_path(project_root_path, dataset_obj, method, model, gridsearch_name=None, exp_name=None,
                          merge_subset_idxs=None, create=False, adaBN=False, adaBNTUNE=False):
    path = os.path.join(project_root_path, 'results', dataset_obj.name, method, model)
    if gridsearch_name is not None:
        if adaBN:
            gridsearch_name += '_adaBN'
        if adaBNTUNE:
            gridsearch_name += '_adaBNTUNE'
        path = os.path.join(path, gridsearch_name)
    if merge_subset_idxs is not None:
        subset_path = "subset={}".format(str(sorted(merge_subset_idxs)))
        path = os.path.join(path, subset_path)
    if exp_name is not None:
        path = os.path.join(path, exp_name)
    if create:
        create_dir(path)
    return path


def get_train_results_path(exp_results_root_path, dataset, method, model, grid_name=None,
                           exp_name=None, filename=None, create=False, joint=False):
    if create and filename is not None:
        print("WARNING: filename is not being created, but superdirs are if not existing.")
    path = os.path.join(exp_results_root_path, dataset.name, method, model)
    if grid_name is not None:
        path = os.path.join(path, 'exp') if grid_name is None else os.path.join(path, 'gridsearch', grid_name)
    if exp_name is not None:
        path = os.path.join(path, exp_name)
    if joint:
        path = os.path.join(path, 'JOINT_BATCH')
    if create:
        create_dir(path)
    if filename is not None:
        path = os.path.join(path, filename)

    return path


def get_exp_name(args, method):
    ret = [method.name]
    hparams = ['lr', 'bs', 'epochs'] + method.hyperparams
    if args.seed is not None:
        hparams += ['seed']
    for hparam in hparams:
        ret.append(hparam + "=" + str(getattr(args, hparam)))
    if args.weight_decay != 0:
        ret.append("L2=" + str(args.weight_decay))
    if hasattr(args, 'deactivate_bias'):
        if args.deactivate_bias:
            ret.append("nobias")
    return '_'.join(ret)


def get_perf_output_filename(method_name, task_count=None, user=None):
    name = ['evalresults', method_name]
    if task_count is not None:
        name.append('task' + str(task_count))
    if user is not None:
        name.append('user' + str(user))
    return '_'.join(name) + ".pth"


def get_FT_model_dirname(task_count, user, it):
    """ Model name when Finetuning on exisiting model."""
    name = ['MODEL_FT', ]
    if task_count is not None:
        name.append('task' + str(task_count))
    if user is not None:
        name.append('user' + str(user))
    if it is not None:
        name.append('it' + str(it))
    name.append("id={}".format(get_unique_id()))
    return '_'.join(name)


def get_img_outpath(dataset_name, method_names, model, title=""):
    return '_'.join([title, dataset_name, "(" + '_'.join(method_names) + ")", model.name])


def get_hyperparams_output_filename():
    return 'hyperparams.pth.tar'


def get_immediate_subdirectories(parent_dir_path, path_mode=False, sort=False):
    """
    :param parent_dir_path: dir to take subdirs from
    :param path_mode: if true, returns subdir paths instead of names
    :param sort: sort subdirectory names
    :return: List of immediate subdirs
    """
    if not path_mode:
        dirs = [name for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    else:
        dirs = [os.path.join(parent_dir_path, name) for name in os.listdir(parent_dir_path)
                if os.path.isdir(os.path.join(parent_dir_path, name))]
    if sort:
        dirs.sort()
    return dirs


def create_symlink(src, dest):
    if not os.path.exists(dest):
        create_dir(os.path.dirname(dest))
        os.symlink(src, dest)


########################################
# MISCELLANEOUS UTILS
########################################
def create_dir(dirpath, print_description=""):
    try:
        if not os.path.exists(dirpath):
            os.makedirs(dirpath, mode=0o750)
    except:
        print("ERROR IN CREATING ", print_description, " PATH:", dirpath)


def float_to_scientific_str(value, sig_count=1):
    """
    {0:.6g}.format(value) also works

    :param value:
    :param sig_count:
    :return:
    """
    from decimal import Decimal
    format_str = '%.' + str(sig_count) + 'E'
    return format_str % Decimal(value)


def debug_add_sys_args(string_cmd, set_debug_option=True):
    """ Add debug arguments as params, this is for IDE debugging usage."""
    warnings.warn("=" * 20 + "SEVERE WARNING: DEBUG CMD ARG USED, TURN OF FOR REAL RUNS" + "=" * 20)

    args = string_cmd.split(' ')
    if set_debug_option:
        args.insert(0, "--debug")
    for arg in args:
        sys.argv.append(str(arg))
