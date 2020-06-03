""" Code altered from: Aljundi, Rahaf, et al. "Memory aware synapses: Learning what (not) to forget." ECCV. 2018."""
import os

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from train.MAS.MAS_utils import Objective_based_Training
import shared_utils.utils as utils
import shared_utils.ImageFolderTrainVal as ImageFolderTrainVal


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0004, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""

    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))

    if epoch > 0 and epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1

    return optimizer


def fine_tune_objective_based_acuumelation(manager, init_model_path, data_dir,
                                           reg_lambda=1, norm='L2', num_epochs=100, lr=0.0008, batch_size=200,
                                           weight_decay=0, b1=True, L1_decay=False, head_shared=False,
                                           model_epoch_saving_freq=5, mode=None):
    """
    In case of accumelating omega for the different tasks in the sequence, baisically to mimic the setup of other method where 
    the reguilizer is computed on the training set. Note that this doesn't consider our adaptation

    """
    use_gpu = torch.cuda.is_available()

    model_ft = torch.load(manager.previous_task_model_path)
    if isinstance(model_ft, dict):
        model_ft = model_ft['model']
    if b1:
        # compute the importance with batch size of 1, to mimic the online setting
        update_batch_size = 1
    else:
        update_batch_size = batch_size
    # update the omega for the previous task, accumelate it over previous omegas
    model_ft = accumulate_objective_based_weights(data_dir, manager.reg_sets, model_ft, update_batch_size, norm,
                                                  test_set="train", mode=mode)
    # set the lambda for the MAS regularizer
    model_ft.reg_params['lambda'] = reg_lambda

    # get the number of features in this network and add a new task head
    if not head_shared:
        last_layer_index = str(len(model_ft.classifier._modules) - 1)
        if not init_model_path is None:
            init_model = torch.load(init_model_path)
            model_ft.classifier._modules[last_layer_index] = init_model.classifier._modules[last_layer_index]

        else:
            model_ft = utils.replace_last_classifier_layer(model_ft, len(manager.dset_classes))

    criterion = nn.CrossEntropyLoss()

    if use_gpu:
        model_ft = model_ft.cuda()

    # call the MAS optimizer
    optimizer_ft = Objective_based_Training.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,
                                                                   weight_decay=weight_decay, L1_decay=L1_decay)

    if not os.path.exists(manager.exp_dir):
        os.makedirs(manager.exp_dir)
    if init_model_path is not None:
        del init_model
    # if there is a checkpoint to be resumed, in case where the training has stopped before on a given task
    resume = os.path.join(manager.exp_dir, 'epoch.pth.tar')

    # Reset BN params
    utils.reset_BN_layers(model_ft)

    # train the model
    # this training functin passes the reg params to the optimizer to be used for penalizing changes on important params
    model_ft, acc = Objective_based_Training.train_model(model_ft, criterion, optimizer_ft, lr, manager.dset_loaders,
                                                         manager.dset_sizes, use_gpu, num_epochs, manager.exp_dir,
                                                         resume,
                                                         model_epoch_saving_freq=model_epoch_saving_freq)
    return model_ft, acc


def get_new_iws(data_dir, reg_sets, model_ft, batch_size, norm='L2', test_set="train", dset_loaders=None, cuda=True,
                mode=None):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if dset_loaders is None:
        dset_loaders = []
        for data_path in reg_sets:

            # if so then the reg_sets is a dataset by its own, this is the case for the mnist dataset
            if data_dir is not None:
                dset = ImageFolderTrainVal(data_dir, data_path, data_transform)
            else:
                dset = torch.load(data_path)
                dset = dset[test_set]

            dset_loader = torch.utils.data.DataLoader(dset, batch_size=batch_size,
                                                      shuffle=False, num_workers=4)
            dset_loaders.append(dset_loader)

    use_gpu = torch.cuda.is_available() and cuda

    if use_gpu:
        model_ft = model_ft.cuda()
    # hack
    if not hasattr(model_ft, 'reg_params'):
        reg_params = Objective_based_Training.initialize_reg_params(model_ft)
        model_ft.reg_params = reg_params

    reg_params = Objective_based_Training.initialize_store_reg_params(model_ft)
    model_ft.reg_params = reg_params
    optimizer_ft = Objective_based_Training.Objective_After_SGD(model_ft.parameters(),
                                                                lr=0.0001, momentum=0.9, cuda=cuda)

    if norm == 'L2':
        print('********************objective with L2 norm***************')
        model_ft = Objective_based_Training.compute_importance_l2(model_ft, optimizer_ft, exp_lr_scheduler,
                                                                  dset_loaders, use_gpu, mode=mode)
    else:
        model_ft = Objective_based_Training.compute_importance(model_ft, optimizer_ft, exp_lr_scheduler, dset_loaders,
                                                               use_gpu)

    # sanitycheck(model_ft)
    return model_ft


def accumulate_objective_based_weights(data_dir, reg_sets, model_ft, batch_size, norm='L2', test_set="train",
                                       dset_loaders=None, mode=None):
    model_ft = get_new_iws(data_dir, reg_sets, model_ft, batch_size, norm, test_set, dset_loaders, mode=mode)

    # Accumulate
    reg_params = Objective_based_Training.accumelate_reg_params(model_ft)
    model_ft.reg_params = reg_params

    # sanitycheck(model_ft)
    return model_ft


def sanitycheck(model, include_bias=True):
    for name, param in model.named_parameters():
        if not include_bias and 'bias' in name:
            continue
        if param in model.reg_params:
            reg_param = model.reg_params.get(param)
            omega = reg_param.get('omega')
            print("{}: MEAN={}, MAX={}, MIN={}"
                  .format(name, omega.mean().item(), omega.max().item(), omega.min().item()))
