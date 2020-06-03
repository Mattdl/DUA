import torch.nn as nn
from torchvision import models

from train.IMM.L2_transfer_utils import L2transfer_Training
from shared_utils.ImageFolderTrainVal import *
import shared_utils.utils as utils


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def update_reg_params(model, freeze_layers=None):
    """

    Constructs dictionary reg_param = {}, with:
        reg_param['omega'] = omega  # Importance weights (here all set to 1, as they are not used in mean-IMM)
        reg_param['init_val'] = init_val # The original weight values of the prev task model

    See "L2transfer_Training.py", where init_val and omegas are used.

    :param model:
    :param freeze_layers:
    :return:
    """
    print("UPDATING REG PARAMS")
    reg_params = model.reg_params

    if freeze_layers is None:
        freeze_layers = []

    for index, (name, param) in enumerate(model.named_parameters()):
        print('Named params: index' + str(index))

        # If the param is a reg_param within the model
        if param in reg_params.keys():  # Restoring SI omega/init_val hyperparams
            if name not in freeze_layers:
                print('updating index' + str(index))

                omega = torch.ones(param.size())
                init_val = param.data.clone()

                reg_param = {}
                reg_param['omega'] = omega  # Importance weights?
                reg_param['init_val'] = init_val  # The original weights of prev network

                # Update model
                reg_params[param] = reg_param
            else:
                reg_param = reg_params.get(param)
                print('removing unused frozen omega', name)
                del reg_param['omega']
                del reg_params[param]
        else:
            print('initializing index' + str(index))
            omega = torch.ones(param.size())
            init_val = param.data.clone()

            reg_param = {}
            reg_param['omega'] = omega
            reg_param['init_val'] = init_val

            # Update model
            reg_params[param] = reg_param

    return reg_params


def fine_tune_l2transfer(manager, model_path, exp_dir, num_epochs=100, lr=0.0004, reg_lambda=100,
                         init_freeze=0, weight_decay=0, model_epoch_saving_freq=5):
    """
    IMM pipeline, only using L2-transfer technique and weight transfer. The zero-point of L2 regularization
    is set to the previous model. This way, instead of minimizing just the weights of the
    current model, the difference with the weights of the previous model is minimized.
    This is not implemented in the loss, but directly in a wrapper class of the optimizer.

    Weight transfer means starting next task model, based on prev task model.

    Using L2-transfer and weight transfer technique shows best performance for both mean and mode-IMM.

    reg_params is dictionary which looks like:
    - x times tensor(){omega=[one-vectors], init_val=[weights prev task net]}
    - lambda = the regularization hyperparameter used
    - 2 param tensors

    :param dataset_path:
    :param model_path:
    :param exp_dir:
    :param batch_size:
    :param num_epochs:
    :param lr:
    :param reg_lambda:  reg hyperparam for the L2-transfer
    :param init_freeze:
    :param weight_decay:
    :return:
    """
    print('lr is ' + str(lr))

    ########################################
    # LOAD INIT MODEL
    ########################################
    resume = os.path.join(exp_dir, 'epoch.pth.tar')

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        print("=> RESUMING FROM CHECKPOINTED MODEL: ", resume)
    else:
        if not os.path.isfile(model_path):
            model_ft = models.alexnet(pretrained=True)
            print("=> STARTING PRETRAINED ALEXNET")

        else:
            model_ft = torch.load(model_path)
            print("=> STARTING FROM OTHER MODEL: ", model_path)

    # Replace last layer classifier, for the amount of classes in the current dataset
    if not init_freeze:
        model_ft = utils.replace_last_classifier_layer(model_ft, len(manager.dset_classes))

    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)


    # If not resuming from an preempted IMM model, cleanup model
    if not os.path.isfile(resume):

        if not hasattr(model_ft, 'reg_params'):
            reg_params = initialize_reg_params(model_ft)
        else:
            # Prev task model, last 2 hyperparam
            parameters = list(model_ft.parameters())
            parameter1 = parameters[-1]
            parameter2 = parameters[-2]

            # Try to remove them from the reg_params
            try:
                model_ft.reg_params.pop(parameter1, None)
                model_ft.reg_params.pop(parameter2, None)
            except:
                print('nothing to remove')

            # The regularization params are the parameters of the prev model (trying to)
            reg_params = update_reg_params(model_ft)
            print('update')
        reg_params['lambda'] = reg_lambda  # The regularization hyperparam
        model_ft.reg_params = reg_params

    # Only transfer here to CUDA, preventing non-cuda network adaptations
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        model_ft = model_ft.cuda()

    # Reset BN params
    utils.reset_BN_layers(model_ft)

    ########################################
    # TRAIN
    ########################################

    # Define Optimizer for IMM: extra loss term in step
    optimizer_ft = L2transfer_Training.Weight_Regularized_SGD(model_ft.parameters(), lr, momentum=0.9,
                                                              weight_decay=weight_decay)
    # Loss
    criterion = nn.CrossEntropyLoss()

    model_ft, acc = L2transfer_Training.train_model(model_ft, criterion, optimizer_ft, lr, manager.dset_loaders,
                                                    manager.dset_sizes, use_gpu, num_epochs, exp_dir, resume,
                                                    model_epoch_saving_freq=model_epoch_saving_freq)

    return model_ft, acc


def initialize_reg_params(model):
    reg_params = {}
    # for param in list(model.parameters()):
    for name, param in model.named_parameters():  # after deadline check
        w = torch.FloatTensor(param.size()).zero_()
        omega = torch.FloatTensor(param.size()).zero_()
        init_val = param.data.clone()
        reg_param = {}
        reg_param['omega'] = omega
        reg_param['w'] = w
        reg_param['init_val'] = init_val
        reg_param['name'] = name
        reg_params[param] = reg_param
    return reg_params


def models_mean_moment_matching(models, task):
    for name, param in models[task].named_parameters():
        # w=torch.FloatTensor(param.size()).zero_()
        print(name)
        mean_param = torch.zeros(param.data.size()).cuda()
        for i in range(0, len(models)):
            if models[i].state_dict()[name].size() != mean_param.size():
                return models[task]
            mean_param = mean_param + models[i].state_dict()[name]
        mean_param = mean_param / len(models)
        param.data = mean_param.clone()
