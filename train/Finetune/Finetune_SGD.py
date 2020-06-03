import warnings
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim

import train.Finetune.SGD_Training as SGD_Training
import shared_utils.utils as utils
from shared_utils.ImageFolderTrainVal import ConcatDatasetDynamicLabels
import train


def fine_tune_SGD(manager, model_path, exp_dir, num_epochs=100, lr=0.0004,
                  weight_decay=0, enable_resume=True, replace_head=True, save_models_mode=True,
                  model_epoch_saving_freq=5, adaBN=False, adaBNTUNE=False, lwf_head=None, deactivate_bias=False,
                  BN_freeze_epochs=0):
    """
    Finetune training pipeline with SGD optimizer.
    (1) Performs training setup: dataloading, init.
    (2) Actual training: SGD_training.py

    :param model_path:       input model to start from
    :param exp_dir:          where to output new model
    :param freeze_mode:     true if only training classification layer
    :param enable_resume:   resume from existing epoch.pth.tar file, overwrite otherwise
    :param replace_head:    replace current head, with head of size dependent on current dataset classes.
    :param lwf_head:        fix for LwF with AdaBN, indicating which stacked head in LwF to select.
    :param BN_freeze_epochs: after training normal BN, freeze BN stats and train for this amount of epochs.
    :param deactivate_bias  deactivate biases in Conv layers
    :return:
    """
    print('lr is ' + str(lr))
    print("DATASETS: ", manager.dataset_path)

    # Resume
    if enable_resume:
        resume = os.path.join(exp_dir, 'epoch.pth.tar')
    else:
        resume = ''  # Don't resume

    if os.path.isfile(resume):  # Resume if there is already a model in the expdir!
        checkpoint = torch.load(resume)
        model_ft = checkpoint['model']
        print("Resumed from model: ", resume)
    else:
        if not os.path.exists(exp_dir):
            os.makedirs(exp_dir)

        if isinstance(model_path, str):
            if not os.path.isfile(model_path):
                raise Exception("None existing previous_task_model_path: {}".format(manager.previous_task_model_path))
            else:
                model_ft = torch.load(model_path)
                print("Starting from model path: ", model_path)
        else:
            model_ft = model_path  # model_path is already loaded model
    # GPU
    use_gpu = torch.cuda.is_available()

    criterion = nn.CrossEntropyLoss()

    # LWF wrapper fix
    if lwf_head is not None:
        model_ft = train.LwF.ModelWrapperLwF.ModelWrapperLwF(model_ft.model, model_ft.last_layer_name)  # Patch
        assert hasattr(model_ft, 'model')
        model_ft.set_finetune_mode(lwf_head)
        model_ft.features = model_ft.model.features
        model_ft.classifier = model_ft.model.classifier

    # Reset last classifier layer
    if replace_head:
        model_ft = utils.replace_last_classifier_layer(model_ft, len(manager.dset_classes))
        print("REPLACED LAST LAYER with {} new output nodes".format(len(manager.dset_classes)))

    if use_gpu:
        model_ft = model_ft.cuda()

    # Reset BN params
    utils.reset_BN_layers(model_ft)

    if adaBNTUNE:
        print("ADA_BN_TUNE mode")
        params_to_optimize = []
        for idx, mod in enumerate(model_ft.features.children()):
            if isinstance(mod, nn.BatchNorm1d) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod,
                                                                                                nn.BatchNorm3d):
                for param in mod.parameters():
                    params_to_optimize.append(param)
                    param.requires_grad = True
            else:
                for param in mod.parameters():
                    param.requires_grad = False
        # Freeze classifier
        for idx, mod in enumerate(model_ft.classifier.children()):
            for param in mod.parameters():
                param.requires_grad = False
        # Only optimize BN params
        optimizer_ft = optim.SGD(iter(params_to_optimize), lr, momentum=0.9)
    else:
        if adaBN:
            print("ADA_BN mode")
            for idx, mod in enumerate(model_ft.features.children()):
                for param in mod.parameters():
                    param.requires_grad = False
            # Freeze classifier
            for idx, mod in enumerate(model_ft.classifier.children()):
                for param in mod.parameters():
                    param.requires_grad = False
            print("Disabled all params, collecting BN stats.")
        # Default optim
        optimizer_ft = optim.SGD(model_ft.parameters(), lr, momentum=0.9, weight_decay=weight_decay)

    if deactivate_bias:
        print("DEACTIVATING BIASES")
        for idx, mod in enumerate(model_ft.features.children()):
            if isinstance(mod, nn.Conv2d):
                mod.bias = None

    # Start training
    model_ft, best_acc = SGD_Training.train_model(model_ft, criterion, optimizer_ft, lr, manager.dset_loaders,
                                                  manager.dset_sizes, use_gpu, num_epochs, exp_dir, resume,
                                                  save_models_mode=save_models_mode,
                                                  model_epoch_saving_freq=model_epoch_saving_freq)

    # Set model in eval mode and train on eval statistics
    if BN_freeze_epochs > 0:
        print("Doing BN Freeze for {} epochs".format(BN_freeze_epochs))
        params_to_optimize = []
        for idx, mod in enumerate(model_ft.features.children()):
            # Freeze BN layers
            if isinstance(mod, nn.BatchNorm1d) or isinstance(mod, nn.BatchNorm2d) or isinstance(mod, nn.BatchNorm3d):
                mod.eval()
                mod.track_running_stats = False
                for param in mod.parameters():
                    param.requires_grad = False
            else:
                for param in mod.parameters():
                    param.requires_grad = True
                    params_to_optimize.append(param)
        for idx, mod in enumerate(model_ft.classifier.children()):
            for param in mod.parameters():
                param.requires_grad = True
                params_to_optimize.append(param)

        # Only optimize BN params
        optimizer_ft = optim.SGD(iter(params_to_optimize), lr, momentum=0.9, weight_decay=weight_decay)

        model_ft, best_acc = SGD_Training.train_model(model_ft, criterion, optimizer_ft, lr, manager.dset_loaders,
                                                      manager.dset_sizes, use_gpu, BN_freeze_epochs, exp_dir,
                                                      # No resume!
                                                      save_models_mode=save_models_mode,
                                                      model_epoch_saving_freq=model_epoch_saving_freq)

    return model_ft, best_acc
