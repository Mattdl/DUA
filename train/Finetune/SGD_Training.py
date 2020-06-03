import time
import os
import copy

import torch
from torch.autograd import Variable

import shared_utils.utils as utils


def set_lr(optimizer, lr, count, decay_threshold=5, early_stop_threshold=10):
    """
    Early stop or decay learning rate by a factor of 0.1 based on count.
    Dynamic decaying speed (count based).

    :param optimizer:
    :param lr:              Current learning rate
    :param count:           Amount of times of not increasing accuracy.
    :param decay_threshold:
    :param early_stop_threshold:
    :return:
    """
    continue_training = True

    # Early Stopping
    if count > early_stop_threshold:
        continue_training = False
        print("training terminated")

    # Decay
    if count == decay_threshold:
        lr = lr * 0.1
        print('lr is set to {}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    return optimizer, lr, continue_training


def exp_lr_scheduler(optimizer, epoch, init_lr=0.0008, lr_decay_epoch=45):
    """
    (Deprecated)
    Lr decay for training from pretrained (start with smaller lr).
    From scratch: init_lr=0.01, lr_decay_epoch=50

    Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs.
    Fixed decaying speed (epoch based).
    """

    lr = init_lr * (0.1 ** (epoch // lr_decay_epoch))
    print('lr is ' + str(lr))
    if epoch % lr_decay_epoch == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def termination_protocol(since, best_acc):
    """
    Final stats printing: time and best validation accuracy.
    :param since:
    :param best_acc:
    :return:
    """
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))


def train_model(model, criterion, optimizer, lr, dset_loaders, dset_sizes, use_gpu, num_epochs, exp_dir='./',
                resume='', save_models_mode=True, model_epoch_saving_freq=30):
    """

    :param model: model object to start from
    :param criterion: loss function
    :param optimizer:
    :param lr:
    :param dset_loaders:    train and val dataset loaders
    :param dset_sizes:      traind and val sizes
    :param use_gpu:
    :param num_epochs:
    :param exp_dir:         where to output trained model
    :param resume:          path to model to resume from, empty string otherwise
    :param stack_head_cutoff: LWF, EBLL wrapper models have stacked heads
    :return:
    """
    print('dictionary length' + str(len(dset_loaders)))
    since = time.time()
    val_beat_counts = 0  # number of time val accuracy not improved
    best_acc = 0.0
    mem_snapshotted = False

    # Resuming from model if specified
    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        checkpoint = torch.load(resume)
        start_epoch = checkpoint['epoch']

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        lr = checkpoint['lr']
        print("lr is ", lr)
        val_beat_counts = checkpoint['val_beat_counts']
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(resume, checkpoint['epoch']))
    else:
        start_epoch = 0
        print("=> no checkpoint found at '{}'".format(resume))

    print(str(start_epoch))

    for epoch in range(start_epoch, num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                optimizer, lr, continue_training = set_lr(optimizer, lr, count=val_beat_counts)
                if not continue_training:
                    termination_protocol(since, best_acc)
                    return model, best_acc
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dset_loaders[phase]:
                # get the inputs
                inputs, labels = data

                # wrap them in Variable
                if use_gpu:
                    inputs, labels = Variable(inputs.cuda()), \
                                     Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs), Variable(labels)

                # zero the parameter gradients
                optimizer.zero_grad()
                model.zero_grad()

                # forward
                outputs = model(inputs)

                # For EBLL and LWF wrapper models (has multiple heads and tests all of them)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                # pdb.set_trace()
                # print(outputs)
                _, preds = torch.max(outputs.data, 1)

                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    try:
                        loss.backward()
                    except:
                        # For empty optimizer
                        loss = Variable(loss, requires_grad=True)
                        loss.backward()
                    # print('step')
                    optimizer.step()

                if not mem_snapshotted:
                    utils.save_cuda_mem_req(exp_dir)
                    mem_snapshotted = True

                # running statistics
                running_loss += loss.data.item()
                running_corrects += torch.sum(preds == labels.data)

            # epoch statistics
            epoch_loss = running_loss / dset_sizes[phase]
            epoch_acc = running_corrects.item() / dset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # new best model: deep copy the model or save to .pth
            if phase == 'val':
                if epoch_acc > best_acc:
                    del outputs
                    del labels
                    del inputs
                    del loss
                    del preds
                    best_acc = epoch_acc
                    if save_models_mode:
                        torch.save(model, os.path.join(exp_dir, 'best_model.pth.tar'))
                        print("-> saved new best model")
                    # best_model = copy.deepcopy(model)
                    val_beat_counts = 0
                else:
                    val_beat_counts += 1

        # Epoch checkpoint
        if save_models_mode and epoch % model_epoch_saving_freq == 0:
            epoch_file_name = exp_dir + '/' + 'epoch' + '.pth.tar'
            save_checkpoint({
                'epoch': epoch + 1,
                'lr': lr,
                'val_beat_counts': val_beat_counts,
                'epoch_acc': epoch_acc,
                'best_acc': best_acc,
                'arch': 'alexnet',
                'model': model,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }, epoch_file_name)
        print()

    termination_protocol(since, best_acc)
    return model, best_acc


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    # best_model = copy.deepcopy(model)
    torch.save(state, filename)
