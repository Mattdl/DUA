from copy import deepcopy

from torch.nn import functional as F
from torch.autograd import Variable
import shared_utils.methods as methods

from shared_utils.ImageFolderTrainVal import *
import shared_utils.utils as utils

import train.MAS.MAS as MAS


def get_merge_weights(args, iw_user, head_param_names, models, last_task_idx, overwrite, cuda=True, eps=1e-10):
    """
    Calculate merged IWs for all the trained models and save them to the same location.
    """
    print("IMM PREPROCESSING: Mode {}, overwrite={}".format(args.IMM_mode, overwrite))
    iw_matrices = []
    sum_iw_matrices = []  # All summed of previous tasks (first task not included)
    sum_iw_matrix = None
    args.iw_paths = []
    args.sum_iw_paths = []

    iw_name = args.method.get_iw_name(args.eval_ds.name, user=iw_user, task_agnostic=args.task_agnostic)
    sum_iw_name = args.method.get_iw_name(args.eval_ds.name, user=iw_user, sum=True, task_agnostic=args.task_agnostic)

    # Checks
    assert len(args.iw_ds_holder) == len(args.merge_target_models), \
        "Subsetting requires both IW dataset and model subsetting."

    for model_list_index in range(0, last_task_idx + 1):
        print("*** ESTIMATING IWS on MODEL of TASK {}".format(model_list_index + 1))
        iw_out_file_path = os.path.join(os.path.dirname(args.merge_target_models[model_list_index]), iw_name)
        sum_iw_out_file_path = os.path.join(os.path.dirname(args.merge_target_models[model_list_index]), sum_iw_name)

        #######################################
        # Get individual model IW matrices
        if os.path.exists(iw_out_file_path) and not overwrite:
            model_iw_matrix = torch.load(iw_out_file_path)
            print('LOADED PRECISION MATRIX FOR TASK {} : {}'.format(model_list_index, iw_out_file_path))
        else:
            model = deepcopy(models[model_list_index])

            # When task-agnostic: iterate over all datasets for 1 model
            task_idxs = [model_list_index] if not args.task_agnostic else list(range(len(args.iw_ds_holder)))
            model_iw_matrix = None
            for task_idx in task_idxs:
                print("* IW: MODEL of TASK {}, on DS of TASK={}".format(model_list_index + 1, task_idx + 1))

                # Init data
                if isinstance(args.iw_ds_holder[task_idx], str):  # Load from path
                    dsets = torch.load(args.iw_ds_holder[task_idx])
                else:  # Imgfolders in memory
                    dsets = args.iw_ds_holder[task_idx]
                dset_loaders = {
                    x: torch.utils.data.DataLoader(dsets[x], batch_size=args.bs, shuffle=True, num_workers=8)
                    for x in args.ds_subsets}

                # Format params
                if args.debug:
                    print("PARAM NAMES")
                    [print(n) for n, p in model.named_parameters() if p.requires_grad]
                model.params = {n: p for n, p in model.named_parameters() if p.requires_grad}

                # Calculate IWS
                if isinstance(args.method, methods.IMM) and args.IMM_mode == 'mode_MAS' \
                        or isinstance(args.method, methods.LocallyAdaptive) and 'plain' in args.LA_mode:
                    dset_loaders = [dset for dset in dset_loaders.values()]
                    task_iw_matrix = MAS_iws(model, dset_loaders, exclude_params=head_param_names,
                                             normalize_mode=args.LA_mode,
                                             cuda=True, mode=args.method.mode)

                elif isinstance(args.method, methods.IMM) and args.IMM_mode == 'mode' \
                        or isinstance(args.method, methods.LocallyAdaptive) and 'FIM' in args.LA_mode:
                    task_iw_matrix = diag_fisher(model, dset_loaders, exclude_params=head_param_names, cuda=True)
                else:
                    raise Exception("NO valid method to get IWS for:{}".format(args.method))

                # Checks
                assert [task_iw_matrix.keys()] == [
                    {name for name, p in model.named_parameters() if name not in head_param_names}]
                if not cuda:
                    task_iw_matrix = {name: p.cpu().clone() for name, p in task_iw_matrix.items()}

                # Sum over all task_idxs
                if model_iw_matrix is None:
                    model_iw_matrix = task_iw_matrix
                else:
                    model_iw_matrix = add_iws(task_iw_matrix, model_iw_matrix)

                # Cleanup
                del model
                dset_loaders.clear()
                torch.cuda.empty_cache()

            if args.store_iws:
                print("Saving IW matrix: ", iw_out_file_path)
                torch.save(model_iw_matrix, iw_out_file_path)
                args.iw_paths.append(iw_out_file_path)
        iw_matrices.append(model_iw_matrix)

        #######################################
        # Update incremental sum matrix for each model
        if sum_iw_matrix is None:
            sum_iw_matrix = model_iw_matrix
        else:
            if os.path.exists(sum_iw_out_file_path) and not overwrite:
                sum_iw_matrix = torch.load(sum_iw_out_file_path)
                print('LOADED SUM-PRECISION MATRIX FOR TASK {} : {}'.format(model_list_index, sum_iw_out_file_path))
            else:
                if args.debug:
                    for name, p in sum_iw_matrix.items():
                        try:
                            print("{}: {} -> {}".format(name, p.shape, model_iw_matrix[name].shape))
                        except:
                            pass
                sum_iw_matrix = add_iws(model_iw_matrix, sum_iw_matrix)

                # This is of key importance to avoid divison by zero
                for name, p in sum_iw_matrix.items():
                    sum_iw_matrix[name][p == 0] = eps

                if args.store_iws:
                    print("Saving SUM precision matrix: ", sum_iw_out_file_path)
                    torch.save(sum_iw_matrix, sum_iw_out_file_path)
                    args.sum_iw_paths.append(sum_iw_out_file_path)
            sum_iw_matrices.append(sum_iw_matrix)
            torch.cuda.empty_cache()

    return iw_matrices, sum_iw_matrices


def add_iws(iw_matrix, sum_iw_matrix):
    """ ADD iw_matrix to sum_iw_matrix."""
    sum_iw_matrix = {name: p + iw_matrix[name]
                     for name, p in sum_iw_matrix.items()}
    assert len([iw_matrix[name] != p for name, p in sum_iw_matrix.items()]) > 0

    return sum_iw_matrix


def merging_preprocess(args, overwrite=False, cuda=True):
    """
    Create and save all merged models for the specified method.
    :param args.merge_target_models: list of all model paths to merge incrementally.
    :param user:  user for which to calculate IWs, None means no specific user.
    :param overwrite: Overwrite if exists.
    :param cuda: Store and load all models, IWs in cuda format. Otherwise only IW calculation itself runs on GPU.
    :return: pathlist of merged models
    """

    IMM_mode = args.method.mode
    use_iws = isinstance(args.method, methods.LocallyAdaptive) or \
              isinstance(args.method, methods.IMM) and (IMM_mode == 'mode' or IMM_mode == 'mode_MAS')
    merged_model_paths = []
    last_task_idx = len(args.merge_target_models) - 1
    merge_model_name = args.method.get_merged_model_name(args.eval_ds.name, args.user)

    models = None

    # Keep first model (no merge needed)
    merged_model_paths.append(args.merge_target_models[0])

    # Create merged model for each task (except first)
    iws_initialized = False

    for task_list_index in range(1, last_task_idx + 1):
        out_file_path = os.path.join(os.path.dirname(args.merge_target_models[task_list_index]), merge_model_name)
        skip_merge = args.skip_merging or os.path.exists(out_file_path) and not overwrite

        # Init only when necessary
        if models is None:
            models = init_models(args, cuda)
            last_layer_index, _ = utils.get_last_neural_layer(models[0])
            head_param_names = ['classifier.{}.{}'.format(last_layer_index, name) for name, p in
                                models[0].classifier._modules[last_layer_index].named_parameters()]
            if args.debug:
                print("HEAD PARAM NAMES")
                [print(name) for name in head_param_names]

        # Mean IMM
        if IMM_mode == 'mean':
            if skip_merge:
                print("SKIPPED MERGING OF: {}".format(out_file_path))
            else:
                merged_model = IMM_merge_models(models, task_list_index, head_param_names, cuda=cuda)

        # Mode IMM
        elif use_iws:
            if not iws_initialized:
                iw, sum_iw = get_merge_weights(args, args.iw_user, head_param_names, models, last_task_idx,
                                               overwrite, cuda=cuda)
                iws_initialized = True

            if skip_merge:
                print("SKIPPED MERGING OF: {}".format(out_file_path))
            else:
                merged_model = IMM_merge_models(models, task_list_index, head_param_names,
                                                iw=iw, sum_iw=sum_iw[task_list_index - 1], cuda=cuda)
        else:
            raise ValueError("IMM mode is not supported: ", str(IMM_mode))

        merged_model_paths.append(out_file_path)
        torch.cuda.empty_cache()

        if not skip_merge:
            # Save merged model on same spot as best_model
            torch.save(merged_model, out_file_path)
            del merged_model
            print(" => SAVED MERGED MODEL: ", out_file_path)

    try:
        del head_param_names, models[:]
        del iw[:], sum_iw[:]
    except:
        pass

    print("MERGED MODELS:")
    print('\n'.join(map(str, merged_model_paths)))

    return merged_model_paths


def init_models(args, cuda):
    models = []
    for model_path in args.merge_target_models:
        if isinstance(model_path, str):
            model = torch.load(model_path, map_location=lambda storage, loc: storage)  # Load all on cpu
        else:
            model = model_path
        if cuda:
            model = model.cuda()
        else:
            model = model.cpu()
        try:
            del model.reg_params
            print("removed reg_params from loaded model")
        except:
            pass

        models.append(model)
        torch.cuda.empty_cache()

    print("MODELS TO MERGE:")
    print('\n'.join(args.merge_target_models))
    print("LOADED ", len(models), " MODELS in MEMORY")
    return models


def diag_fisher(model, dataset, exclude_params=None, cuda=True):
    """ FIM-IMM IWs as in EWC."""
    print("Calculating precision matrix")
    if cuda:
        model = model.cuda()

    # initialize space for precision_matrix
    precision = {}
    for n, p in deepcopy(model.params).items():
        if n in exclude_params:
            # print("Skipping diag calculation for param ", n)
            continue
        p.data.zero_()
        precision[n] = Variable(p.data + 1e-8)

    # fill matrix
    model.eval()
    for phase in dataset.keys():
        for input in dataset[phase]:
            model.zero_grad()
            x, label = input
            x, label = Variable(x), Variable(label, requires_grad=False)

            if cuda:
                x = x.cuda()
                label = label.cuda()
            output = model(x)  # .view(1, -1)
            temp = F.softmax(output).data

            targets = torch.reshape(Variable(torch.multinomial(temp, 1).clone()), (-1,))

            if cuda:
                targets.cuda()

            # label = output.max(1)[1].view(-1)
            loss = F.nll_loss(F.log_softmax(output, dim=1), targets, size_average=True)

            loss.backward()

            for n, p in model.named_parameters():
                if n in exclude_params:
                    continue
                precision[n].data += p.grad.data ** 2 / len(dataset[phase])

    precision_param = {n: p for n, p in precision.items()}
    return precision_param


def MAS_iws(model, dset_loaders, norm='L2', exclude_params=None, normalize_mode=None, cuda=True, mode=None):
    """MAS."""
    print("Calculating MAS IWs.")
    model = MAS.get_new_iws(None, None, model, None, dset_loaders=dset_loaders, norm=norm, cuda=cuda,
                            mode=mode).cpu()
    iws = {name: model.reg_params[param]['omega'].clone() for name, param in model.named_parameters()
           if param in model.reg_params and name not in exclude_params}
    del model
    torch.cuda.empty_cache()

    if normalize_mode == 'layernorm':
        iws = {name: iw / torch.max(iw) for name, iw in iws.items()}
        print("LAYERWISE NORMALIZATION")
    elif normalize_mode == 'modelnorm':
        max_iws = [torch.max(val) for val in iws.values()]
        max_iw = max(max_iws)
        print("MAX IW = {}".format(max_iw))
        iws = {name: iw / max_iw for name, iw in iws.items()}
        print("MODELWISE NORMALIZATION")
    elif normalize_mode == 'stdlayernorm':
        iws = {name: iw / torch.std(iw) for name, iw in iws.items()}
        print("STD LAYERWISE NORMALIZATION")
    elif normalize_mode == 'stdmodelnorm':
        std_iws = [torch.std(val) for val in iws.values()]
        max_iw = max(std_iws)
        print("MAX STD IW = {}".format(max_iw))
        iws = {name: iw / max_iw for name, iw in iws.items()}
        print("STD MODELWISE NORMALIZATION")

    return iws


def IMM_merge_models(models, task_list_idx, head_param_names, iw=None, sum_iw=None, cuda=True):
    """
    Mean-IMM, averaging all the parameters of the trained models up to the given task.
    Mode-IMM, see paper: using preciison and sum_precision

    Here alphas are all equal (1/ #models). All alphas must sum to 1.

    :param models: list with all models preceding and current model of param task
    :param task_list_idx: up to and including which task the models should be merged
    :return: new merged model, as we don't want to update the models param ()
    """
    mean_mode = iw is None and sum_iw is None

    print("MERGE MODE=[MEAN={}, IW={}] Merging models for TASK {}"
          .format(mean_mode, not mean_mode, str(task_list_idx + 1)))
    merged_model = deepcopy(models[task_list_idx])

    total_task_count = task_list_idx + 1  # e.g. task_idx 1, means to avg over task_idx 0 and 1 => 2 tasks

    # Iterate params
    for param_name, param_value in merged_model.named_parameters():
        # Don't merge heads (we use separate heads)
        if param_name in head_param_names:
            print("NOT MERGING PARAM {}, as it is a head param name".format(param_name))
            continue

        # Calculate Mean
        mean_param = torch.zeros(param_value.data.size())
        if cuda:
            mean_param = mean_param.cuda()
        for merge_task_idx in range(0, total_task_count):  # Avg over all preceding + including current task

            # Error check
            if models[merge_task_idx].state_dict()[param_name].size() != mean_param.size():
                print("ERROR WHEN MERGING MODELS")
                raise Exception("ERROR WHEN MERGING MODELS: PRECEDING MODEL PARAMS TASK",
                                str(merge_task_idx), " != PARAM SIZE OF REF TASK", str(task_list_idx))

            if mean_mode:  # MEAN IMM
                state_dict = models[merge_task_idx].state_dict()
                param_value = state_dict[param_name]
                mean_param = mean_param + param_value
            else:  # MODE IMM
                merge_weighting = iw[merge_task_idx][param_name] / sum_iw[param_name]
                d_mean_param = merge_weighting.data * models[merge_task_idx].state_dict()[param_name]
                mean_param += d_mean_param
                assert torch.all(torch.eq(torch.sum(merge_weighting.gt(1)), 0)), \
                    print("MERGE VALUE > 1: {}".format(merge_weighting[torch.nonzero(merge_weighting.gt(1))]))
                assert not torch.any(torch.isnan(merge_weighting))

        # Task_idx is count of how many iterated
        if mean_mode:
            mean_param = mean_param / total_task_count  # Cancels out in mode IMM
        # Update this avged param
        param_value.data = mean_param.clone()

    return merged_model
