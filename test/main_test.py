import torch
import os
import argparse
import traceback
import copy

import test.test_seq as test_seq

import shared_utils.datasets as datasets
import shared_utils.methods as methods
import shared_utils.models as models
import shared_utils.utils as utils

import train.MAS.MAS as MAS
import train.IMM.merge as IMM
from train.main_train import Manager

parser = argparse.ArgumentParser(description='LACL Testing')

# REQUIRED ARGS
parser.add_argument('--model_name', choices=models.model_names, default="alexnet_pretrain")
parser.add_argument('--method_name', choices=methods.method_names, default=methods.EWC.name)
parser.add_argument('--ds_name', choices=datasets.dataset_names, default="MITindoorscenes")

# OPTIONS
parser.add_argument('--iterations', type=int, default=5, help="Iterations per user, to gain mean/std of evalresults.")
parser.add_argument('--scriptmode', choices=["eval", "analyze_models", "plot_iws"], default="eval")
parser.add_argument('--gridsearch_name', type=str, default=None, help="Name of the TRAINING gridsearch.")
parser.add_argument('--outgrid', type=str, default=None, help="Name of the TEST gridsearch name.")
parser.add_argument('--exp_name', type=str, default=None, help="Name of the experiment.")
parser.add_argument('--starting_task_count', type=int, default=1)
parser.add_argument('--max_task_count', type=int, default=None)
parser.add_argument('--bs', type=int, default=20, help="batch size.")

# MODES
parser.add_argument('--BN_mode', choices=['adaBN', 'adaBNTUNE', "None"], type=str, default=None,
                    help="Adaptive Batch Normalization per User per task.")
parser.add_argument('--debug', action="store_true", help="Not saving or overwriting any results.")
parser.add_argument('--overwrite_mode', action="store_true")
parser.add_argument('--cpu', action="store_true")
parser.add_argument('--check_pretrained', action="store_true",
                    help="Check MAS IWs magnitude for pretrained model.")

# METHODS
parser.add_argument('--IMM_mode', choices=methods.IMM.modes, type=str, default=None)
parser.add_argument('--LA_mode', choices=methods.LocallyAdaptive.modes, type=str, default=None,
                    help='plain: MAS IWs, FIM: EWC IWs')
parser.add_argument('--merge_subset', type=str, default=None, help="Tasks to subset merging of the models on.")

# Dataset alterations
parser.add_argument('--usercount', type=int, default=None, help="Nb users.")
parser.add_argument('--user_cat_pref', type=int, default=None, help="Nb preferences per user.")
parser.add_argument('--user_cat_prefimgs', type=int, default=None, help="Nb images of a preferred category per user.")
parser.add_argument('--transform_severeness', type=int, default=None, help="TransPrior transforms severeness.")


def main(overwrite_args=None):
    """ User-specific training if applicable, and testing per user."""
    utils.set_random()

    ########################################
    # ARGS Postprocessing
    ########################################
    args = parser.parse_args()
    if overwrite_args is not None:
        for k, v in overwrite_args.items():  # Enables direct calls to main method
            setattr(args, k, v)

    args.cuda = not args.cpu

    # ADD EXTERNAL PROJECT PATHS
    config = utils.get_parsed_config()
    args.test_results_root_path = utils.read_from_config(config, 'test_results_root_path')
    args.tr_results_root_path = utils.read_from_config(config, 'tr_results_root_path')
    args.models_root_path = utils.read_from_config(config, 'models_root_path')

    # PARSE DS, METHOD, MODEL
    args.preload_ds = True if "MIT" in args.ds_name else False  # Use in-memory imagefolders
    args.BN = True if 'BN' in args.model_name.split("_") else False  # BN in name
    args.tr_ds, args.eval_ds = datasets.parse(args.ds_name, args.usercount, args.user_cat_pref, args.user_cat_prefimgs,
                                              args.transform_severeness, save_imgfolders=(not args.preload_ds))
    args.method = methods.parse(args.method_name)
    args.base_model = models.parse_model_name(args.models_root_path, args.model_name, args.eval_ds.input_size)
    if args.BN_mode == 'adaBN':
        args.adaBN = True
        args.adaBNTUNE = False
    elif args.BN_mode == 'adaBNTUNE':
        args.adaBNTUNE = True
        args.adaBN = False
    else:
        args.adaBNTUNE = False
        args.adaBN = False

    # CHECKS
    if args.starting_task_count < 1 or args.starting_task_count > args.eval_ds.task_count:
        raise ValueError("ERROR: Starting task count should be in appropriate range for dataset! Value = ",
                         args.starting_task_count)

    if isinstance(args.method, methods.Finetune):
        assert not args.adaBN and not args.adaBNTUNE, "FT already inherently adaptive."

    # INIT
    args.max_task_count = args.eval_ds.task_count if args.max_task_count is None else args.max_task_count
    args.cleanup_merged_models, args.cleanup_FT_models, args.cleanup_BN_models = False, False, False
    args.BN_model_paths = None
    args.in_memory_models = True

    # Parse idxes
    if args.merge_subset is None:  # Select all
        args.user_merge_subset_idxs = [t_idx for t_idx in range(args.starting_task_count - 1, args.max_task_count)]
    else:
        args.user_merge_subset_idxs = [int(idx) for idx in args.merge_subset.split(',')]
    # Tasks to actually evaluate
    args.tasks_to_eval = [task_idx in args.user_merge_subset_idxs
                          for task_idx in range(args.starting_task_count - 1, args.max_task_count)]

    ########################################
    # Config Data and Model paths
    ########################################
    if isinstance(args.method, methods.Finetune):  # FT based on other method models
        if args.method.mode == 'pretrained_IMM':
            tr_method_name = methods.IMM(mode='train').name
        else:
            raise NotImplementedError()
    else:
        tr_method_name = args.method.name

    args.tr_results_path = utils.get_train_results_path(args.tr_results_root_path, args.tr_ds, tr_method_name,
                                                        model=args.base_model.name, grid_name=args.gridsearch_name)
    # Iterate all gridsearch paths if no specified
    exp_dirs = utils.get_immediate_subdirectories(args.tr_results_path) \
        if args.exp_name is None else [args.exp_name]

    if isinstance(args.method, methods.LocallyAdaptive):
        args.method.set_mode(args.LA_mode)
    elif isinstance(args.method, methods.IMM):
        args.method.set_mode(args.IMM_mode)

    args.method_name = args.method.eval_name
    print(args.method)

    ########################################
    # ITERATE EXPERIMENTS
    ########################################
    print("STARTING TESTING FOR METHOD: ", args.method_name)
    print("=> For dir: ", args.tr_results_path)
    for exp_dir in exp_dirs:  # Iterate experiments in gridsearch
        print("Testing for exp: ", exp_dir)
        try:
            args.exp_dir = exp_dir
            args.outgrid = args.gridsearch_name if args.outgrid is None else args.outgrid
            args.out_path = utils.get_test_results_path(args.test_results_root_path, args.eval_ds,
                                                        method=args.method_name, model=args.base_model.name,
                                                        gridsearch_name=args.outgrid, exp_name=exp_dir,
                                                        create=True,
                                                        merge_subset_idxs=args.user_merge_subset_idxs,
                                                        adaBN=args.adaBN,
                                                        adaBNTUNE=args.adaBNTUNE)
            # Iterate over users
            args = iterate_users(args)

            if len(args.all_iw_paths) > 0:
                return args.all_iw_paths
        except Exception as e:
            print(e)
            traceback.print_exc()
            print("Break exp {}".format(exp_dir))


def iterate_users(args):
    args.users_perf = None
    args.skip_merging = False
    args.store_iws = False
    args.unmerged_model_paths = None

    args.all_iw_paths = {}
    for user in range(1, args.eval_ds.usercount + 1):
        args.user = user
        args = collect_user_paths(args)  # Init paths

        # Performance save init
        user_out_filepath = os.path.join(args.out_path,
                                         utils.get_perf_output_filename(args.method_name, user=user))

        # Init
        args.users_perf = {} if args.users_perf is None else args.users_perf  # Append prev results
        if ('user' + str(args.user)) not in args.users_perf:
            args.users_perf['user' + str(args.user)] = {}

        # EVAL MODE
        if args.scriptmode == 'eval':
            if not args.debug and (not args.overwrite_mode and os.path.exists(user_out_filepath)):
                print("[OVERWRITE=False] SKIPPING, Already exists: ", user_out_filepath)
                continue

            for user_it in range(1, args.iterations + 1):  # Multiple iterations
                args.user_it = user_it
                args = eval_iteration(args)

            # Save user results
            if not args.debug:
                torch.save(args.users_perf, user_out_filepath)
                print("Saved exp results to: ", user_out_filepath)

        # ANALYZE MODELS MODE
        elif args.scriptmode == 'analyze_models':
            args = preprocessing(args)  # Init paths
            analyze_user_models(args)

        # PLOT IWS MODE
        elif args.scriptmode == 'plot_iws':
            args.store_iws = True
            args.skip_merging = True
            args = preprocessing(args)  # Init paths
            args.all_iw_paths[args.user] = args.iw_paths

    return args


def eval_iteration(args):
    """ One iteration to evaluate on the user data. User data is reassigned randomly."""
    print("*" * 40, " ITERATION {} ".format(args.user_it), "*" * 40)
    # Renew user allocated imgs
    if hasattr(args.eval_ds, 'reassign_user_paths'):
        args.eval_ds.reassign_user_paths(seed=args.user_it)
    else:
        if args.user_it > 1:  # Numbers dataset has fixed allocation to users.
            print("[WARN] SKIPPING iteration, as no randomness introduced.")
            return args

    # Merged models from IWs
    args.model_paths = args.model_paths if args.unmerged_model_paths is None else args.unmerged_model_paths  # Restore
    args.unmerged_model_paths = copy.deepcopy(args.model_paths)  # Backup for next iteration
    args = preprocessing(args)  # Init paths

    # Eval
    args.user_perf = eval_user(args)
    args.users_perf['user' + str(args.user)]['it' + str(args.user_it)] = args.user_perf
    print("USER{} results: {}".format(args.user, args.user_perf))

    # CLEANUP MERGED MODELS

    models_del = []
    # Cleanup model_paths
    if args.cleanup_merged_models:
        models_del.extend(args.model_paths[1:])

    elif args.cleanup_FT_models:
        models_del.extend(args.model_paths)
    # Cleanup BN models
    if args.cleanup_BN_models:
        models_del.extend(args.BN_model_paths)

    for model_path in models_del:
        if isinstance(model_path, str) and os.path.exists(model_path):
            os.remove(model_path)

    del args.model_paths[:]
    if args.cleanup_BN_models:
        del args.BN_model_paths[:]

    return args


def collect_user_paths(args):
    """
    Collects all dataset and model paths for all tasks for a specific user.
    :param args: args.user is the target user
    :return:
    """
    print("\nUSER {}".format(args.user))

    ########################################
    # COLLECT DATASET/MODEL PATHS
    ########################################
    args.eval_ds_holder = []
    args.eval_ds_paths = []
    args.tr_ds_holder = []
    args.tr_ds_paths = []
    args.model_paths = []
    args.task_lengths = []
    default_model_name = 'best_model.pth.tar'
    model_name = default_model_name

    for dataset_index, do_eval in enumerate(args.tasks_to_eval):
        task_counter = dataset_index + 1
        task_name = args.eval_ds.get_taskname(task_counter)
        if not do_eval:
            print("SKIPPING TASK {} (idx {} not in defined subset {})"
                  .format(task_name, dataset_index, args.user_merge_subset_idxs))
            continue

        # ADD DATASET
        eval_path = args.eval_ds.get_imgfolder_path(task_name, train=False, user=args.user)
        tr_path = args.eval_ds.get_imgfolder_path(task_name, train=True)
        args.eval_ds_paths.append(eval_path)
        args.tr_ds_paths.append(tr_path)

        if args.preload_ds:
            eval = args.eval_ds.imgfolders[eval_path]
            tr = args.eval_ds.imgfolders[eval_path]
        else:
            eval = eval_path
            tr = tr_path

        args.eval_ds_holder.append(copy.deepcopy(eval))
        args.tr_ds_holder.append(copy.deepcopy(tr))

        if len(args.task_lengths) < args.eval_ds.task_count:
            args.task_lengths.append(len(args.eval_ds.cat_per_task[task_name]))

        # ADD MODEL
        if isinstance(args.method, methods.Joint):
            task_dir = 'JOINT_BATCH'
        else:
            task_dir = 'task_' + str(task_counter)
        task_exp_model = os.path.join(args.tr_results_path, args.exp_dir, task_dir, 'TASK_TRAINING', model_name)
        args.model_paths.append(task_exp_model)

    if args.check_pretrained:
        print("PRETRAINED MODE")
        base_model = models.parse_model_name(args.models_root_path, args.model_name, args.eval_ds.input_size)
        args.model_paths = [base_model.path]

    print("\nCOLLECTED USER DATASETS:")
    print('\n'.join(args.eval_ds_paths))

    print("\nCOLLECTED USER MODELS:")
    print('\n'.join(map(str, args.model_paths)))

    return args


def preprocessing(args):
    """ IW calculation/ user-specific finetuning/ BN models"""

    # IWs
    if isinstance(args.method, methods.LocallyAdaptive) or isinstance(args.method, methods.IMM):
        args = merge_preprocessing(args)

    # finetuning preprocessing train
    elif isinstance(args.method, methods.Finetune):
        args.cleanup_FT_models = True
        args = finetune_preprocessing(args)

    # BN preprocessing
    if args.BN:
        if args.adaBN:  # adaptive task-aware BN
            assert not isinstance(args.method, methods.Finetune), "FT already obtains stats in training."
            args.cleanup_BN_models = True
            orig_model_paths = args.model_paths
            args = finetune_preprocessing(args, adaBN=True)
            args.BN_model_paths = args.model_paths
            args.model_paths = orig_model_paths
        elif args.adaBNTUNE:  # adaptive task-aware BN
            assert not isinstance(args.method, methods.Finetune), "FT already obtains stats in training."
            args.cleanup_BN_models = True
            orig_model_paths = args.model_paths
            args = finetune_preprocessing(args, adaBNTUNE=True)
            args.BN_model_paths = args.model_paths
            args.model_paths = orig_model_paths
        else:  # Regular task-aware BN
            args.BN_model_paths = args.model_paths

        print("\nBN MODELS:")
        print('\n'.join(map(str, args.BN_model_paths)))

    print("\nAFTER PREPROCESSING USER DATASETS:")
    print('\n'.join(args.eval_ds_paths))

    print("\nAFTER PREPROCESSING USER MODELS:")
    print('\n'.join(map(str, args.model_paths)))

    return args


def merge_preprocessing(args, overwrite=False):
    """ Merge the models and save them as separate merged models. Alter args model paths accordingly."""
    args.iw_user = args.user if args.method.user_specific else None

    # IW data
    if isinstance(args.method, methods.LocallyAdaptive):
        args.iw_ds_holder = args.eval_ds_holder
        args.ds_subsets = ['iw']

    elif isinstance(args.method, methods.IMM):
        print("-" * 40)
        print("IMM preprocessing: '{}' mode".format(args.method.mode))
        if args.IMM_mode == 'mode' \
                or args.IMM_mode == 'mode_MAS':
            args.iw_ds_holder = args.tr_ds_paths
            args.ds_subsets = ['train', 'val']
        else:
            raise Exception("UNKNOWN IMM mode: {}".format(args.IMM_mode))

    # Merge
    args.task_agnostic = 'task_agnostic' in args.method.mode
    args.merge_target_models = args.model_paths
    print("\nDATASETS FOR MERGING (IWS, task_agnostic={}):\n{}".format(args.task_agnostic,
                                                                       '\n'.join(map(str, args.iw_ds_holder))))
    print("\nMODELS FOR MERGING:\n{}".format('\n'.join(map(str, args.merge_target_models))))
    args.model_paths = IMM.merging_preprocess(args, overwrite=overwrite, cuda=args.cuda)
    args.cleanup_merged_models = True
    print("-" * 40)
    return args


def finetune_preprocessing(args, adaBN=False, adaBNTUNE=False, task_models=None):
    """ Start from trained models and train some iterations on user IW data. """
    args.model_paths = []
    base_model_name = 'best_model.pth.tar'

    for dataset_index, do_eval in enumerate(args.tasks_to_eval):
        task_counter = dataset_index + 1
        task_name = args.eval_ds.get_taskname(task_counter)
        if not do_eval:
            print("SKIPPING TASK {} (idx {} not in defined subset {})"
                  .format(task_name, dataset_index, args.user_merge_subset_idxs))
            continue

        print("")
        # ADD MODEL
        if isinstance(args.method, methods.Joint):
            task_dir = 'JOINT_BATCH'
        else:
            task_dir = 'task_' + str(task_counter)

        if task_models is None:
            base_task_exp_model = os.path.join(args.tr_results_path, args.exp_dir, task_dir, 'TASK_TRAINING',
                                               base_model_name)
        else:
            base_task_exp_model = task_models[dataset_index]

        tune_model_dirname = utils.get_FT_model_dirname(task_counter, args.user, args.user_it)
        super_dir = 'adaBN_MODELS' if adaBN or adaBNTUNE else 'FINETUNED'
        tune_task_exp_dir = os.path.join(args.tr_results_path, args.exp_dir, task_dir, super_dir, tune_model_dirname)

        tr_args = type("Arguments", (object,), {})()
        if adaBN:
            tr_args.epochs = 1  # Only need statistics
            tr_args.lr = 0.001
            tr_args.bs = 20
            tr_args.weight_decay = 0
            tr_args.model_epoch_saving_freq = 50
        elif adaBNTUNE:
            tr_args.epochs = 5  # Give BN params the chance to adapt
            tr_args.lr = 0.001
            tr_args.bs = 20
            tr_args.weight_decay = 0
            tr_args.model_epoch_saving_freq = 50
        else:
            tr_args.epochs = 10
            tr_args.lr = 0.0005
            tr_args.bs = 20
            tr_args.weight_decay = 0
            tr_args.model_epoch_saving_freq = 50

        manager = Manager()
        if isinstance(args.method, methods.Joint):
            if dataset_index == 0:
                manager.dataset_path = [
                    args.eval_ds.get_imgfolder_path(args.eval_ds.get_taskname(task_counter),
                                                    train=False, user=args.user)
                    for task_counter in range(1, args.eval_ds.task_count + 1)
                ]
                preload_dsets = [args.eval_ds.imgfolders[ds_path] for ds_path in manager.dataset_path] \
                    if args.preload_ds else None
                manager.merge_datasets(args, dsets=preload_dsets, subset_mapping=['iw', 'eval'])
            else:
                # Only need one model BN stats on all data
                break
        else:
            manager.dataset_path = args.eval_ds.get_imgfolder_path(task_name, train=False, user=args.user)
            preload_dsets = args.eval_ds.imgfolders[manager.dataset_path] if args.preload_ds else None
            manager.load_datasets(tr_args, dsets=preload_dsets, subset_mapping=['iw', 'eval'])  # Train on IW data

        manager.exp_dir = tune_task_exp_dir
        manager.previous_task_model_path = base_task_exp_model

        # LWF patch
        lwf_head = dataset_index if isinstance(args.method, methods.LWF) and dataset_index > 0 else None

        # train
        post_model, acc = methods.Finetune().train_task(manager, tr_args, adaBN=adaBN, adaBNTUNE=adaBNTUNE,
                                                        replace_head=False, save_models_mode=False, lwf_head=lwf_head)

        # No grads needed
        try:
            for param in post_model.parameters():
                param.requires_grad = False
        except:
            try:
                for param in post_model.model.parameters():
                    param.requires_grad = False
            except:
                pass

        torch.cuda.empty_cache()

        tune_task_exp_model = os.path.join(tune_task_exp_dir, base_model_name)
        args.model_paths.append(post_model)
        print("FINETUNED MODEL {}".format(map(str, tune_task_exp_model)))
    return args


def analyze_user_models(args):
    """
    Sanity check of reg params on the selected models.
    """
    # ANALYZE MODE OF MODELS ONLY
    print('-o' * 20, " Analyzing models ", '-o' * 20)
    for model_path in args.model_paths:
        print("\n MODEL: {}".format(model_path))
        model = torch.load(model_path)
        try:
            MAS.sanitycheck(model, include_bias=False)
        except:
            print("No analyzing possible for: {}", model_path)


def eval_user(args):
    ########################################
    # TESTING
    ########################################
    print("Testing on ", len(args.eval_ds_holder), " task datasets")

    with torch.no_grad():
        if isinstance(args.method, methods.Joint):
            user_perf = eval_single_model_all_tasks(args)
        else:
            user_perf = eval_all_models_all_tasks(args)
    print("FINISHED testing for user: ", args.user)

    return user_perf


def eval_single_model_all_tasks(args):
    """ Joint: Each acc result is a single value of the final model."""
    dataset = args.eval_ds

    # Get correct mapping to outputs from training setup
    # Merge to 1 batch of training data
    manager = Manager()
    manager.dataset_path = [
        dataset.get_imgfolder_path(args.tr_ds.get_taskname(ds_task_counter), train=True)
        for ds_task_counter in range(1, dataset.task_count + 1)]
    manager.merge_datasets(args)
    cummulative_classes_len = manager.dset_loaders['train'].dataset.cumulative_classes_len

    acc_all = {}
    try:
        ds = args.eval_ds_holder

        # Load model
        model = args.model_paths[0]
        if isinstance(model, str):
            model = torch.load(model)

        # BN: replace original model stats with adaBN stats
        if args.BN_model_paths is not None:
            if isinstance(args.BN_model_paths[0], str):
                BN_model = torch.load(args.BN_model_paths[0])
            else:
                BN_model = args.BN_model_paths[0]
            test_seq.replace_BN_params(BN_model, model)

        # All task results
        for dataset_index, do_eval in enumerate(args.tasks_to_eval):
            task_name = dataset.get_taskname(dataset_index + 1)
            if not do_eval:
                print("SKIPPING TASK {} (idx {} not in defined subset {})"
                      .format(task_name, dataset_index, args.user_merge_subset_idxs))
                continue

            acc = test_seq.test_task_joint_model(model, ds[dataset_index], dataset_index,
                                                 cummulative_classes_len, dataset.cat_per_task[task_name],
                                                 batch_size=args.bs)
            # Collect results
            acc_all[dataset_index] = acc

        print("FINAL RESULTS: ", acc_all)

    except Exception as e:
        print("TESTING ERROR: ", e)
        print("No results saved...")
        traceback.print_exc()
        raise Exception(e)

    return {'acc': acc_all, 'forgetting': []}


def eval_all_models_all_tasks(args):
    """Each acc/forgetting result is a sequence, testing per task."""
    acc_all = {}
    forgetting_all = {}
    ds = args.eval_ds_holder
    dataset = args.eval_ds

    # All task results
    eval_idx = 0
    for dataset_index, do_eval in enumerate(args.tasks_to_eval):
        task_name = dataset.get_taskname(dataset_index + 1)

        if not do_eval:
            print("SKIPPING TASK {} (idx {} not in defined subset {})"
                  .format(task_name, dataset_index, args.user_merge_subset_idxs))
            continue

        # Test
        try:
            seq_acc, seq_forgetting, seq_head_acc = test_seq.eval_task_modelseq(args.model_paths, ds,
                                                                                dataset_index, eval_idx, args.method,
                                                                                dataset.cat_per_task[task_name],
                                                                                batch_size=args.bs,
                                                                                debug=True,
                                                                                task_lengths=args.task_lengths,
                                                                                BN_model_paths=args.BN_model_paths)

            if len(seq_acc[dataset_index]) == 0:
                msg = "SKIPPING SAVING: acc empty: ", seq_acc[dataset_index]
                print(msg)
                raise Exception(msg)

            # Collect results
            acc_all[dataset_index] = seq_acc[dataset_index]
            forgetting_all[dataset_index] = seq_forgetting[dataset_index]
            eval_idx += 1

        except Exception as e:
            print("TESTING ERROR: ", e)
            traceback.print_exc()
            raise Exception(e)

    return {'acc': acc_all, 'forgetting': forgetting_all}


if __name__ == "__main__":
    main()
