""" Main training script iterating over task sequence. """
import os
import traceback
import torch
import argparse
import shutil

import shared_utils.utils as utils
import shared_utils.datasets as datasets
import shared_utils.models as models
import shared_utils.methods as methods
from shared_utils.ImageFolderTrainVal import ConcatDatasetDynamicLabels

parser = argparse.ArgumentParser(description='Locally Adaptive CL')

parser.add_argument('--model_name', choices=models.model_names, default="alexnet_pretrain")
parser.add_argument('--method_name', choices=methods.method_names, default=methods.EWC.name)
parser.add_argument('--ds_name', choices=datasets.dataset_names, default="MITindoorscenes")

# TRAIN
parser.add_argument('--bs', type=int, default=30, help="Batch size.")
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--seed', type=int, default=None, help="Seed.")
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--lr_pretrain', type=float, default=None,
                    help='Learning rate first task, when starting from scratch model.')

# DEFAULT SCRIPT VALUES
parser.add_argument('--model_epoch_saving_freq', type=int, default=30,
                    help="How frequently to store a model checkpoint (in terms of epochs)")
parser.add_argument('--gridsearch_name', type=str, default='grid',
                    help="Group experiments: <tr_results_root>/<gridsearch_name>/exp_name/")

# Special Run Modes
parser.add_argument('--debug', action="store_true", help="Debug Mode")
parser.add_argument('--disable_cudnn', action="store_true", help="Disable CUDA Mode")
parser.add_argument('--deactivate_bias', action="store_true", help="Deactivate Conv2d biases.")
parser.add_argument('--cleanup_exp', action="store_true", help="Restart experiment from scratch.")

# DEFAULT HYPERPARAM VALUES
parser.add_argument('--lmb', type=float, default=1, help="lambda")
parser.add_argument('--lmbL2trans', type=float, default=1, help="lambda for L2 transfer in IMM")
parser.add_argument('--alpha', type=float, default=1, help="Merging param IMM")


def main():
    ########################################
    # ARGS Postprocessing
    ########################################
    args = parser.parse_args()
    utils.set_random(seed=args.seed)

    # ADD EXTERNAL PROJECT PATHS
    config = utils.get_parsed_config()
    args.tr_results_root_path = utils.read_from_config(config, 'tr_results_root_path')
    args.models_root_path = utils.read_from_config(config, 'models_root_path')

    if args.debug:
        print('#' * 20, " RUNNING IN DEBUG MODE ", '#' * 20)
        args.epochs = 2

    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
        print("DISABLED CUDA")

    # SET DATASET
    dataset, _ = datasets.parse(args.ds_name)

    # METHOD
    args.method = methods.parse(args.method_name)

    # GET INIT MODELPATH
    args.pretrained = 'pretrained' not in args.model_name
    base_model = models.parse_model_name(args.models_root_path, args.model_name, dataset.input_size)
    print("RUNNING MODEL: ", base_model.name)

    # Store init LR
    args.lr_main = args.lr
    args.lr_pretrain = args.lr if args.lr_pretrain is None else args.lr_pretrain

    ########################################
    # Config Data and Model paths
    ########################################
    # Where specific training results go (specific tuning dir)
    args.exp_name = utils.get_exp_name(args, args.method)
    parent_exp_dir = utils.get_train_results_path(args.tr_results_root_path,
                                                  dataset, args.method_name,
                                                  model=base_model.name,
                                                  grid_name=args.gridsearch_name,
                                                  exp_name=args.exp_name)
    # INIT MODEL PATH
    previous_task_model_path = base_model.path
    if not os.path.exists(previous_task_model_path):
        raise Exception("NOT EXISTING previous_task_model_path = ", previous_task_model_path)
    else:
        print("Starting from model = ", previous_task_model_path)

    ########################################
    # Train each of the Tasks
    ########################################
    utils.set_random(seed=args.seed)

    # Manager
    manager = Manager(dataset, args.method, previous_task_model_path, parent_exp_dir, base_model)
    print("Starting with ARGS=", vars(args))

    # Remove previous exp results (e.g. form erroneous run)
    if args.cleanup_exp and os.path.isdir(manager.parent_exp_dir):
        shutil.rmtree(parent_exp_dir)
        print("=====> CLEANING UP EXP: Removing all to start from scratch <=====")

    if isinstance(args.method, methods.Joint):
        batch_tasks(manager, args, dataset)
    else:
        # Iterate Tasks
        iterate_tasks(manager, args, dataset)


def iterate_tasks(manager, args, dataset):
    # Iterate tasks
    for task_counter in range(1, dataset.task_count + 1):
        print("\n", "*" * 80)
        print("TRAINING Task ", task_counter)
        print("*" * 80)

        # PREP
        method = methods.Finetune() if task_counter == 1 else args.method
        args.lr = args.lr_pretrain if task_counter == 1 and args.pretrained else args.lr_main
        grid_name = None if task_counter == 1 else args.gridsearch_name
        prev_task_name = None if task_counter == 1 else dataset.get_taskname(task_counter - 1)
        task_name = manager.dataset.get_taskname(task_counter)
        args.exp_name = utils.get_exp_name(args, method)

        manager.parent_exp_dir = utils.get_train_results_path(
            args.tr_results_root_path, dataset, method.name,
            model=manager.base_model.name, grid_name=grid_name, exp_name=args.exp_name
        )
        manager.exp_dir = os.path.join(manager.parent_exp_dir, 'task_' + str(task_counter), 'TASK_TRAINING')
        manager.dataset_path = manager.dataset.get_imgfolder_path(task_name, train=True)
        manager.load_datasets(args)

        manager.reg_sets = [manager.dataset.get_imgfolder_path(prev_task_name, train=True)]
        print('manager=', vars(manager))

        # TRAIN
        try:
            if not os.path.exists(manager.get_success_token_path(manager.exp_dir)):
                model, acc = method.train_task(manager, args)
            else:
                print("Skipping task: already successfully finished.")
                print("Token at: {}".format(manager.get_success_token_path(manager.exp_dir)))

        except RuntimeError as e:
            print("[ERROR] training:", e)
            traceback.print_exc()
            break

        # POST PREP
        manager.previous_task_model_path = os.path.join(manager.exp_dir, 'best_model.pth.tar')
        manager.create_success_token(manager.exp_dir)
        chkpt_path = os.path.join(manager.exp_dir, 'epoch.pth.tar')
        if os.path.exists(chkpt_path):
            os.remove(chkpt_path)  # Cleanup chkpt
        if task_counter == 1:
            dest_dir = utils.get_train_results_path(args.tr_results_root_path, dataset, args.method.name,
                                                    model=manager.base_model.name, grid_name=args.gridsearch_name,
                                                    exp_name=utils.get_exp_name(args, args.method))
            dest_dir = os.path.join(dest_dir, 'task_' + str(task_counter), 'TASK_TRAINING')
            utils.create_symlink(manager.exp_dir, dest_dir)  # 1st task symlink


def batch_tasks(manager, args, dataset):
    """ Joint training on all task data in 1 batch. """

    method = args.method
    assert isinstance(method, methods.Joint)
    args.exp_name = utils.get_exp_name(args, method)

    manager.parent_exp_dir = utils.get_train_results_path(
        args.tr_results_root_path, dataset, method.name,
        model=manager.base_model.name, grid_name=args.gridsearch_name, exp_name=args.exp_name
    )
    manager.exp_dir = os.path.join(manager.parent_exp_dir, 'JOINT_BATCH', 'TASK_TRAINING')

    # Merge to 1 batch of data
    manager.dataset_path = [
        dataset.get_imgfolder_path(dataset.get_taskname(ds_task_counter), train=True)
        for ds_task_counter in range(1, dataset.task_count + 1)]
    manager.merge_datasets(args)
    print('manager=', vars(manager))

    # TRAIN
    if not os.path.exists(manager.get_success_token_path(manager.exp_dir)):
        model, acc = args.method.train_task(manager, args)
    else:
        print("Skipping task: already successfully finished.")
        print("Token at: {}".format(manager.get_success_token_path(manager.exp_dir)))

    # CLEANUP
    manager.create_success_token(manager.exp_dir)
    chkpt_path = os.path.join(manager.exp_dir, 'epoch.pth.tar')
    if os.path.exists(chkpt_path):
        os.remove(chkpt_path)  # Cleanup chkpt


class Manager(object):
    """ Holder object for single task required info."""
    token_name = 'SUCCESS.FLAG'

    def __init__(self, dataset=None, method=None, previous_task_model_path=None, parent_exp_dir=None, base_model=None):
        self.dataset = dataset
        self.method = method
        self.base_model = base_model

        # Paths
        self.previous_task_model_path = previous_task_model_path
        self.parent_exp_dir = parent_exp_dir
        self.exp_dir = None
        self.init_model_path = None
        self.data_dir = None

        # Dataset
        self.dataset_path = None
        self.dset_loaders = None
        self.dset_sizes = None
        self.dset_classes = None

    def save_hyperparams(self, output_dir, hyperparams):
        """
        Add extra stats (memory,...) and save to output_dir.

        :param output_dir: Dir to export the dictionary to.
        :param hyperparams: Dictionary with hyperparams to save
        :return:
        """

        utils.create_dir(output_dir)
        hyperparams_outpath = os.path.join(output_dir, utils.get_hyperparams_output_filename())
        torch.save(hyperparams, hyperparams_outpath)
        print("Saved hyperparams to: ", hyperparams_outpath)

    def get_success_token_path(self, exp_dir):
        """
        Creates a token for exp_dir, representing successful finishing of training.
        :param exp_dir:
        :return:
        """
        return os.path.join(exp_dir, self.token_name)

    def create_success_token(self, exp_dir):
        if not os.path.exists(self.get_success_token_path(exp_dir)):
            torch.save('', self.get_success_token_path(exp_dir))

    def load_datasets(self, args, dsets=None, subset_mapping=None):
        """
        :param subset_mapping: map e.g. [iw, eval] in the dataset to [train, val] in ds loader.
        """
        loader_subsets = ['train', 'val']

        if dsets is None:
            dsets = torch.load(self.dataset_path)
        if subset_mapping is None:
            subset_mapping = loader_subsets
        self.dset_loaders = {
            subs: torch.utils.data.DataLoader(dsets[subs_m], batch_size=args.bs, shuffle=True, num_workers=4)
            for subs, subs_m in zip(loader_subsets, subset_mapping)}
        self.dset_sizes = {subs: len(dsets[subs_m]) for subs, subs_m in zip(loader_subsets, subset_mapping)}

        if hasattr(dsets[subset_mapping[0]], 'classes'):
            self.dset_classes = dsets[subset_mapping[0]].classes
        else:
            self.dset_classes = dsets[subset_mapping[0]].dataset.classes

    def merge_datasets(self, args, dsets=None, subset_mapping=None):
        """ Merge list of datasetpaths into 1 dataset and init loaders."""
        assert isinstance(self.dataset_path, list), \
            "Can only merge pathlist for datasets, is not list: {}".format(self.dataset_path)
        loader_subsets = ['train', 'val']
        if subset_mapping is None:
            subset_mapping = loader_subsets
        [print("MAPPING {} to {}".format(subs, subs_m)) for subs, subs_m in zip(loader_subsets, subset_mapping)]
        dset_imgfolders = {x: [] for x in loader_subsets}
        dset_task_classes = {x: [] for x in loader_subsets}
        dset_sizes = {x: [] for x in loader_subsets}
        it = self.dataset_path if dsets is None else dsets
        for ds in it:
            if dsets is None:
                dset_wrapper = torch.load(ds)
            else:
                dset_wrapper = ds

            for subs, subs_m in zip(loader_subsets, subset_mapping):
                dset_imgfolders[subs].append(dset_wrapper[subs_m])
                dset_sizes[subs].append(len(dset_wrapper[subs_m]))
                if hasattr(dset_wrapper[subs_m], 'classes'):
                    dset_task_classes[subs].append(dset_wrapper[subs_m].classes)
                else:
                    dset_task_classes[subs].append(dset_wrapper[subs_m].dataset.classes)

        self.dset_sizes = {mode: sum(dset_sizes[mode]) for mode in dset_sizes}
        dset_class_lengths = {x: [len(classes) for classes in dset_task_classes[x]] for x in loader_subsets}

        assert dset_task_classes['train'] == dset_task_classes['val'], "Train and val should have same classes."
        self.dset_classes = []
        for classes in dset_task_classes['train']:
            self.dset_classes.extend(classes)

        # Concat into 1 dataset
        self.dset_loaders = {x: torch.utils.data.DataLoader(
            ConcatDatasetDynamicLabels(dset_imgfolders[x], dset_class_lengths[x]),
            batch_size=args.bs, shuffle=True, num_workers=4)
            for x in loader_subsets}

        print("dset_classes: ", self.dset_classes)
        print("dset_sizes: ", self.dset_sizes)


if __name__ == "__main__":
    main()
