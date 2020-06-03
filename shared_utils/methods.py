from abc import ABCMeta, abstractmethod

import train.EWC.Finetune_EWC
import train.IMM.Finetune_l2transfer
import train.Finetune.Finetune_SGD
import train.MAS.MAS
import train.LwF.Finetune_SGD_LWF
import shared_utils.utils as utils

method_names = ["LA", "EWC", "IMM", "MAS", 'LWF', 'JOINT', 'FT']


def parse(method_name, mode=None):
    if method_name == LocallyAdaptive.eval_name:
        return LocallyAdaptive(mode)
    elif method_name == LWF.name:
        return LWF()
    elif method_name == EWC.name:
        return EWC()
    elif method_name == IMM.name:
        return IMM(mode)
    elif MAS.name in method_name:
        if method_name == "MAS_afterSoftmax":
            return MAS(mode="afterSoftmax")
        elif method_name == "MAS_GT":
            return MAS(mode="ground_truth_only")
        else:
            return MAS(mode=None)
    elif method_name == Joint.name:
        return Joint()
    elif method_name == Finetune.alias or method_name == Finetune.name:
        return Finetune()
    else:
        raise NotImplementedError("Method not yet parseable")


class Method(metaclass=ABCMeta):
    @property
    @abstractmethod
    def name(self): pass

    @property
    @abstractmethod
    def eval_name(self): pass

    @property
    @abstractmethod
    def hyperparams(self): pass

    @property
    @abstractmethod
    def user_specific(self): pass

    @abstractmethod
    def train_task(self, manager, args): pass


# BASELINES
class EWC(Method):
    name = "EWC"
    eval_name = name
    hyperparams = ['lmb']
    user_specific = False

    def train_task(self, manager, args):
        model, acc = train.EWC.Finetune_EWC.fine_tune_EWC_acuumelation(
            manager,
            previous_task_model_path=manager.previous_task_model_path,
            exp_dir=manager.exp_dir,
            data_dir=manager.data_dir,
            reg_sets=manager.reg_sets,
            reg_lambda=args.lmb,
            batch_size=args.bs,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_epoch_saving_freq=args.model_epoch_saving_freq)
        return model, acc


class MAS(Method):
    name = "MAS"
    eval_name = name
    hyperparams = ['lmb']
    user_specific = False
    modes = [None, 'afterSoftmax', 'ground_truth_only']

    def __init__(self, mode=None):
        assert mode in self.modes, "MAS mode unkown: {} not in {}".format(mode, self.modes)
        self.mode = mode

        if self.mode is not None:
            self.name += "_" + self.mode
            self.eval_name = self.name

    def train_task(self, manager, args):
        model, acc = train.MAS.MAS.fine_tune_objective_based_acuumelation(
            manager,
            init_model_path=manager.init_model_path,
            data_dir=manager.data_dir,
            reg_lambda=args.lmb,
            batch_size=args.bs,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            lr=args.lr, norm='L2', b1=False,
            model_epoch_saving_freq=args.model_epoch_saving_freq,
            mode=self.mode)
        return model, acc


class LWF(Method):
    name = "LWF"
    eval_name = name
    hyperparams = ['lmb']
    user_specific = False

    def train_task(self, manager, args):
        model, acc = train.LwF.Finetune_SGD_LWF.fine_tune_SGD_LwF(
            manager,
            init_model_path=manager.init_model_path,
            reg_lambda=args.lmb,
            weight_decay=args.weight_decay,
            num_epochs=args.epochs,
            lr=args.lr,
            init_freeze=0,
            model_epoch_saving_freq=args.model_epoch_saving_freq,
        )
        return model, acc


class IMM(Method):
    """
    modes:
        train: not eval mode
        mean: mean-IMM
        mode: mode-IMM with EWC IWs
        mode_MAS: mode-IMM with MAS IWs
    """
    name = "IMM"
    eval_name = name
    base_name = name
    modes = ['train', 'mean', 'mode', 'mode_MAS']
    user_specific = False

    hyperparams = ['lmbL2trans', 'alpha']

    def __init__(self, mode):
        """Only difference is in testing, in training mode/mean IMM are the same"""
        mode = 'train' if mode is None else mode
        assert mode in self.modes, "mode '{}' not in defined modes '{}'".format(mode, self.modes)
        self.set_mode(mode)

    def set_mode(self, mode):
        """
        Set the IMM mode (mean and mode), this is only required after training.
        :param mode:
        :return:
        """
        if mode not in self.modes:
            raise Exception("TRY TO SET NON EXISTING MODE '{}', not in '{}'".format(mode, self.modes))
        self.mode = mode
        self.eval_name = self.base_name + "_" + self.mode
        print("RUNNING IMM({}) in '{}' mode".format(self.name, self.mode))

    def get_merged_model_name(self, ds_name, user=None):
        if user is None:
            return 'modelmerge_{}_DS={}_id={}.pth.tar'.format(self.mode, ds_name, utils.get_unique_id())
        else:
            return 'modelmerge_user{}_{}_DS={}_id={}.pth.tar'.format(user, self.mode, ds_name, utils.get_unique_id())

    def get_iw_name(self, ds_name, user=None, sum=False, task_agnostic=False):
        if user is None:
            name = 'IW_{}_DS={}_TA={}_id={}.pth.tar'.format(self.mode, ds_name, task_agnostic, utils.get_unique_id())
        else:
            name = 'IW_user{}_{}_DS={}_TA={}_id={}.pth.tar'.format(user, self.mode, ds_name, task_agnostic,
                                                                   utils.get_unique_id())

        if sum:
            name = 'sum_' + name
        return name

    def train_task(self, manager, args):
        model, acc = train.IMM.Finetune_l2transfer.fine_tune_l2transfer(
            manager,
            model_path=manager.previous_task_model_path,
            exp_dir=manager.exp_dir,
            reg_lambda=args.lmb,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            model_epoch_saving_freq=args.model_epoch_saving_freq)
        return model, acc


class LocallyAdaptive(IMM):
    """
    LA exploits IMM training, only merging step is altered with MAS iws.
    """
    base_modes = ['plain', 'FIM']
    modes = base_modes
    eval_name = "LA"
    base_name = eval_name
    user_specific = True

    def __init__(self, mode):
        mode = 'plain' if mode is None else mode
        super().__init__(mode)


class Finetune(Method):
    alias = 'FT'
    name = "finetuning"
    eval_name = name
    hyperparams = []
    user_specific = False
    modes = ['pretrained_IMM']  # Finetune the IMM models with user data
    mode = modes[0]

    def train_task(self, manager, args, adaBN=False, replace_head=True, save_models_mode=True, lwf_head=None,
                   adaBNTUNE=False):
        deactivate_bias = args.deactivate_bias if hasattr(args, 'deactivate_bias') else False

        model, acc = train.Finetune.Finetune_SGD.fine_tune_SGD(
            manager,
            model_path=manager.previous_task_model_path,
            exp_dir=manager.exp_dir,
            num_epochs=args.epochs, lr=args.lr,
            weight_decay=args.weight_decay,
            replace_head=replace_head,
            model_epoch_saving_freq=args.model_epoch_saving_freq,
            adaBN=adaBN,
            save_models_mode=save_models_mode,
            lwf_head=lwf_head,
            deactivate_bias=deactivate_bias,
            adaBNTUNE=adaBNTUNE)
        return model, acc


class Joint(Method):
    name = "JOINT"
    eval_name = name
    hyperparams = []
    user_specific = False

    def train_task(self, manager, args):
        """ Do Finetune, but with all task data in 1 batch."""
        return Finetune().train_task(manager, args)
