import os
import traceback
import numpy as np
import copy

import torch

import shared_utils.utils as utils
from shared_utils.methods import IMM, Finetune, Joint


class ExperimentDataEntry(object):
    """Experiment data and plot config, including all users."""

    def __init__(self, exp_name, exp_parent_path, dataset, method, model, label=None, label_prefix=None,
                 between_head_acc=False, user=None,
                 copy_exp=None):
        if user is not None and copy_exp is not None:
            self.make_user_exp_entry(user, copy_exp)
        else:
            self.exp_name = exp_name
            self.exp_parent_path = exp_parent_path  # Test_paths
            self.exp_path = os.path.join(exp_parent_path, exp_name)

            self.dataset = dataset
            self.method = method
            self.model = model
            self.label = '_'.join([label_prefix, exp_name]) if label is None else label

            # Iterations
            self.it_seq_acc = {}
            self.it_seq_forgetting = {}
            self.it_seq_acc_ref = {}  # seq_acc but caclulated on difference towards ref results

            # Avg seqs
            self.seq_acc = {}  # For exp: avges over all users; for 1 user: avges over all its
            self.seq_acc_std = {}  # Std of iterations over users or iterations
            self.seq_acc_ref_std = {}  # Std wrt reference results
            self.seq_forgetting = {}
            self.seq_forgetting_std = {}
            self.final_model_seq_test_acc = []

            # User Avg (over all user avges)
            self.avg_acc = 0
            self.avg_acc_var = 0  # summing variances over both tasks and users for final model acc
            self.avg_forgetting = 0
            self.avg_forgetting_var = 0

            # self.userstyles =
            self.linestyle = self.get_linestyle_config(method)
            self.marker, self.markersize = '1', 3
            self.users = {user: ExperimentDataEntry(exp_name, exp_parent_path, dataset, method, model,
                                                    user=user, copy_exp=self)
                          for user in range(1, dataset.usercount + 1)}
            self.raw_user_results = None

            # STYLES
            self.between_head_acc = between_head_acc  # Plot between-head acc instead of normal acc

            print("MADE EXP ENTRY: {}".format(self.label))

    def make_user_exp_entry(self, user, copy_exp):
        """ Used to make exp entry for specific user."""
        attributes = [a for a in dir(copy_exp) if not a.startswith('__')
                      and not callable(getattr(copy_exp, a))
                      and a != 'users']
        for attr in attributes:
            setattr(self, attr, copy.deepcopy(getattr(copy_exp, attr)))
        self.label = 'U{}'.format(user) + ":" + copy_exp.label
        print("MADE USER ENTRY: {}".format(self.label))

    def __str__(self):
        return ', '.join([self.label, self.dataset.name, self.method.name, self.model.name, self.exp_name])

    def create_label(self, label_prefix, exp_name_segment_idxs, method, join_arg=','):
        if label_prefix is None:
            label_prefix = [self.method.name, self.model.name]
        elif not isinstance(label_prefix, list):
            label_prefix = [label_prefix]
        label_suffix = self.exp_name.split("_")
        label_segments = label_prefix + label_suffix
        if exp_name_segment_idxs is not None:
            label_segments = [label_segment for idx, label_segment in enumerate(label_segments)
                              if idx in exp_name_segment_idxs and idx < len(label_segments)]
        label = join_arg.join(label_segments)

        # Not using heuristic
        if method.name == IMM.name or method.name is Joint.name:
            label += "*"

        return label

    def get_linestyle_config(self, method):
        """linestyles = ['-', '--', '-.', ':']"""
        if method.name == Finetune.name:
            return ':'
        return '-'


def avg_exp(exp_data_entries):
    ref_exp = copy.deepcopy(exp_data_entries[0])
    ref_exp.label = 'AVG_' + ref_exp.label
    for exp in exp_data_entries[1:]:
        ref_exp.avg_acc += copy.deepcopy(exp.avg_acc)
        ref_exp.avg_forgetting += copy.deepcopy(exp.avg_forgetting)

    ref_exp.avg_acc /= len(exp_data_entries)
    ref_exp.avg_forgetting /= len(exp_data_entries)

    return ref_exp


def select_users(exp_data_entries, users):
    if users is None:
        return exp_data_entries

    assert isinstance(users, list)
    ret_entries = []
    for exp_data_entry in exp_data_entries:
        for user in users:
            ret_entries.append(exp_data_entry.users[user])
    return ret_entries


def collect_grid_exps(holder, gridname, exp_names=None, exp_name_contains=None, exp_name_excludes=None,
                      merge_subset_idxs=None, adaBN=False, adaBNTUNE=False):
    """
    Iterate over exps in the gridsearch and possibly filter to subset.
    :param label_prefix: Prefix of the label (exp_name is suffix), default: <method.name>_<model.name>
    :param label_segment_idxs: Which segments of the label to keep. (On Split)
    :param curve_shape:
    :param task_agnostic_mode: True if task agnostic eval experiment
    :param between_head_acc: plot between_head_acc
    :return:
    """

    exp_dir = utils.get_test_results_path(holder.test_results_root_path, holder.dataset,
                                          method=holder.method.eval_name, model=holder.model.name,
                                          gridsearch_name=gridname,
                                          merge_subset_idxs=merge_subset_idxs,
                                          adaBN=adaBN,
                                          adaBNTUNE=adaBNTUNE)

    # No exp defined, collect all from gridsearch
    if exp_names is None:
        exp_names = utils.get_immediate_subdirectories(exp_dir, path_mode=False, sort=True)
    else:
        if isinstance(exp_names, list):
            exp_names = [x.strip() for x in exp_names]
        else:
            exp_names = [exp_names.strip()]

    # Make subselection based on name
    if exp_name_contains is not None:
        if exp_name_excludes is not None:
            exp_names = [experiment_name for experiment_name in exp_names
                         if exp_name_contains in experiment_name
                         and exp_name_excludes not in experiment_name]
        else:
            exp_names = [experiment_name for experiment_name in exp_names
                         if exp_name_contains in experiment_name]
    elif exp_name_excludes is not None:
        exp_names = [experiment_name for experiment_name in exp_names
                     if exp_name_excludes not in experiment_name]

    experiment_data_entries = [ExperimentDataEntry(exp_name, exp_dir, holder.dataset, holder.method, holder.model,
                                                   label=holder.label, label_prefix=holder.label_prefix)
                               for idx, exp_name in enumerate(exp_names)]
    return experiment_data_entries


def format_results(exp_data_entries, merge_subset_idxs=None):
    """Iterate exps and fill with results."""

    max_task_count = 0
    for exp_idx, exp_data_entry in enumerate(exp_data_entries[:]):
        print("preprocessing experiment: ", exp_data_entry)
        try:
            exp_target = exp_data_entries[exp_idx]
            joint_mode = True if isinstance(exp_data_entry.method, Joint) else False  # single perf file
            if exp_data_entry.dataset.task_count > max_task_count:
                max_task_count = exp_data_entry.dataset.task_count

            tasks = list(range(1, exp_data_entry.dataset.task_count + 1)) if merge_subset_idxs is None \
                else [task_idx + 1 for task_idx in merge_subset_idxs]
            final_model_idx = 0 if isinstance(exp_target.method,
                                              Finetune) else -1  # FT has user-specific model learned (first)

            # Iterate tasks
            for user in range(1, exp_target.dataset.usercount + 1):
                user_target = exp_target.users[user]
                user_acc_filename = utils.get_perf_output_filename(exp_data_entry.method.eval_name, user=user)
                user_results_file = os.path.join(exp_data_entry.exp_path, user_acc_filename)

                for task_count in tasks:
                    task_idx = task_count - 1

                    user_perf = torch.load(user_results_file)  # LOAD EVAL RESULTS
                    user_perf = user_perf['user{}'.format(user)]
                    exp_target.raw_user_results = user_perf

                    metric_dict_key = 'acc'  # ACC
                    if exp_data_entry.between_head_acc:
                        metric_dict_key = 'head_acc'

                    ####################################
                    # AVG over iterations (for user)
                    for it_key, it_perf in user_perf.items():
                        eval_results = it_perf[metric_dict_key]  # {task_idx:[results]}
                        if joint_mode:
                            eval_results = reformat_single_sequence(eval_results, task_idx)
                        add_iteration(user_target, eval_results, task_count)  # Collect iteration results from user
                    # Calculate metrics over iterations
                    avg_iterations(user_target, task_count, final_model_idx)

                    ####################################
                    # AVG over users (for experiment): Sum users per task
                    add_iteration(exp_target, user_target.seq_acc, task_count)

                # Divide sums to get avges, but variances remain summed
                user_target.avg_acc = user_target.avg_acc / len(tasks)
                user_target.avg_forgetting = user_target.avg_forgetting / len(tasks)
                exp_target.avg_acc_var += user_target.avg_acc_var  # Experiment uncertainty on final acc

            # Collect avg per task for total exp
            for task_count in tasks:
                avg_iterations(exp_target, task_count, final_model_idx)

            exp_target.avg_acc = exp_target.avg_acc / len(tasks)
            exp_target.avg_forgetting = exp_target.avg_forgetting / len(tasks)
        except Exception:
            print("LOADING performance ERROR: REMOVING FROM PLOT EXPS")
            del exp_target
            exp_data_entries[exp_idx] = None
            traceback.print_exc(5)

    exp_data_entries = [exp_data_entry for exp_data_entry in exp_data_entries if exp_data_entry is not None]
    return exp_data_entries, max_task_count


def add_iteration(target, it_eval_results, task_count, sum_mode=False, replace_mode=False):
    """All results for testing on 1 task, collected in target object.
        task_count: always gather results for specific task.
        sum_mode: sum the new results for the given key.
    """
    idx = task_count - 1

    # Accuracy
    if sum_mode:
        target.it_seq_acc[idx] = it_eval_results[idx] if idx not in target.it_seq_acc \
            else [sum(x) for x in zip(it_eval_results[idx], target.it_seq_acc[idx])]
    else:
        if replace_mode:
            target.it_seq_acc[idx] = it_eval_results[idx]
        else:
            target.it_seq_acc[idx] = [] if idx not in target.it_seq_acc else target.it_seq_acc[idx]
            target.it_seq_acc[idx].append(it_eval_results[idx])

    # Forgetting
    if len(it_eval_results[idx]) > 1:
        it_forg = [it_eval_results[idx][0] - task_res for task_res in it_eval_results[idx][1:]]
    else:
        it_forg = []

    if sum_mode:
        target.it_seq_forgetting[idx] = it_forg if idx not in target.it_seq_forgetting \
            else [sum(x) for x in zip(it_forg, target.it_seq_forgetting[idx])]
    else:
        target.it_seq_forgetting[idx] = [] if idx not in target.it_seq_forgetting \
            else target.it_seq_forgetting[idx]
        target.it_seq_forgetting[idx].append(it_forg)


def avg_iterations(target, task_count, final_model_idx=-1):
    """Avg over the list of iterations."""
    task_idx = task_count - 1

    # avg/std acc
    acc_its = target.it_seq_acc[task_idx]  # All iterations
    try:
        target.seq_acc[task_idx] = np.mean(acc_its, axis=0)
        target.seq_acc_std[task_idx] = np.std(acc_its, axis=0)
    except:
        pass
    target.final_model_seq_test_acc.append(target.seq_acc[task_idx][final_model_idx])
    target.avg_acc += target.seq_acc[task_idx][final_model_idx]  # Final model acc on task
    target.avg_acc_var += np.var(acc_its, axis=0)[final_model_idx]  # Final model variance

    # avg/std forgetting
    forg_its = target.it_seq_forgetting[task_idx]  # All iterations
    target.seq_forgetting[task_idx] = np.mean(forg_its, axis=0)
    target.seq_forgetting_std[task_idx] = np.std(forg_its, axis=0)

    if len(target.seq_forgetting[task_idx]) > 0:
        target.avg_forgetting += target.seq_forgetting[task_idx][final_model_idx]
        target.avg_forgetting_var += np.var(forg_its, axis=0)[final_model_idx]  # Final model variance


def reformat_single_sequence(eval_results, dataset_index, plot_full_curve=False):
    """
    Each acc value is end result of testing on a task.
    :param exp_data_entry:
    :param eval_results:
    :param dataset_index:
    :return:
    """
    repeatings_for_curve = len(eval_results)
    if not plot_full_curve:
        repeatings_for_curve -= dataset_index
    extended_result = {dataset_index: [eval_results[dataset_index]] * repeatings_for_curve}
    return extended_result


def check_existing(exp_list):
    success = True
    for exp in exp_list:
        try:
            format_results(exp_list)
            print('EXP {}: success'.format(exp.label))
        except Exception as e:
            print('EXP {}: failed'.format(exp.label))
            # print(e)
            success = False
    return success


def set_joint_std(exp_data_entries, joint_entry):
    """ Std dev w.r.t. JOINT performance for all entries. """
    diff_copies = copy.deepcopy(exp_data_entries)
    for exp_idx, exp_data_entry in enumerate(diff_copies):
        avg_acc_var = 0
        # Sum over tasks
        for task_idx, iterations in exp_data_entry.it_seq_acc.items():
            # SUm variance over iterations
            for it_idx in range(len(iterations)):
                exp_data_entry.it_seq_acc[task_idx][it_idx] = [res - joint_entry.it_seq_acc[task_idx][it_idx][0]
                                                               for idx, res in
                                                               enumerate(exp_data_entry.it_seq_acc[task_idx][it_idx])]
            avg_acc_var += np.var(exp_data_entry.it_seq_acc[task_idx], axis=0)[-1]  # Final model variance
        # UPDATE
        exp_data_entries[exp_idx].joint_avg_acc_var = avg_acc_var
    return exp_data_entries
