import shared_utils.datasets as datasets
import shared_utils.methods as methods
import shared_utils.models as models
import shared_utils.utils as utils

import plot.plot as plot
import plot.format_data as format_data

# Demo methods
plot_mode_IMM = True
plot_LA = False

# Turn on/off
plot_joint = False
plot_FT = False
plot_LWF = False
plot_EWC = False
plot_MAS = False
plot_IMM_mode_MAS = False
plot_LA_FIM = False

# Plot all (None), or specific user (define in list, e.g. [1] for user 1)
users = None

#############################################
# INIT
method_names = []
pool_exp_entries = []
holder = type("Holder", (object,), {})()

config = utils.get_parsed_config()
holder.test_results_root_path = utils.read_from_config(config, 'test_results_root_path')
holder.models_root_path = utils.read_from_config(config, 'models_root_path')

# DATASET
# holder.dataset = datasets.IndoorScenesTransformDataset() # TransPrior
holder.dataset = datasets.IndoorScenesDataset()  # CatPrior

# MODEL
holder.model = models.VGG11(holder.models_root_path, (224, 224))

# CONFIG OPTIONS
holder.title = 'demo'
holder.save_img = True
holder.img_extention = 'png'  # 'eps' for latex
holder.hyperparams_selection = []
holder.plot_seq_acc = True
holder.plot_seq_forgetting = False
holder.include_test_table = True
holder.label_segment_idxs = []
holder.exp_name_contains = ''
holder.task_agnostic_mode = False
holder.softmax_mode = False
holder.label_prefix = None
holder.label = None

# Common to exps
test_grid = 'demo'
user_merge_subset_idxs = [t_idx for t_idx in range(0, holder.dataset.task_count)]

#############################################
# MODE IMM METHOD
if plot_mode_IMM:
    holder.method = methods.IMM(mode='mode')
    method_names.append(holder.method.eval_name)

    exp_names = None
    holder.label_prefix = 'FIM-IMM'

    exp_entries = format_data.collect_grid_exps(holder, test_grid, exp_names, merge_subset_idxs=user_merge_subset_idxs)
    pool_exp_entries.extend(exp_entries)

#############################################
# LACL METHOD
if plot_LA:
    holder.method = methods.LocallyAdaptive(mode='plain')
    method_names.append(holder.method.eval_name)

    exp_names = None  # Or select specific exp in list
    holder.label_prefix = 'MAS-LACL'

    exp_entries = format_data.collect_grid_exps(holder, test_grid, exp_names, merge_subset_idxs=user_merge_subset_idxs)
    pool_exp_entries.extend(exp_entries)

#############################################
# ANALYZE
#############################################
print(pool_exp_entries)
holder.save_img_name = None
if holder.save_img:
    holder.save_img_name = utils.get_img_outpath(holder.dataset.name, method_names, holder.model, title=holder.title)

exp_data_entries, max_task_count = format_data.format_results(pool_exp_entries)
exp_data_entries = format_data.select_users(exp_data_entries, users)  # make selection of users
plot.print_exp_statistics(exp_data_entries)
