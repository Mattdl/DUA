import numpy as np
import matplotlib.pyplot as plt

def print_exp_statistics(experiment_data_entries, table_sep='\t', plot_std=False):
    print()
    print("-" * 50)
    print("SUMMARY")
    print("-" * 50)

    header = ["'Method'", "'Avg acc(forg)'"]
    if plot_std:
        header.append("'std'")
    print(table_sep.join(header))
    for experiment_data_entry in experiment_data_entries:
        out = [str(experiment_data_entry.label)]
        out.append("{:.2f} ({:.2f})".format(experiment_data_entry.avg_acc, experiment_data_entry.avg_forgetting))
        if plot_std:
            out.append(r'%.2f' % (np.sqrt(experiment_data_entry.avg_acc_var)))
        out = table_sep.join(out)
        print(out)
        if 'AVG' in str(experiment_data_entry.label):
            print()

def imshow_tensor(inp, title=None, denormalize=True, ):
    """
    Imshow for Tensor.

    :param inp: input Tensor of img
    :param title:
    :param denormalize: denormalize input or not
    :return:
    """
    inp = tensor_to_img(inp, denormalize=denormalize)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)

    plt.pause(0.1)  # pause a bit so that plots are updated


def tensor_to_img(inp, denormalize=True,
                  mean=np.array([0.485, 0.456, 0.406]),
                  std=np.array([0.229, 0.224, 0.225])):
    inp = inp.cpu().numpy().transpose((1, 2, 0))
    if denormalize:
        inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    return inp


def plot_figures(figures, nrows=1, ncols=1, title=None, imgtitle=False):
    """Plot a dictionary of figures.

    Parameters
    ----------
    figures : <title, figure> dictionary
    ncols : number of columns of subplots wanted in the display
    nrows : number of rows of subplots wanted in the figure
    """

    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    if title is not None:
        fig.suptitle(title)
        fig.tight_layout()
    for ind, title in enumerate(figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        if imgtitle:
            axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.subplots_adjust(wspace=0, hspace=0)
    # plt.tight_layout()  # optional
    plt.pause(0.5)  # pause a bit so that plots are updated
    plt.show()
