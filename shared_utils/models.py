import os
from abc import ABCMeta, abstractmethod

import torch
from torchvision import models
import torch.nn.functional as F

import shared_utils.utils

########################################
# PARSING and NAMING
########################################
# Models you can choose as arg to run main scripts
model_names = ["alexnet_pretrain", "alexnet", 'vgg11_pretrain', 'vgg11_BN_pretrain', 'MLP_cl_100_100']


def parse_model_name(models_root_path, model_name, input_size):
    """
    Parses model name into model type object.
    :param model_name: e.g. vgg11_pretrain
    :param input_size: Size of the input: (w , h)
    :return: model wrapper object
    """
    pretrained = True if 'pretrain' in model_name else False

    if AlexNet.base_name in model_name:
        base_model = AlexNet(models_root_path, pretrained=pretrained, create=True)
    elif VGG11.vgg_config in model_name:
        base_model = VGG11(models_root_path, input_size, model_name, pretrained=pretrained, create=True)
    elif MLP.base_name in model_name:
        base_model = MLP(models_root_path, model_name, create=True)
    else:
        raise NotImplementedError("MODEL NOT IMPLEMENTED YET: ", model_name)

    return base_model


########################################
# MODELS
########################################
class Model(metaclass=ABCMeta):
    @property
    @abstractmethod
    def last_layer_idx(self): pass

    @abstractmethod
    def name(self): pass

    @abstractmethod
    def path(self): pass


class ModelRegularization(object):
    dropout = 'DROP'
    batchnorm = 'BN'


############################################################
############################################################
# AlexNet
############################################################
############################################################
class AlexNet(Model):
    last_layer_idx = 6
    base_name = "alexnet"

    def __init__(self, models_root_path, pretrained=True, create=False):
        if not os.path.exists(os.path.dirname(models_root_path)):
            raise Exception("MODEL ROOT PATH FOR {} DOES NOT EXIST: {}".format(self.base_name, models_root_path))

        name = [self.base_name]
        if pretrained:
            name.append("pretrained_imgnet")
        else:
            name.append("scratch")
        self.name = '_'.join(name)
        self.path = os.path.join(models_root_path,
                                 self.name + ".pth.tar")  # In training scripts: AlexNet pretrained on Imgnet when empty

        if not os.path.exists(self.path):
            if create:
                torch.save(models.alexnet(pretrained=pretrained), self.path)
                print("SAVED NEW {} MODEL (name={}) to {}".format(self.base_name, self.name, self.path))
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("STARTING FROM EXISTING {} MODEL (name={}): {}".format(self.base_name, self.name, self.path))

    def name(self):
        return self.name

    def path(self):
        return self.path


############################################################
############################################################
# LINEAR NETS
############################################################
############################################################
class MLP(Model):
    """
    MLP model with multiple layers.
    """
    base_name = "MLP"
    in_dim = 3 * 28 * 28
    out_dim = 10
    last_layer_idx = None

    def __init__(self, models_root_path, name, create=False):
        """
        :param name: e.g. MPL_cl_100_100, is MPL model with 2 hidden layers of each 100 units.
        """
        assert os.path.exists(os.path.dirname(models_root_path)), \
            print("MODEL ROOT PATH NOT EXISTING: ", models_root_path)
        assert self.base_name in name, print("Wrong MLP naming: " + name)
        self.name = name
        self.path = os.path.join(models_root_path, self.name + '(in_dim={}).pth.tar'.format(self.in_dim))

        if not os.path.exists(self.path):
            if create:
                classifier_list = parse_classifier_name(name)
                model = LinearNet(self.in_dim, classifier_list, self.out_dim)
                torch.save(model, self.path)
                print("SAVED NEW MLP MODEL (name=", self.name, ") to ", self.path)
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("STARTING FROM EXISTING MLP MODEL (name=", self.name, ") in ", self.path)

    def name(self):
        return self.name

    def path(self):
        return self.path


class LinearNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        """

        :param D_in: Input dim
        :param H: list of hidden layer dimensions, determines the amount of layers.
        :param D_out: Output dim
        """
        super(LinearNet, self).__init__()
        self.D_in = D_in
        self.D_out = D_out

        self.classifier = torch.nn.ModuleList()
        self.dim_seq = [D_in] + H + [D_out]
        print("LayerNet with dims: {}".format(self.dim_seq))

        for idx_dim in range(0, len(self.dim_seq) - 1):
            dim1 = int(self.dim_seq[idx_dim])
            dim2 = int(self.dim_seq[idx_dim + 1])
            self.classifier.append(torch.nn.Linear(dim1, dim2))
            if idx_dim < len(self.dim_seq) - 2:  # No relu for outputs
                self.classifier.append(torch.nn.ReLU())
        print("LayerNet with layers: {}".format(self.classifier))

    def forward(self, x):
        out = torch.flatten(x, 1)  # Keep batch size, flatten rest
        for mod in self.classifier:
            out = mod(out)
        return out


############################################################
############################################################
# VGG MODELS
############################################################
############################################################
class VGGModel(Model):
    """
    VGG based models.
    small_VGG9_cl_512_512_DROP_BN
    """
    base_name = "VGG"
    last_layer_idx = 4  # vgg_classifier_last_layer_idx
    pooling_layers = 4  # in all our models 4 max pooling layers with stride 2
    final_featmap_count = 512

    def __init__(self, models_root_path, input_size, model_name, model,
                 overwrite_mode=False, create=False, dropout=False):
        if not os.path.exists(os.path.dirname(models_root_path)):
            raise Exception("MODEL ROOT PATH FOR ", model_name, " DOES NOT EXIST: ", models_root_path)

        self.name = model_name
        parent_path = os.path.join(models_root_path, "pytorch_vgg_input={}x{}"
                                   .format(str(input_size[0]), str(input_size[1])))
        self.path = os.path.join(parent_path, self.name + ".pth.tar")

        # After classifier name
        dropout = ModelRegularization.dropout in model_name.split("_") or dropout  # DROP in name

        if dropout:
            self.last_layer_idx = 6

        if overwrite_mode or not os.path.exists(self.path):
            if create:
                shared_utils.utils.create_dir(parent_path)
                torch.save(model, self.path)
                print("SAVED NEW {} MODEL (name={}) to {}".format(self.base_name, self.name, self.path))
            else:
                raise Exception("Not creating non-existing model: ", self.name)
        else:
            print("MODEL ", model_name, " already exist in path = ", self.path)

    def name(self):
        return self.name

    def path(self):
        return self.path


class VGG11(VGGModel):
    vgg_config = "vgg11"

    def __init__(self, models_root_path, input_size, model_name=vgg_config,
                 overwrite_mode=False, create=False, pretrained=True):
        """
        :param model_name: defined in main script, e.g. vgg11_BN_pretrain
        :param overwrite_mode: Overwrite if model already exists
        """
        batch_norm = ModelRegularization.batchnorm in model_name.split("_")  # BN in name
        model = models.vgg11_bn(pretrained=pretrained) if batch_norm else models.vgg11(pretrained=pretrained)

        if pretrained:
            model_name = '_'.join([model_name.replace('_pretrain', ''), "pretrained_imgnet"])
        super().__init__(models_root_path, input_size, model_name, model,
                         overwrite_mode=overwrite_mode, create=create, dropout=True)


############################################################
# MODEL UTILITIES
############################################################
def parse_classifier_name(model_name, classifier_layers=3):
    """
    Takes in model name (e.g. small_VGG9_cl_512_512_BN), and returns classifier sizes: [512,512]
    :param model_name:
    :return:
    """
    return model_name[model_name.index("cl_"):].split("_")[1:classifier_layers]
