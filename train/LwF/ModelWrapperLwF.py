import torch.nn as nn
import torch


class ModelWrapperLwF(nn.Module):
    def __init__(self, model, last_layer_name):
        super(ModelWrapperLwF, self).__init__()
        self.model = model
        self.last_layer_name = last_layer_name
        self.finetune_mode = False

    def set_finetune_mode(self, idx):
        self.finetune_mode = True
        self.finetune_idx = idx

    def forward(self, x):
        sub_index = 0
        last_layer = False
        for name, module in self.model._modules.items():
            for namex, modulex in module._modules.items():
                if last_layer:
                    outputs.append(modulex(x))
                else:
                    try:
                        x = modulex(x)
                    except:
                        x = modulex(torch.flatten(x, 1))  # MLP fix LWF
                if name == 'classifier' and namex == str(self.last_layer_name - 1):
                    last_layer = True
                    outputs = []

            # for reshaping the fully connected layers
            # need to be changed for
            if sub_index == 0:
                if len(x.shape) == 4:  # MLP fix
                    x = x.view(x.size(0), x.size(1) * x.size(2) * x.size(3))
            sub_index += 1

        assert len(outputs) > 0, "No outputs from heads extracted."

        # In this case only return for one head an output
        if self.finetune_mode:
            if self.finetune_idx > 0:
                outputs = outputs[self.finetune_idx]
        return outputs
