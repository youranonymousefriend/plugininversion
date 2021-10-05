from augmentation import Clip, Jitter, Focus, RepeatBatch, Zoom, ColorJitterR
from inversion import ImageNetVisualizer
from inversion.utils import new_init
from loss import LossArray, ViTCrossEntropyLoss, CrossEntropyLoss
from model import model_library
from saver import ExperimentSaver
from utils import exp_starter_pack

import torch
import torch.nn as nn

from collections import OrderedDict
import numpy as np
import pdb 


def summary(model, input_size, batch_size=-1, device="cuda")-> int:
    # Code taken from torchsummary
    def register_hook(module):

        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            # if 'ConvHeadPooling-62' in m_key:
            #    pdb.set_trace()
            if isinstance(input[0], list):
                if len(input) == 1:
                    input = input[0]
                else:
                    pdb.set_trace()
            elif isinstance(input, tuple):
                if len(input[0]) == 2:
                    input = (input[0][0],)
                    output = output[0]
                else:
                    pdb.set_trace()
            # if 'ConvHeadPooling-62' in m_key:
            #    pdb.set_trace()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
                not isinstance(module, nn.Sequential)
                and not isinstance(module, nn.ModuleList)
                and not (module == model)
        ):
            hooks.append(module.register_forward_hook(hook))

    device = device.lower()
    assert device in [
        "cuda",
        "cpu",
    ], "Input device is not valid, please specify 'cuda' or 'cpu'"

    if device == "cuda" and torch.cuda.is_available():
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype) for in_size in input_size]
    # print(type(x[0]))

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    print("----------------------------------------------------------------")
    line_new = "{:>20}  {:>25} {:>15}".format("Layer (type)", "Output Shape", "Param #")
    print(line_new)
    print("================================================================")
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]
        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        print(line_new)

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(input_size) * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. / (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params.numpy() * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    total_cuda = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2) # for MB

    print("================================================================")
    print("Total params: {0:,}".format(total_params))
    print("Trainable params: {0:,}".format(trainable_params))
    print("Non-trainable params: {0:,}".format(total_params - trainable_params))
    print("----------------------------------------------------------------")
    print("Input size (MB): %0.2f" % total_input_size)
    print("Forward/backward pass size (MB): %0.2f" % total_output_size)
    print("Params size (MB): %0.2f" % total_params_size)
    print("Estimated Total Size (MB): %0.2f" % total_size)
    print("----------------------------------------------------------------")
    print("Total CUDA memory (MB): %0.2f" % total_cuda)
    estimate = int(np.ceil((total_cuda - total_params_size) / (total_size) * 0.75))
    print("Estimated BS: %0.2f" % estimate)
    return estimate


def estimate_batch_size(model: torch.nn.Module, image_size: int, pre: torch.nn.Module = None) -> int:
    seq = [Focus(image_size, 0), Zoom(out_size=image_size), Jitter(), RepeatBatch(1), ColorJitterR(1, True)]
    if pre is None:
        pre = torch.nn.Sequential(*seq)
    model = torch.nn.Sequential(pre, model).cuda()
    return summary(model, (3, image_size, image_size))


def main():
    exp_name, args, _ = exp_starter_pack()
    network = args.network
    print(len(model_library))
    model, image_size, batch_size, name = model_library[network]()
    estimate = estimate_batch_size(model, image_size) 

    with open(f'batch_size/{name}.txt', 'w') as f:
        print(f'{name}\t{estimate}', file=f, flush=True)
    batch_size = estimate 

    target, way = 0, 1
    saver = ExperimentSaver(f'{name}{way}_{target}', save_id=True, disk_saver=True)

    loss = LossArray()
    loss += ViTCrossEntropyLoss(model, torch.tensor([target] * batch_size).long().cuda(), coefficient=1.)

    post = torch.nn.Sequential(Clip())
    image = None  # torch.rand(size=(1, 3, image_size // way, image_size // way)).cuda()

    image = new_init(image_size, last=image, padding=0)
    seq = [Focus(image_size, 0), Zoom(out_size=image_size), Jitter(), RepeatBatch(batch_size),
           ColorJitterR(batch_size, True)]
    pre = torch.nn.Sequential(*seq)

    visualizer = ImageNetVisualizer(loss, saver, pre, post, print_every=1, lr=.1, steps=10, save_every=100)
    image.data = visualizer(image).data

    with open(f'batch_size/{name}.txt', 'a') as f:
        print(f'{name}\t{estimate} Works!', file=f, flush=True)
        print(f'{name}\t{estimate} Works!', flush=True)


if __name__ == '__main__':
    main()
