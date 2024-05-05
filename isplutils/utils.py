"""
Random utils file

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
"""

from pprint import pprint


def make_train_tag(net_class: str,
                   lr: float,
                   batch_size: int,
                   p_train_test: float,
                   p_train_val: float,
                   split_seed: int,
                   suffix: str,
                   debug: bool,
                   in_channels: int = None,
                   init_period: int = None,
                   jpeg_bs: int = None,
                   fl_stride: int = None,
                   grayscale: bool = None,
                   random_crop: bool = False,
                   aa_pool_only: bool = False,
                   ):
    # Training parameters and tag
    tag_params = dict(net=net_class,
                      lr=lr,
                      batch_size=batch_size,
                      split_train_test= p_train_test,
                      split_train_val= p_train_val,
                      split_seed=split_seed,
                      in_channels=in_channels,
                      init_period=init_period,
                      jpeg_bs=jpeg_bs,
                      fl_stride=fl_stride,
                      grayscale=grayscale,
                      random_crop=random_crop,
                      )

    # Add aa_pool_only to tag_params only if model contains AA
    if 'AA' in net_class:
        tag_params['aa_pool_only'] = aa_pool_only

    # Remove None values from the dictionary
    tag_params = {k: v for k, v in tag_params.items() if v is not None}

    print('Parameters')
    pprint(tag_params)
    tag = 'debug_' if debug else ''
    tag += '_'.join(['-'.join([key, str(tag_params[key])]) for key in tag_params])
    if suffix is not None:
        tag += '_' + suffix
    print('Tag: {:s}'.format(tag))
    return tag


def plot_lr_schedule(lr, T_0, eta_min, steps):
    import matplotlib.pyplot as plt
    import torch

    # Create a dummy optimizer for the plot
    optimizer = torch.optim.SGD([torch.randn(1)], lr=lr)

    # Create a new scheduler with the same parameters
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0,
                                                                     eta_min=eta_min)

    lrs = []

    # Simulate a number of steps
    for step in range(steps):
        scheduler.step()
        # Append the current learning rate to our list
        lrs.append(scheduler.get_last_lr()[0])

    # Plot the learning rates
    plt.plot(lrs)
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.show()
