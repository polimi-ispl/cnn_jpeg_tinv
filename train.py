"""
Main training script for the various architectures considered in the paper.
The BondiNet and AABondiNet architectures are trained on the custom block dataset, while the other architectures are trained on the 8x8 grid dataset.
Please note that these architectures can vary the value for the stride used in the first layer.

Authors:
Edoardo Daniele Cannas - edoardodanielecannas@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it

"""

# TODO: fix the data in the new folders for sharing
# --- Libraries import
from tqdm import tqdm
import os
import argparse
import shutil
import warnings
import numpy as np
import torch
torch.manual_seed(21)
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from isplutils.data import CustomBlockJPEGBalancedDataset, JPEG8x8BalancedDataset, balanced_collate_fn
from albumentations.pytorch import ToTensorV2
import albumentations as A
from architectures.fornet import create_model
from architectures.utils import save_model, batch_forward
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import SubsetRandomSampler, DataLoader
from isplutils.utils import make_train_tag
import sys

# TODO: debug the new dataloader
# --- Main script
def main(args):

    # --- Parse arguments
    gpu = args.gpu
    model_name = args.model
    batch_size = args.batch_size
    lr = args.lr
    min_lr = args.min_lr
    es_patience = args.es_patience
    sched_patience = args.sched_patience
    init_period = args.init_period
    epochs = args.epochs
    workers = args.workers
    p_train_val = args.perc_train_val
    p_train_test = args.perc_train_test
    weights_folder = args.models_dir
    logs_folder = args.log_dir
    data_root = args.data_root
    debug = args.debug
    suffix = args.suffix
    initial_model = args.init
    train_from_scratch = args.scratch
    in_channels = 1 if args.grayscale else 3
    fl_stride = args.first_layer_stride
    aa_pool_only = args.aa_pool_only
    jpeg_bs = args.jpeg_bs if 'BondiNet' in model_name else 8  # Block size = 8 for the SOTA models, whatever for the BondiNet
    random_crop = args.random_crop
    if (fl_stride is not None) and ('BondiNet' in model_name):
        # For the BondiNet models, the patch size must be a multiple of 64*fl_stride
        # PLEASE NOTE that the Dataset will then adjust the patch size to be aligned to the JPEG grid!
        patch_size = 64*fl_stride
    else:
        # For the SOTA models, we are considering a 224x224 patch size
        patch_size = 224


    # --- GPU configuration
    device = 'cuda:{}'.format(gpu) if torch.cuda.is_available() else 'cpu'

    # --- Instantiate network
    params = {'in_channels': in_channels, 'num_classes': 1, 'first_layer_stride': fl_stride,
              'pool_only': aa_pool_only}  # Create a dictionary of parameters
    net = create_model(model_name, params, device)  # call the factory function to create the model

    # --- Instantiate Dataset and DataLoader
    if 'BondiNet' in model_name:
        # Using the custom block dataset for the BondiNet experiments
        transforms = [A.RandomCrop(patch_size, patch_size), ToTensorV2()] if random_crop else [ToTensorV2()]
        transforms = A.Compose(transforms)
        dataset = CustomBlockJPEGBalancedDataset(data_root=data_root, patch_size=patch_size, transforms=transforms,
                                                 grayscale=args.grayscale, jpeg_bs=jpeg_bs)
    else:
        # Using the 8x8 grid dataset for the other experiments
        net_normalizer = net.get_normalizer()
        transforms = [A.Normalize(mean=net_normalizer.mean, std=net_normalizer.std), ToTensorV2()]
        transforms = A.Compose(transforms)
        dataset = JPEG8x8BalancedDataset(data_root=data_root, patch_size=patch_size, transforms=transforms,
                                         grayscale=args.grayscale, disaligned_grid_patch=random_crop, )

    # Split in training and validation
    dataset_idxs = list(range(len(dataset)))
    np.random.seed(args.split_seed)  # setting the seed for training-val split
    np.random.shuffle(dataset_idxs)
    test_split_index = int(np.floor((1 - p_train_test) * len(dataset)))
    train_val_idxs, test_idxs = dataset_idxs[test_split_index:], dataset_idxs[:test_split_index]
    val_split_index = int(np.floor((1 - p_train_val) * len(train_val_idxs)))
    train_idx, val_idx = train_val_idxs[val_split_index:], train_val_idxs[:val_split_index]

    # --- Create Samplers
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)

    # --- Create DataLoaders
    train_dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=True,
                          sampler=train_sampler, collate_fn=balanced_collate_fn,)
    val_dl = DataLoader(dataset=dataset, batch_size=batch_size, num_workers=workers, shuffle=False, drop_last=True,
                        sampler=val_sampler, collate_fn=balanced_collate_fn,)

    # --- DEBUG DATASET --- #
    if debug:
        for batch_data in tqdm(train_dl, desc='Training loader', leave=False, total=len(train_dl)):
            img, label = batch_data

    # --- Optimization
    optimizer = torch.optim.Adam(net.get_trainable_parameters(), lr=lr)
    criterion = torch.nn.BCEWithLogitsLoss()
    if sched_patience is not None:
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min',
                                                                  patience=sched_patience, verbose=True)
    else:
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,
                                                                           T_0=init_period, eta_min=min_lr,
                                                                           verbose=True)

    # --- Checkpoint paths
    train_tag = make_train_tag(net_class=model_name, lr=lr, batch_size=batch_size, p_train_val=p_train_val,
                               p_train_test=p_train_test, split_seed=args.split_seed, suffix=suffix, debug=debug,
                               in_channels=in_channels, init_period=init_period,
                               jpeg_bs=jpeg_bs, random_crop=random_crop,
                               fl_stride=fl_stride if 'BondiNet' in model_name else None,
                               aa_pool_only=aa_pool_only if 'AA' in model_name else None)
    os.makedirs(os.path.join(weights_folder, train_tag), exist_ok=True)
    bestval_path = os.path.join(weights_folder, train_tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, train_tag, 'last.pth')

    # --- Load model from checkpoint
    min_val_loss = 100
    epoch = 0
    net_state = None
    opt_state = None
    if initial_model is not None:
        # If given load initial model
        print('Loading model form: {}'.format(initial_model))
        state = torch.load(initial_model, map_location='cpu')
        net_state = state['net']
    elif not train_from_scratch and os.path.exists(last_path):
        print('Loading model form: {}'.format(last_path))
        state = torch.load(last_path, map_location='cpu')
        net_state = state['net']
        opt_state = state['opt']
        epoch = state['epoch']
    if not train_from_scratch and os.path.exists(bestval_path):
        state = torch.load(bestval_path, map_location='cpu')
        min_val_loss = state['val_loss']
    if net_state is not None:
        incomp_keys = net.load_state_dict(net_state, strict=False)
        print(incomp_keys)
    if opt_state is not None:
        for param_group in opt_state['param_groups']:
            param_group['lr'] = lr
        optimizer.load_state_dict(opt_state)

    # --- Initialize Tensorboard
    logdir = os.path.join(logs_folder, train_tag)
    if epoch == 0:
        # If training from scratch or initialization remove history if exists
        shutil.rmtree(logdir, ignore_errors=True)

    # --- Tensorboard instance
    tb = SummaryWriter(log_dir=logdir)
    if epoch == 0:
        patch_size = patch_size if random_crop else jpeg_bs*round(patch_size/jpeg_bs)
        dummy = torch.randn((1, in_channels if 'BondiNet' in model_name else 3, patch_size, patch_size), device=device)
        dummy = dummy.to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # Dry-run first
            net(dummy)
            # Add the graph after
            #tb.add_graph(net, [dummy, ], verbose=False)

    # --- Training-validation loop
    train_tot_it = 0
    val_tot_it = 0
    es_counter = 0
    cur_lr = lr
    epochs = 1 if debug else epochs
    train_len = len(train_dl)
    for e in range(epochs):

        # Training
        net.train()
        optimizer.zero_grad()
        train_loss = train_acc = train_num = 0
        for batch_idx, batch_data in enumerate(tqdm(train_dl, desc='Training epoch {}'.format(e), leave=False, total=len(train_dl))):

            # Fetch data
            batch_img, batch_label = batch_data

            # Forward pass
            batch_loss, batch_acc = batch_forward(net, device, criterion, batch_img, batch_label)

            # Backpropagation
            batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Statistics
            batch_num = len(batch_label)
            train_num += batch_num
            train_tot_it += batch_num
            train_loss += batch_loss.item() * batch_num
            train_acc += batch_acc.item() * batch_num

            # Iteration logging
            tb.add_scalar('train/it-loss', batch_loss.item(), train_tot_it)
            tb.add_scalar('train/it-acc', batch_acc.item(), train_tot_it)

        # Validation
        net.eval()
        val_loss = val_acc = val_num = 0
        for batch_data in tqdm(val_dl, desc='Validating epoch {}'.format(e), leave=False, total=len(val_dl)):
            # Fetch data
            batch_img, batch_label = batch_data

            with torch.no_grad():
                # Forward pass
                batch_loss, batch_acc = batch_forward(net, device, criterion, batch_img, batch_label)

            # Statistics
            batch_num = len(batch_label)
            val_num += batch_num
            val_tot_it += batch_num
            val_loss += batch_loss.item() * batch_num
            val_acc += batch_acc.item() * batch_num

            # Iteration logging
            tb.add_scalar('validation/it-loss', batch_loss.item(), val_tot_it)
            tb.add_scalar('validation/it-acc', batch_acc.item(), val_tot_it)

        print('\nEpoch {}:\nTraining loss:{:.4f}, acc:{:.4f}\nValidation loss:{:.4f}, acc:{:.4f}'
              .format(e, train_loss / train_num, train_acc / train_num, val_loss / val_num, val_acc / val_num))

        # Logging
        train_loss /= train_num
        train_acc /= train_num
        val_loss /= val_num
        val_acc /= val_num
        tb.add_scalar('lr', optimizer.param_groups[0]['lr'], e)
        tb.add_scalar('train/epoch-loss', train_loss, e)
        tb.add_scalar('train/epoch-accuracy', train_acc, e)
        tb.add_scalar('validation/epoch-loss', val_loss, e)
        tb.add_scalar('validation/epoch-accuracy', val_acc, e)
        tb.flush()

        # Scheduler step
        if isinstance(lr_scheduler, torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
            # If it's CosineAnnealingWarmRestarts, pass the epoch fraction
            lr_scheduler.step(e + batch_idx / train_len)
        else:
            # Otherwise, pass the validation loss
            lr_scheduler.step(val_loss)

        # Epoch checkpoint
        save_model(net, optimizer, train_loss, val_loss, batch_size, epoch, last_path)
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            save_model(net, optimizer, train_loss, val_loss, batch_size, epoch, bestval_path)
            es_counter = 0
        else:
            es_counter += 1

        if optimizer.param_groups[0]['lr'] <= min_lr:
            print('Reached minimum learning rate. Stopping.')
            break
        elif es_counter == es_patience:
            print('Early stopping patience reached. Stopping.')
            break

    # Needed to flush out last events for the logger
    tb.close()

    print('Training completed! Bye!')


if __name__ == '__main__':

    # --- Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--model', help='Model name', type=str, default='FOCALCNN',
                        choices=['BondiNet', 'AABondiNet', 'DenseNet121', 'AADenseNet121', 'ResNet50', 'AAResNet50',])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--min_lr', type=float, default=1e-8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--es_patience', type=int, default=10,
                        help='Patience for stopping the training if no improvement'
                             'on the validation loss is seen')
    # Add mutually exclusive group for the two schedulers
    scheduler_args = parser.add_mutually_exclusive_group(required=False)
    scheduler_args.add_argument('--init_period', type=int, default=10, help='Period for the CosineAnnealingWarmRestart')
    scheduler_args.add_argument('--sched_patience', type=int, default=None, help='Patience for the ReduceLROnPlateau scheduler')
    parser.add_argument('--workers', type=int, default=os.cpu_count() // 2)
    parser.add_argument('--perc_train_test', type=float, help='Fraction of trainval/test set', default=0.75)
    parser.add_argument('--perc_train_val', type=float, help='Fraction of train/val set', default=0.75)
    parser.add_argument('--split_seed', type=int, help='Random seed for training/validation split', default=42)
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to the folder containing the datasets')
    parser.add_argument('--jpeg_bs', type=int, help='Block size for the JPEG compression', default=8)
    parser.add_argument('--log_dir', type=str, help='Directory for saving the training logs',
                        default='./logs')
    parser.add_argument('--models_dir', type=str, help='Directory for saving the models weights',
                        default='./models')
    parser.add_argument('--init', type=str, help='Weight initialization file')
    parser.add_argument('--scratch', action='store_true',
                        help='Train from scratch the model, or use the last checkpoint available')
    parser.add_argument('--debug', action='store_true', help='Activate debug')
    parser.add_argument('--suffix', type=str, help='Suffix to default tag')
    parser.add_argument('--grayscale', action='store_true', help='Whether to work on grayscale images or not')
    parser.add_argument('--first_layer_stride', type=int,
                        help='Stride of the first layer for the BondiNet', default=2)
    parser.add_argument('--aa_pool_only', action='store_true', help='Whether to use the BlurPool only in '
                                                                                  'the maxPool layers of the AABondiNets')
    parser.add_argument('--random_crop', action='store_true', help='Whether to use random crop or not')
    args = parser.parse_args()

    # --- CALL MAIN FUNCTION --- #
    try:
        main(args)
    except Exception as e:
        print(e)

    # --- Exit the script
    sys.exit(0)