from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

#***********************************************************************************************************
# PATHS
#***********************************************************************************************************
dataset    = ''                     # required=True,               help='cifar10 | lsun | imagenet | folder | lfw '
dataroot   = ''                     # required=True,               help='path to dataset'

netG       = ''                     # default='',                  help="path to netG (to continue training)"
netD       = ''                     # default='',                  help="path to netD (to continue training)"
outf       = '.'                    # default='.',                help='folder to output images and model checkpoints'
#***********************************************************************************************************

#***********************************************************************************************************
# MAIN PARAMS
#***********************************************************************************************************
batchSize  = 64                     # type=int,   default=64,        help='input batch size'
imageSize  = 64                     # type=int,   default=64,        help='the height / width of the input image to network'
nz         = 64                     # type=int,   default=100,       help='size of the latent z vector'
ngf        = 100                    # type=int,   default=64)
ndf        = 64                     # type=int,   default=64)
epoch      = 25                     # type=int,   default=25,        help='number of epochs to train for'
lr         = 0.0002                 # type=float, default=0.0002,    help='learning rate, default=0.0002'
beta1      = 0.5                    # type=float, default=0.5,       help='beta1 for adam. default=0.5'

loss       = nn.BCELoss()           # loss function
#***********************************************************************************************************

#***********************************************************************************************************
# OTHER PARAMS
#***********************************************************************************************************
nc         = 3
workers    = 2                      # type=int, default=2,         help='number of data loading workers'
cuda       = True                   # action='store_true',         help='enables cuda'
ngpu       = 8                      # type=int, default=1,        help='number of GPUs to use'
seed       = None                   # type=int,                    help='manual seed'
real_label = 1
fake_label = 0
#***********************************************************************************************************

try:
    os.makedirs(outf)
except OSError:
    pass

if seed is None:
    seed = random.randint(1, 10000)

print("Random Seed: ", seed )
random.seed(seed )
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed_all(seed)

cudnn.benchmark = True

if torch.cuda.is_available() and not cuda:
    print("WARNING: You have a CUDA device, so you should probably run with cuda = True")

if dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(imageSize),
                                   transforms.CenterCrop(imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif dataset == 'lsun':
    dataset = dset.LSUN(db_path=dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(imageSize),
                            transforms.CenterCrop(imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif dataset == 'cifar10':
    dataset = dset.CIFAR10(root=dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batchSize,
                                         shuffle=True, num_workers=int(workers))

