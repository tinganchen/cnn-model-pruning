import argparse
import os 
# import pdb

parser = argparse.ArgumentParser(description = 'Generatvie Adversarial Learning')

#MIU = 1
#LAMBDA = 0.6

# 1. training: main.py [cifar-10]
# 2. fine-tune: finetune.py [cifar-10]


## Warm-up 
parser.add_argument('--gpus', type = int, nargs = '+', default = [0], help = 'Select gpu to use')
parser.add_argument('--dataset', type = str, default = 'cifar10', help = 'Dataset to train')

parser.add_argument('--data_dir', type = str, default = os.getcwd() + '/data/cifar10/', #'/home/yanchenqian/data/cifar10/',
                    help = 'The directory where the input data is stored.')
parser.add_argument('--job_dir', type = str, default = 'experiment/resnet/ft_lambda_0.6_miu_1/', help = 'The directory where the summaries will be stored.') # 'experiments/'
# 1. train: default = f'experiment/resnet/lambda_{LAMBDA}_miu_{MIU}/'
# 2. fine_tuned: default = f'experiment/resnet/ft_lambda_{LAMBDA}_miu_{MIU}/'

parser.add_argument('--teacher_dir', type = str, default = 'pretrained/', help = 'The directory where the teacher model saved.')
parser.add_argument('--teacher_file', type = str, default = 'resnet_56.pt', help = 'The file the teacher model weights saved as.')
# 'model_best.pt', 'resnet_56.pt'

parser.add_argument('--reset', action = 'store_true', help = 'Reset the directory?')
parser.add_argument( '--resume',  type = str, default = None, help = 'Load the model from the specified checkpoint.')
# 1. train: default = None
# 2. fine_tuned: default = 'experiment/resnet/lambda_0.6_miu_1/resnet_pruned_15.pt'

parser.add_argument('--refine', type = str, default = 'experiment/resnet/lambda_0.6_miu_1/checkpoint/model_best.pt', help = 'Path to the model to be fine tuned.') # None
# 1. train: default = None
# 2. fine_tuned: default = f'experiment/resnet/lambda_0.6_miu_1/checkpoint/model_best.pt'

## Training
parser.add_argument('--arch', type = str, default = 'resnet', help = 'Architecture of teacher and student')
parser.add_argument('--target_model', type = str, default = 'gal_05', help = 'The target model.')
parser.add_argument('--student_model', type = str, default = 'resnet_56_sparse', help = 'The model of student.')
parser.add_argument('--teacher_model', type = str, default = 'resnet_56', help = 'The model of teacher.')
parser.add_argument('--num_epochs', type = int, default = 50, help = 'The num of epochs to train.') # 100
# 1. train: default = 100
# 2. fine_tuned: default = 50

parser.add_argument('--train_batch_size', type = int, default = 128, help = 'Batch size for training.')
parser.add_argument('--eval_batch_size', type = int, default = 100, help = 'Batch size for validation.')

parser.add_argument('--momentum', type = float, default = 0.9, help = 'Momentum for MomentumOptimizer.')
parser.add_argument('--lr', type = float, default = 0.01)
# 1. train: default = 0.1
# 2. fine_tuned: default = 0.01

parser.add_argument('--lr_decay_step',type = int, default = 30)
parser.add_argument('--mask_step', type = int, default = 200, help = 'The frequency of mask to update')
parser.add_argument('--weight_decay', type = float, default = 2e-4, help = 'The weight decay of loss.')
parser.add_argument('--miu', type = float, default = 0.6, help = 'The miu of data loss.')
parser.add_argument('--lambda', dest = 'sparse_lambda', type = float, default = 1, help = 'The sparse lambda for l1 loss') # 0.6
parser.add_argument('--random', action = 'store_true', help = 'Random weight initialize for finetune')
parser.add_argument('--pruned', action = 'store_true', default = False, help = 'Load pruned model')
# 1. train: default = False
# 2. fine_tuned: default = True

parser.add_argument('--thre', type = float, default = 0.0, help = 'Thred of mask to be pruned')
parser.add_argument('--keep_grad', action = 'store_true', help = 'Keep gradients of mask for finetune')

## Status
parser.add_argument('--print_freq', type = int, default = 200, help = 'The frequency to print loss.')
parser.add_argument('--test_only', action = 'store_true', default = False, help = 'Test only?') 


args = parser.parse_args()

if args.resume is not None and not os.path.isfile(args.resume):
    raise ValueError('No checkpoint found at {} to resume'.format(args.resume))

if args.refine is not None and not os.path.isfile(args.refine):
    raise ValueError('No checkpoint found at {} to refine'.format(args.refine))

