import os
import numpy as np
import utils.common as utils
from utils.options import args
from tensorboardX import SummaryWriter
from importlib import import_module

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from fista import FISTA
from model import Discriminator

from data import cifar10
# import pdb 

from ptflops import get_model_complexity_info 

device = torch.device(f"cuda:{args.gpus[0]}")
checkpoint = utils.checkpoint(args)
print_logger = utils.get_logger(os.path.join(args.job_dir, "logger.log"))
writer_train = SummaryWriter(args.job_dir + '/run/train')
writer_test = SummaryWriter(args.job_dir + '/run/test')


def main():

    start_epoch = 0
    best_prec1 = 0.0
    best_prec5 = 0.0

    # Data loading
    print('=> Preparing data..')
    loader = cifar10(args)

    # Create model
    print('=> Building model...')
    model_t = import_module(f'model.{args.arch}').__dict__[args.teacher_model]().to(device)

    # Load teacher model
    ckpt_t = torch.load(args.teacher_dir + args.teacher_file, map_location = device) # torch.load(args.teacher_dir, map_location=device)
    # pdb.set_trace()  
    
    if args.arch == 'densenet':
        state_dict_t = {}
        for k, v in ckpt_t['state_dict'].items():
            new_key = '.'.join(k.split('.')[1:])
            if new_key == 'linear.weight':
                new_key = 'fc.weight'
            elif new_key == 'linear.bias':
                new_key = 'fc.bias'
            state_dict_t[new_key] = v
    else:
        state_dict_t = ckpt_t[list(ckpt_t.keys())[0]] # ckpt_t['state_dict']


    model_t.load_state_dict(state_dict_t)
    model_t = model_t.to(device)

    for para in list(model_t.parameters())[:-2]:
        para.requires_grad = False

    model_s = import_module(f'model.{args.arch}').__dict__[args.student_model]().to(device)

    model_dict_s = model_s.state_dict()
    model_dict_s.update(state_dict_t)
    model_s.load_state_dict(model_dict_s)

    if len(args.gpus) != 1:
        model_s = nn.DataParallel(model_s, device_ids=args.gpus)

    model_d = Discriminator().to(device) 

    models = [model_t, model_s, model_d]

    optimizer_d = optim.SGD(model_d.parameters(), lr = args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    param_s = [param for name, param in model_s.named_parameters() if 'mask' not in name]
    param_m = [param for name, param in model_s.named_parameters() if 'mask' in name]

    optimizer_s = optim.SGD(param_s, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    optimizer_m = FISTA(param_m, lr=args.lr, gamma=args.sparse_lambda)

    scheduler_d = StepLR(optimizer_d, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_s = StepLR(optimizer_s, step_size=args.lr_decay_step, gamma=0.1)
    scheduler_m = StepLR(optimizer_m, step_size=args.lr_decay_step, gamma=0.1)

    resume = args.resume
    if resume:
        print('=> Resuming from ckpt {}'.format(resume))
        ckpt = torch.load(resume, map_location = device)
        best_prec1 = ckpt['best_prec1']
        start_epoch = ckpt['epoch']

        model_s.load_state_dict(ckpt['            state_dict_s'])
        model_d.load_state_dict(ckpt['state_dict_d'])
        optimizer_d.load_state_dict(ckpt['optimizer_d'])
        optimizer_s.load_state_dict(ckpt['optimizer_s'])
        optimizer_m.load_state_dict(ckpt['optimizer_m'])
        scheduler_d.load_state_dict(ckpt['scheduler_d'])
        scheduler_s.load_state_dict(ckpt['scheduler_s'])
        scheduler_m.load_state_dict(ckpt['scheduler_m'])
        print('=> Continue from epoch {}...'.format(start_epoch))


    if args.test_only:
        test_prec1, test_prec5 = test_ONLY(args, loader.loader_test, model_s)
        print('=> Test Prec@1: {:.2f}'.format(test_prec1))
        return

    optimizers = [optimizer_d, optimizer_s, optimizer_m]
    schedulers = [scheduler_d, scheduler_s, scheduler_m]
    for epoch in range(start_epoch, args.num_epochs):
        for s in schedulers:
            s.step(epoch)

        train(args, loader.loader_train, models, optimizers, epoch)
        test_prec1, test_prec5 = test(args, loader.loader_test, model_s, epoch)

        is_best = best_prec1 < test_prec1
        best_prec1 = max(test_prec1, best_prec1)
        best_prec5 = max(test_prec5, best_prec5)

        model_state_dict = model_s.module.state_dict() if len(args.gpus) > 1 else model_s.state_dict()

        state = {
            'state_dict_s': model_state_dict,
            'state_dict_d': model_d.state_dict(),
            'best_prec1': best_prec1,
            'best_prec5': best_prec5,
            'optimizer_d': optimizer_d.state_dict(),
            'optimizer_s': optimizer_s.state_dict(),
            'optimizer_m': optimizer_m.state_dict(),
            'scheduler_d': scheduler_d.state_dict(),
            'scheduler_s': scheduler_s.state_dict(),
            'scheduler_m': scheduler_m.state_dict(),
            'epoch': epoch + 1
        }
        checkpoint.save_model(state, epoch + 1, is_best)

    print_logger.info(f"Best @prec1: {best_prec1:.3f} @prec5: {best_prec5:.3f}")

    best_model = torch.load(f'{args.job_dir}checkpoint/model_best.pt', map_location = device)

    model = import_module('utils.preprocess').__dict__[f'{args.arch}'](args, best_model['state_dict_s'])

    # Model compression info
    flops, params = get_model_complexity_info(model.to(device), (3, 32, 32), as_strings = False, print_per_layer_stat = False)
    compressionInfo(flops, params)
    

def compressionInfo(flops, params, org_gflops = 0.127087232, org_params = 0.853018):
    GFLOPs = flops / 10 ** 9
    params_num = params
    params_mem = params / 1000 ** 2
    pruned_FLOPs_ratio = (org_gflops - GFLOPs) / org_gflops
    pruned_param_ratio = (org_params - params_mem) / org_params
    
    print(f'Model GFLOPs: {GFLOPs} (-{round(pruned_FLOPs_ratio, 4) * 100} %)')
    print(f'Model params: {params_mem} (-{round(pruned_param_ratio, 4) * 100} %) MB')
    print(f'Model num of params: {round(params_num)}')
    
    if not os.path.isdir(args.job_dir + '/run/plot'):
        os.makedirs(args.job_dir + '/run/plot')
    
    with open(args.job_dir + 'run/plot/compressInfo.txt', 'w') as f:
        f.write('Model GFLOPs: {0} (-{1} %)\n'.format(round(GFLOPs, 2), round(pruned_FLOPs_ratio, 4) * 100))
        f.write('Model params: {0} (-{1} %) MB\n'.format(round(params_mem, 2), round(pruned_param_ratio, 4) * 100))
        f.write('Model num of params: {}\n'.format(round(params_num)))
        
        
def train(args, loader_train, models, optimizers, epoch):
    losses_d = utils.AverageMeter()
    losses_data = utils.AverageMeter()
    losses_g = utils.AverageMeter()
    losses_sparse = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    model_t = models[0]
    model_s = models[1]
    model_d = models[2]

    bce_logits = nn.BCEWithLogitsLoss()

    optimizer_d = optimizers[0]
    optimizer_s = optimizers[1]
    optimizer_m = optimizers[2]

    # switch to train mode
    model_d.train()
    model_s.train()
        
    num_iterations = len(loader_train)

    real_label = 1
    fake_label = 0

    for i, (inputs, targets) in enumerate(loader_train, 1):
        num_iters = num_iterations * epoch + i

        inputs = inputs.to(device)
        targets = targets.to(device)

        features_t = model_t(inputs)
        features_s = model_s(inputs)

        ############################
        # (1) Update D network
        ###########################

        for p in model_d.parameters():  
            p.requires_grad = True  

        optimizer_d.zero_grad()

        output_t = model_d(features_t.detach())
        # print('output_t',output_t)
        labels_real = torch.full_like(output_t, real_label, device = device)
        error_real = bce_logits(output_t, labels_real)

        output_s = model_d(features_s.to(device).detach())
        # print('output_s',output_t)
        labels_fake = torch.full_like(output_s, fake_label, device = device)
        error_fake = bce_logits(output_s, labels_fake)

        error_d = error_real + error_fake

        labels = torch.full_like(output_s, real_label, device = device)
        error_d += bce_logits(output_s, labels)

        error_d.backward()
        losses_d.update(error_d.item(), inputs.size(0))

        writer_train.add_scalar('Discriminator_loss', error_d.item(), num_iters)

        optimizer_d.step()

        ############################
        # (2) Update student network
        ###########################

        for p in model_d.parameters():  
            p.requires_grad = False  

        optimizer_s.zero_grad()
        optimizer_m.zero_grad()

        error_data = args.miu * F.mse_loss(features_t, features_s.to(device))

        losses_data.update(error_data.item(), inputs.size(0))
        
        writer_train.add_scalar('Data_loss', error_data.item(), num_iters)
        
        error_data.backward(retain_graph = True)

        # fool discriminator
        output_s = model_d(features_s.to(device))
        
        labels = torch.full_like(output_s, real_label, device = device)
        error_g = bce_logits(output_s, labels)

        losses_g.update(error_g.item(), inputs.size(0))
        
        writer_train.add_scalar('Generator_loss', error_g.item(), num_iters)
        
        error_g.backward(retain_graph = True)

        # train mask
        mask = []
        for name, param in model_s.named_parameters():
            if 'mask' in name:
                mask.append(param.view(-1))
        mask = torch.cat(mask)
        error_sparse = args.sparse_lambda * torch.norm(mask, 1)
        error_sparse.backward()

        losses_sparse.update(error_sparse.item(), inputs.size(0))
        
        writer_train.add_scalar('Sparse_loss', error_sparse.item(), num_iters)

        optimizer_s.step()

        decay = (epoch % args.lr_decay_step == 0 and i == 1)
        if i % args.mask_step == 0:
            optimizer_m.step(decay)

        prec1, prec5 = utils.accuracy(features_s, targets, topk = (1, 5))
        top1.update(prec1[0], inputs.size(0))
        top5.update(prec5[0], inputs.size(0))
        
        writer_train.add_scalar('Prec@1', top1.avg, num_iters)
        writer_train.add_scalar('Prec@5', top5.avg, num_iters)

        if i % args.print_freq == 0:
            print_logger.info(
                'Epoch[{0}]({1}/{2}): \n'
                'Loss_sparse (L1(mask)): {loss_sparse.val:.4f} ({loss_sparse.avg:.4f})\n'
                'Loss_data (MSE):        {loss_data.val:.4f} ({loss_data.avg:.4f})\n'
                'Loss_d (dicriminator):  {loss_d.val:.4f} ({loss_d.avg:.4f})\n'
                'Loss_g (generator):     {loss_g.val:.4f} ({loss_g.avg:.4f})\n'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f}), '
                'Prec@5 {top5.val:.3f} ({top5.avg:.3f})\n'
                .format(
                epoch, i, num_iterations, 
                loss_sparse = losses_sparse, 
                loss_data = losses_data, 
                loss_g = losses_g, loss_d = losses_d, top1 = top1, top5 = top5))
            
            mask = []
            pruned = 0
            num = 0
            
            for name, param in model_s.named_parameters():
                if 'mask' in name:
                    weight_copy = param.clone()
                    param_array = np.array(weight_copy.detach().cpu())
                    pruned += sum(w == 0 for w in param_array)
                    num += len(param_array)
                    
            print_logger.info("Pruned {} / {}\n".format(pruned, num))

def test_ONLY(args, loader_test, model_s):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_s.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_s(inputs).to(device)
            loss = cross_entropy(logits, targets)

            prec1, prec5 = utils.accuracy(logits, targets, topk=(1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            # writer_test.add_scalar('Prec@1', top1.avg, num_iters)
            # writer_test.add_scalar('Prec@5', top5.avg, num_iters)
        
    print_logger.info('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '===============================================\n'
    .format(top1 = top1, top5 = top5))

    return top1.avg, top5.avg

def test(args, loader_test, model_s, epoch):
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()

    cross_entropy = nn.CrossEntropyLoss()

    # switch to eval mode
    model_s.eval()
    
    num_iterations = len(loader_test)

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader_test, 1):
            num_iters = num_iterations * epoch + i
            
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits = model_s(inputs).to(device)
            loss = cross_entropy(logits, targets)
            
            writer_test.add_scalar('Test_loss', loss.item(), num_iters)
        
            prec1, prec5 = utils.accuracy(logits, targets, topk = (1, 5))
            losses.update(loss.item(), inputs.size(0))
            top1.update(prec1[0], inputs.size(0))
            top5.update(prec5[0], inputs.size(0))
            
            writer_test.add_scalar('Prec@1', top1.avg, num_iters)
            writer_test.add_scalar('Prec@5', top5.avg, num_iters)
        
    print_logger.info('Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}\n'
                      '===============================================\n'
                      .format(top1 = top1, top5 = top5))

    return top1.avg, top5.avg

if __name__ == '__main__':
    main()


