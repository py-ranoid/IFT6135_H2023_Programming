import math
import os
import random
import shutil
import warnings
import wandb


import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np

from q3_solution  import SimSiam

from q3_misc import TwoCropsTransform, load_checkpoints, load_pretrained_checkpoints

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# train for one epoch 
def train(train_loader, model, optimizer, device):

    # switch to train mode
    model.train()

    losses = []
    for i, (images, _) in enumerate(train_loader):

        if device is not None:
            images[0] = images[0].to(device, non_blocking=True)
            images[1] = images[1].to(device, non_blocking=True)

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = model.loss(p1,p2,z1,z2,similarity_function='CosineSimilarity')

        losses.append(loss.item())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return losses

# save checkpoints 
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

# test using a knn monitor
def test(net, memory_data_loader, test_data_loader, device, knn_k, knn_t ):
    net.eval()
    classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for i, (data, target) in enumerate(memory_data_loader):
            feature = net(data.to(device,non_blocking=True))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)
        # loop test data to predict the label by weighted knn search
        for i, (data, target) in enumerate(test_data_loader):
            data, target = data.to(device,non_blocking=True), target.to(device,non_blocking=True)
            feature = net(data)
            feature = F.normalize(feature, dim=1)
            
            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()

    return total_top1 / total_num * 100

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

# adjust LR
def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

# %%
# train for one epoch
def train_clf(train_loader, model, criterion, optimizer, device):

    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.eval()
    losses=[]
    top1=[]
    top5=[]
    for i, (images, target) in enumerate(train_loader):

        if device is not None:
            images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.append(loss.item())
        top1.append(acc1[0].cpu())
        top5.append(acc5[0].cpu())

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return top1

# validation
def validate_clf(val_loader, model, criterion, device):

    # switch to evaluate mode
    model.eval()
    losses=[]
    top1=[]
    top5=[]
    with torch.no_grad():

        for i, (images, target) in enumerate(val_loader):
            if device is not None:
                images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.append(loss.item())
            top1.append(acc1[0].cpu())
            top5.append(acc5[0].cpu())

    return top1


def sanity_check(state_dict, pretrained_weights):
    """
    Linear classifier should not change any weights other than the linear layer.
    This sanity check asserts nothing wrong happens (e.g., BN stats updated).
    """
    print("=> loading '{}' for sanity check".format(pretrained_weights))
    checkpoint = torch.load(pretrained_weights, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']

    for k in list(state_dict.keys()):
        # only ignore fc layer
        if 'fc.weight' in k or 'fc.bias' in k:
            continue

        # name in pretrained model
        k_pre = 'encoder.' + k[len('module.'):] \
            if k.startswith('module.') else 'encoder.' + k

        assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
            '{} is changed in linear classifier training.'.format(k)

    print("=> sanity check passed.")


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
# define train and test augmentations for pretraining step 
# data
dir ='../data'
batch_size = 1024

# general
seed = 2022
num_workers = 2
save_path = './'
resume = None # None or a path to a pretrained model (e.g. *.pth.tar')
start_epoch = 0
epochs = 200 # Number of epoches (for this question 200 is enough, however for 1000 epoches, you will get closer results to the original paper)

train_transform = [
    transforms.RandomResizedCrop(32, scale=(0.08, 1.)), 
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])]

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

# datasets and loaders
train_data = datasets.CIFAR10(root=dir, train=True, transform=TwoCropsTransform(transforms.Compose(train_transform)), download=True)
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)

memory_data = datasets.CIFAR10(root=dir, train=True, transform=test_transform, download=True)
memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

test_data = datasets.CIFAR10(root=dir, train=False, transform=test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

# %%
# define train and test augmentation for linear evaluation 
train_transform_cifar = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

val_transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

train_dataset = datasets.CIFAR10(
    dir,
    transform=train_transform_cifar,
    download=True,
    train=True
    )
val_dataset = datasets.CIFAR10(
    dir,
    transform=val_transform_cifar,
    download=True,
    train=False
    )


train_loader_clf = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, #(train_sampler is None),
    num_workers=num_workers, pin_memory=True) #, sampler=train_sampler)

val_loader_clf = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=256, shuffle=False,
    num_workers=num_workers, pin_memory=True)

# Siamese backbone model
arch = "resnet18"
fix_pred_lr = True # fix the learning rate of the predictor network

#Simsiam params
dim=2048
pred_dim=512

# ablation experiments
stop_gradient=True # (True or False)
MLP_mode=None # None|'no_pred_mlp'

# optimizer
lr = 0.03
momentum = 0.9
weight_decay = 0.0005

# knn params
knn_k = 200 #k in kNN monitor
knn_t = 0.1 #softmax temperature in kNN monitor; could be different with moco-t

# %%
# set seeds
random.seed(seed)
torch.manual_seed(seed)
cudnn.deterministic = True


if __name__ == "__main__":            

    print("=> creating model '{}'".format(arch))
    # MLP_mode = 'no_pred_mlp'
    stop_gradient=True
    dim=4096
    epochs = 100
    exp_args = {"stop_grad":stop_gradient, "mlp_mode_none":MLP_mode==None, "fix_pred_lr":fix_pred_lr, "arch":arch, "dim":dim, "pred_dim":pred_dim, "lr":lr, "momentum":momentum, "weight_decay":weight_decay, "knn_k":knn_k, "knn_t":knn_t, "epochs":epochs, "batch_size":batch_size, "seed":seed, "num_workers":num_workers, "save_path":save_path, "resume":resume, "start_epoch":start_epoch}
    print(exp_args)
    wandb.init(project="RepLearning - A3", config=exp_args)
    model = SimSiam(models.__dict__[arch], dim, pred_dim, stop_gradient=stop_gradient, MLP_mode=MLP_mode)
    model.to(device)

    init_lr = lr * batch_size / 256
    if fix_pred_lr:
        optim_params = [{'params': model.encoder.parameters(), 'fix_lr': False},
                        {'params': model.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    # define optimizer
    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)


    # %%
    # We can resume from a previous checkpoint
    # if resume:
    #     model, optimizer, start_epoch = load_checkpoints(os.path.join(resume),model,optimizer,device)

    # train loop 
    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, init_lr, epoch, epochs)

        # train for one epoch
        losses = train(train_loader, model, optimizer, device)
        print('Train Epoch: [{}/{}] Train Loss:{:.5f}'.format(epoch, epochs,np.array(losses).mean() ))
        wandb.log({"Train Loss":np.array(losses).mean()})


        # test every 10 epochs
        if epoch % 5==0:
            acc1 = test(model.encoder, memory_loader, test_loader, device, knn_k, knn_t)
            print('Test Epoch: [{}/{}] knn_Acc@1: {:.2f}%'.format(epoch, epochs, acc1))
            wandb.log({"Test knn_Acc@1":acc1})
            
        # save a checkpoint every 20 epochs
        if epoch == epochs-1:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': arch,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
            }, is_best=False, filename=save_path + '/ss_checkpoint_model.pth.tar')

    # linear eval
    print("=> creating model '{}'".format(arch))
    model = models.__dict__[arch]()

    # freeze all layers but the last fc
    for name, param in model.named_parameters():
        if name not in ['fc.weight', 'fc.bias']:
            param.requires_grad = False

    # init the fc layer
    model.fc.weight.data.normal_(mean=0.0, std=0.01)
    model.fc.bias.data.zero_()
    print(model)

    # %%
    # load the pre-trained model from previous steps
    pretrained = './ss_checkpoint_model.pth.tar'
    if pretrained:
        model, optimizer, start_epoch = load_pretrained_checkpoints(os.path.join(pretrained),model,optimizer,device)
    if device is not None:
        model.to(device)

    # %%
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().to(device)

    # optimize only the linear classifier
    parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    assert len(parameters) == 2  # fc.weight, fc.bias

    optimizer = torch.optim.SGD(parameters, init_lr,
                                momentum=momentum,
                                weight_decay=weight_decay)
    # %%
    # train for the classififcation task
    for epoch in range(start_epoch, epochs):

        adjust_learning_rate(optimizer, init_lr, epoch, epochs)

        # train for one epoch
        acc1 = train_clf(train_loader_clf, model, criterion, optimizer,device)
        print('Train Epoch: [{}/{}] Train acc1:{:.2f}%'.format(epoch, epochs,np.array(acc1).mean() ))
        wandb.log({"Finetuning Training Acc":np.array(acc1).mean()})

        # evaluate on validation set
        acc1 = validate_clf(val_loader_clf, model, criterion, device)
        print('Val Epoch: [{}/{}] Val acc1:{:.2f}%'.format(epoch, epochs,np.array(acc1).mean() ))
        wandb.log({"Finetuning Val Acc":np.array(acc1).mean()})
