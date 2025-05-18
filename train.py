#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, print_function
import os
import sys
import argparse
import random
from copy import deepcopy
from typing import List, Dict, Tuple, Optional, Any, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.backends import cudnn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
import torchvision.models as models
import torchvision.transforms as transforms

import losses
from utils import RandomIdentitySampler, mkdir_if_missing, logging
from ImageFolder import ImageFolder

# Disable PyTorch warnings for cleaner output
import warnings
warnings.filterwarnings("ignore")

# Enable CUDA benchmark mode for faster training
cudnn.benchmark = True


class DeepInversionFeatureHook:
    """
    Hook for the DeepInversion feature distribution regularization.
    Registers a forward hook on the given module to compute feature distributions.
    """
    def __init__(self, module: nn.Module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.r_feature = None
        
    def hook_fn(self, module: nn.Module, input: torch.Tensor, output: torch.Tensor) -> None:
        """
        Compute DeepInversion's feature distribution regularization.
        Forces mean and variance to match between two distributions.
        """
        nch = input[0].shape[1]
        
        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        # Compute norm between running statistics and batch statistics
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
            module.running_mean.data.type(mean.type()) - mean, 2)
        
        self.r_feature = r_feature
    
    def close(self) -> None:
        """Remove the hook."""
        self.hook.remove()


def get_image_prior_losses(inputs_jit: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute total variation regularization loss.
    
    Args:
        inputs_jit: Jittered input images tensor
        
    Returns:
        Tuple containing L1 and L2 total variation losses
    """
    # Calculate pixel differences in various directions
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + \
                 (diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    
    return loss_var_l1, loss_var_l2


def scatter_loss(inputs: torch.Tensor, num_instances: int, bs: int) -> Tuple:
    """
    Compute metrics to measure class scatter in the feature space.
    
    Args:
        inputs: Normalized feature embeddings
        num_instances: Number of instances per class in batch
        bs: Batch size
        
    Returns:
        Tuple containing various scatter metrics and features
    """
    num_class_in_minibatch = int(bs / num_instances)

    # Compute pairwise distance for first class
    A_idx = torch.LongTensor(range(0, num_instances)).cuda()
    input1 = inputs.index_select(0, A_idx)
    mean1 = torch.mean(input1, dim=0)
    input1_centered = input1 - mean1
    dist1 = input1_centered.T @ input1_centered

    # Compute pairwise distance for second class
    A_idx = torch.LongTensor(range(num_instances, (2 * num_instances))).cuda()
    input2 = inputs.index_select(0, A_idx)
    mean2 = torch.mean(input2, dim=0)
    input2_centered = input2 - mean2
    dist2 = input2_centered.T @ input2_centered

    means = torch.stack([mean1, mean2], dim=0)
    n5 = means.size(0)
    dist_mean = torch.pow(means, 2).sum(dim=1, keepdim=True).expand(n5, n5)
    dist_mean = dist_mean + dist_mean.t()
    dist_mean.addmm_(means, means.t(), beta=1, alpha=-2)
    dist_mean = dist_mean.clamp(min=1e-12).sqrt()  # for numerical stability
    
    # Initialize other variables
    dist3 = []
    dist4 = []
    input3 = []
    input4 = []
    mean3 = []
    mean4 = []
    means = torch.stack([mean1, mean2], dim=0)
    mean_t = torch.mean(means, dim=0)
    
    # Handle 4-class case
    if num_class_in_minibatch == 4:
        # Compute pairwise distance for third class
        A_idx = torch.LongTensor(range(2 * num_instances, (3 * num_instances))).cuda()
        input3 = inputs.index_select(0, A_idx)
        mean3 = torch.mean(input3, dim=0)
        input3_centered = input3 - mean3
        dist3 = input3_centered.T @ input3_centered

        # Compute pairwise distance for fourth class
        A_idx = torch.LongTensor(range(3 * num_instances, (4 * num_instances))).cuda()
        input4 = inputs.index_select(0, A_idx)
        mean4 = torch.mean(input4, dim=0)
        input4_centered = input4 - mean4
        dist4 = input4_centered.T @ input4_centered

        means = torch.stack([mean1, mean2, mean3, mean4], dim=0)
        mean_t = torch.mean(means, dim=0)
        n5 = means.size(0)
        dist_mean = torch.pow(means, 2).sum(dim=1, keepdim=True).expand(n5, n5)
        dist_mean = dist_mean + dist_mean.t()
        dist_mean.addmm_(means, means.t(), beta=1, alpha=-2)
        dist_mean = dist_mean.clamp(min=1e-12).sqrt()  # for numerical stability

    elif num_class_in_minibatch != 2:
        print("Error: number of classes in each mini-batch must be 2 or 4")

    return dist1, dist2, dist3, dist4, dist_mean, input1, input2, input3, input4, mean1, mean2, mean3, mean4, mean_t


def extract_features(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from the model using the given data loader.
    
    Args:
        model: Neural network model
        data_loader: Data loader for feature extraction
        
    Returns:
        Tuple containing features and corresponding labels
    """
    model = model.cuda()
    model.eval()

    features = []
    labels = []

    for i, data in enumerate(data_loader, 0):
        imgs, pids = data

        inputs = imgs.cuda()
        with torch.no_grad():
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            outputs = F.normalize(outputs, p=2, dim=1)
            outputs = outputs.cpu().numpy()

        # Store features and labels
        if not features:
            features = outputs
            labels = pids
        else:
            features = np.vstack((features, outputs))
            labels = np.hstack((labels, pids))

    return features, labels


def freeze_model(model: nn.Module) -> nn.Module:
    """
    Freeze all parameters of the model.
    
    Args:
        model: Neural network model
        
    Returns:
        Model with frozen parameters
    """
    for param in model.parameters():
        param.requires_grad = False
    return model


def initial_train_fun(args: argparse.Namespace, 
                      trainloader: torch.utils.data.DataLoader, 
                      dataset_sizes_train: int, 
                      num_class: int, 
                      dictlist: Dict[int, int]) -> None:
    """
    Train the initial model (non-incremental training phase).
    
    Args:
        args: Arguments from the argument parser
        trainloader: Data loader for training
        dataset_sizes_train: Size of the training dataset
        num_class: Number of classes
        dictlist: Dictionary mapping class indices
    """
    # Initialize model
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_class)
    model = model.cuda()
    best_model_wts = deepcopy(model.state_dict())
    
    # Setup logging and loss
    log_dir = os.path.join('checkpoints', args.log_dir)
    criterion = nn.CrossEntropyLoss()
    
    # Setup optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_0, momentum=args.momentum_0,
                               weight_decay=args.weight_decay_0)
    exp_lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.lr_step)

    # Training loop
    for epoch in range(args.epochs_0):
        print(f'Epoch {epoch}/{args.epochs_0 - 1}')
        print('-' * 50)

        model.train()  # Set model to training mode
        dataloaders = deepcopy(trainloader)
        dataset_sizes = dataset_sizes_train
        running_loss = 0.0
        running_corrects = 0

        # Iterate over batches
        for inputs, labels in dataloaders:
            inputs = inputs
            labels = labels
            labels_np = labels.numpy()

            # Map labels according to dictionary
            for ii in range(len(labels_np)):
                labels_np[ii] = dictlist[labels_np[ii]]

            labels = labels.type(torch.LongTensor)
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                exp_lr_scheduler.step()

    # Save best model weights
    best_model_wts = deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    model = nn.Sequential(*list(model.children())[:-1])

    # Save model state
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    os.makedirs(log_dir, exist_ok=True)
    torch.save(state,
              os.path.join(log_dir, f"{args.method}_task_0_{args.epochs}_model.pt"))


def train_fun(args: argparse.Namespace, 
              train_loader: torch.utils.data.DataLoader, 
              current_task: int, 
              old_labels: List, 
              current_labels: List) -> None:
    """
    Train the model for continual learning.
    
    Args:
        args: Arguments from the argument parser
        train_loader: Data loader for training
        current_task: Current task index
        old_labels: List of old class labels
        current_labels: List of current class labels
    """
    # Setup logging directory
    log_dir = os.path.join('checkpoints', args.log_dir)
    mkdir_if_missing(log_dir)
    sys.stdout = logging.Logger(os.path.join(log_dir, 'log.txt'))
    num_classes_in_minibatch = int(args.BatchSize / args.num_instances)

    # Initialize model based on task
    if current_task == 0:
        if args.method != 'MLRPTM':
            if args.data == 'cifar100':
                model_res = models.resnet18(pretrained=True)
            else:
                model_res = models.resnet18(pretrained=False)

            model = nn.Sequential(*list(model_res.children())[:-1])
        else:
            model_res = models.resnet18(pretrained=False)
            model = nn.Sequential(*list(model_res.children())[:-1])
            state = torch.load(os.path.join(log_dir, f"{args.method}_task_{current_task}_{args.epochs}_model.pt"))
            model.load_state_dict(state['state_dict'])
    else:
        # For incremental tasks, load previous checkpoint
        model_res = models.resnet18(pretrained=False)
        model = nn.Sequential(*list(model_res.children())[:-1])
        state1 = torch.load(os.path.join(log_dir, f"{args.method}_task_{current_task-1}_{args.epochs}_model.pt"))
        model.load_state_dict(state1['state_dict'])
        
        # Setup methods for continual learning
        if args.method != 'Fine_tuning' and args.method != 'NoisyFine_tuning':
            model_old = deepcopy(model)
            model_old.eval()
            model_old = freeze_model(model_old)
            model_old = model_old.cuda()
            model_gen = deepcopy(model)
            model_gen.eval()

    # Move model to GPU and set evaluation mode initially
    model = model.cuda()
    model.eval()

    # Setup loss functions and optimizer
    criterion = losses.create(args.loss_m, margin=args.margin, num_instances=args.num_instances).cuda()
    criterion_task = losses.create(args.loss_confusion).cuda()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)

    # Setup for MLRPTM method in incremental tasks
    if current_task > 0 and args.method == 'MLRPTM':
        filename_csv = f'means_{args.data}.csv'

        # Read means CSV file
        means_df = pd.read_csv(filename_csv)

        # Randomly choose ten rows from means_df
        random_rows = np.random.choice(len(means_df), size=min(10, len(means_df)), replace=False)
        selected_means = means_df.iloc[random_rows].values
        means_df = means_df.drop(random_rows)
        # Save the modified dataframes back to the CSV files
        means_df.to_csv(filename_csv, index=False)

        # Initialize feature hooks for DeepInversion
        loss_r_feature_layers = []
        for module in model_gen.modules():
            if isinstance(module, nn.BatchNorm2d):
                loss_r_feature_layers.append(DeepInversionFeatureHook(module))
    
    # Initialize data for generation
    all_gen_input = []
    all_gen_labels = []
    
    # Setup dummy loader for non-MLRPTM methods or initial task
    if args.method != 'MLRPTM' or current_task == 0:
        inputs_dummy = torch.randn((len(train_loader) * args.batch_size_gen, 3, 8, 8),
                                  requires_grad=True,
                                  device='cuda', dtype=torch.float)
        targets_dummy = torch.LongTensor([0] * len(train_loader) * args.batch_size_gen).to('cuda')

        trainset_dummy = torch.utils.data.TensorDataset(inputs_dummy, targets_dummy)
        train_loader_synth = torch.utils.data.DataLoader(trainset_dummy, batch_size=args.batch_size_gen, drop_last=True,
                                                        num_workers=args.nThreads)
    
    # Main training loop
    for epoch in range(args.start, args.epochs + 1):
        running_loss = 0.0

        # Generate synthetic data for MLRPTM at the beginning of training
        if epoch == 0 and current_task > 0 and args.method == 'MLRPTM':
            print('=' * 50)
            print("Synthetic data generation starting...")

            for ii in range(len(train_loader)):
                data_type = torch.float
                inputs_d = torch.randn((args.batch_size_gen, 3, args.resolution, args.resolution),
                                      requires_grad=True,
                                      device='cuda', dtype=data_type)
                optimizer_gen = torch.optim.Adam([inputs_d], lr=args.lr_gen)
                optimizer_gen.state = collections.defaultdict(dict)
                lim_0, lim_1 = 6, 6
                prev_label = old_labels[current_task - 1]
                prev_label = list(set(prev_label))
                
                # Create targets based on number of classes in minibatch
                if num_classes_in_minibatch == 2:
                    targets = torch.LongTensor([prev_label[0]] * int(args.batch_size_gen / 2)
                                              + [prev_label[1]] * int(args.batch_size_gen / 2)).to('cuda')
                elif num_classes_in_minibatch == 4:
                    prev_label = sorted(list(set(prev_label)))
                    tr1 = random.sample(prev_label, k=4)
                    targets = torch.LongTensor(
                        [tr1[0]] * int(args.batch_size_gen / 4) + [tr1[1]] * int(args.batch_size_gen / 4)
                        + [tr1[2]] * int(args.batch_size_gen / 4) + [tr1[3]] * int(args.batch_size_gen / 4)).to(
                        'cuda')
                else:
                    print("Error: number of classes in each mini-batch must be 2 or 4")

                # Synthetic data generation loop
                for epoch2 in range(args.epoch_gen):
                    # Apply jittering to images
                    off1 = random.randint(-lim_0, lim_0)
                    off2 = random.randint(-lim_1, lim_1)
                    inputs_jit = torch.roll(inputs_d, shifts=(off1, off2), dims=(2, 3))

                    # Forward with jittered images
                    optimizer_gen.zero_grad()
                    model_gen.zero_grad()
                    model_gen = model_gen.cuda()
                    model_gen.eval()

                    embed_feat_gen = model_gen(inputs_jit)
                    embed_feat_gen = torch.squeeze(embed_feat_gen)
                    embed_feat_normal_gen = F.normalize(embed_feat_gen, p=2, dim=1)

                    # Calculate triplet loss
                    loss_gen, _, _, _ = criterion(embed_feat_normal_gen, targets)

                    # Calculate prior losses
                    loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)

                    # Calculate feature loss
                    rescale = [args.first_bn_mul] + [1. for _ in range(len(loss_r_feature_layers) - 1)]
                    loss_r_feature = sum(
                        [mod.r_feature * rescale[idx] for (idx, mod) in enumerate(loss_r_feature_layers)])

                    # L2 loss on images
                    loss_l2 = torch.norm(inputs_jit.view(args.BatchSize, -1), dim=1).mean()

                    # Combine losses
                    loss_aux = args.tv_l2 * loss_var_l2 + \
                              args.tv_l1 * loss_var_l1 + \
                              args.bn_reg_scale * loss_r_feature + \
                              args.l2 * loss_l2

                    loss_gen = args.main_mul * loss_gen + loss_aux

                    # Backprop and update
                    loss_gen.backward()
                    optimizer_gen.step()

                # Store generated data
                all_gen_input.append(inputs_d)
                all_gen_labels.append(targets)

            # Create dataset and loader for synthetic data
            ge_input = torch.cat(all_gen_input)
            ge_label = torch.cat(all_gen_labels)
            trainset = torch.utils.data.TensorDataset(ge_input, ge_label)
            train_loader_synth = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size_gen, drop_last=True,
                                                            num_workers=args.nThreads)
            print("Synthetic data generation completed.")

        # Batch training loop
        for jj, (img, img1) in enumerate(zip(train_loader, train_loader_synth), 0):
            # Prepare negative samples for MLRPTM
            if current_task > 0 and args.method == 'MLRPTM':
                generated_negative_samples = []
                for iii in range(len(selected_means)):
                    mean = selected_means[iii]
                    generated_negative_samples.append(mean)
            else:
                generated_negative_samples = []

            # Prepare inputs and synthetic samples
            inputs, labels = img
            inputs_synth, labels_synth = img1
            inputs_synth = Variable(inputs_synth.cuda())

            # Move to GPU and setup for training
            inputs = Variable(inputs.cuda())
            labels = Variable(labels).cuda()
            optimizer.zero_grad()
            
            # Forward pass
            embed_feat = model(inputs)
            embed_feat = torch.squeeze(embed_feat)
            embed_feat_normalize = F.normalize(embed_feat.clone(), p=2, dim=1)
            
            # Add noise if using noisy methods
            if args.method == 'MLRPTM' or args.method == 'NoisyFine_tuning':
                embed_feat_noisy = embed_feat_normalize.clone()
                random_perm2 = np.random.permutation(args.BatchSize)
                sample_index = random_perm2[:int((3 * args.BatchSize) / 4)]
                for j in range(0, int((3 * args.BatchSize) / 4)):
                    embed_feat_noisy[sample_index[j], :] = embed_feat_noisy[sample_index[j], :] \
                                                          + args.Noise_Power * torch.randn(1, args.dim, device='cuda')

            # Calculate augmentation loss based on training method
            if current_task == 0:
                loss_aug = 0 * torch.sum(embed_feat)
            elif current_task > 0:
                if args.method == 'Fine_tuning':
                    loss_aug = 0 * torch.sum(embed_feat)
                elif args.method == 'MLRPTM':
                    # Initialize losses
                    loss_aug = 0 * torch.sum(embed_feat)
                    loss_inter_class = 0.0
                    loss_intra_class = 0.0
                    loss_intra_cluster1 = 0.0
                    loss_intra_cluster2 = 0.0
                    loss_intra_cluster3 = 0.0
                    loss_intra_cluster4 = 0.0
                    loss_inter_class1 = 0.0
                    loss_inter_class2 = 0.0
                    loss_inter_class3 = 0.0
                    loss_inter_class4 = 0.0
                    loss_task = 0.0
                    de_num = 2

                    # Process old model features
                    embed_feat_old = model_old(inputs_synth)
                    embed_feat_old = torch.squeeze(embed_feat_old)
                    embed_feat_old_normal = F.normalize(embed_feat_old, p=2, dim=1)

                    # Process current model features for synthetic data
                    embed_feat_synth = model(inputs_synth)
                    embed_feat_synth = torch.squeeze(embed_feat_synth)
                    embed_feat_normal_synth = F.normalize(embed_feat_synth, p=2, dim=1)

                    # Calculate scatter losses
                    dis1, dis2, dis3, dis4, dis_mean, sample_c1, sample_c2, sample_c3, sample_c4, m1, m2, m3, m4, mt = \
                        scatter_loss(embed_feat_normal_synth, args.num_instances_gen, args.batch_size_gen)

                    dis1_T, dis2_T, dis3_T, dis4_T, dis_mean_T, sample_c1_T, sample_c2_T, sample_c3_T, sample_c4_T, \
                    m1_T, m2_T, m3_T, m4_T, mt_T = scatter_loss(embed_feat_old_normal, args.num_instances_gen, 
                                                               args.batch_size_gen)

                    # Intra-cluster loss calculations
                    loss_intra_cluster1 += torch.mean(torch.norm(sample_c1 - sample_c1_T, p=2, dim=1))
                    loss_intra_cluster2 += torch.mean(torch.norm(sample_c2 - sample_c2_T, p=2, dim=1))

                    # Center-aligned feature differences
                    Sc1 = sample_c1 - mt
                    Sc1_T = sample_c1_T - mt_T
                    Sc2 = sample_c2 - mt
                    Sc2_T = sample_c2_T - mt_T

                    # Inter-class loss calculations
                    loss_inter_class1 += torch.mean(torch.norm(Sc1 - Sc1_T, p=2, dim=1))
                    loss_inter_class2 += torch.mean(torch.norm(Sc2 - Sc2_T, p=2, dim=1))

                    # Handle 4-class case
                    if dis4 != [] and dis4_T != []:
                        Sc3 = sample_c3 - mt
                        Sc3_T = sample_c3_T - mt_T
                        Sc4 = sample_c4 - mt
                        Sc4_T = sample_c4_T - mt_T

                        loss_intra_cluster3 += torch.mean(torch.norm(sample_c3 - sample_c3_T, p=2, dim=1))
                        loss_intra_cluster4 += torch.mean(torch.norm(sample_c4 - sample_c4_T, p=2, dim=1))

                        loss_inter_class3 += torch.mean(torch.norm(Sc3 - Sc3_T, p=2, dim=1))
                        loss_inter_class4 += torch.mean(torch.norm(Sc4 - Sc4_T, p=2, dim=1))

                        de_num = 4

                    # MSE loss between features
                    mse_loss = F.mse_loss(embed_feat_synth, embed_feat_old)

                    # Combine losses
                    loss_intra_class += ((loss_intra_cluster1 + loss_intra_cluster2 + loss_intra_cluster3
                                         + loss_intra_cluster4) / de_num)

                    loss_inter_class += (
                            (loss_inter_class1 + loss_inter_class2 + loss_inter_class3 + loss_inter_class4) / de_num)
                    
                    # Add task confusion loss if not cifar10
                    if args.data != 'cifar10':
                        loss_task = criterion_task(embed_feat_normalize, labels, generated_negative_samples, args.method)

                    # Combine all augmentation losses
                    loss_aug += args.lambda_scatter * (loss_inter_class + loss_intra_class) + (
                            args.lambda_mse * mse_loss) + (args.lambda_task * loss_task)

            # Calculate main loss based on method
            if args.method == 'MLRPTM' or args.method == 'NoisyFine_tuning':
                loss, inter_, dist_ap, dist_an = criterion(embed_feat_noisy, labels)
            else:
                loss, inter_, dist_ap, dist_an = criterion(embed_feat_normalize, labels)

            # Add augmentation loss
            loss += loss_aug

            # Backprop and update
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.data
            
            # Print starting message
            if epoch == 0 and jj == 0:
                print('=' * 50)
                print('Training started...')

        # Print epoch summary
        print(f'[Epoch {epoch+1}/{args.epochs}] Total Loss: {running_loss/len(train_loader):.3f}')

        # Save checkpoint at specified intervals
        if epoch % args.save_step == 0: