#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MLRPTM Testing Module

This module provides functionality for evaluating MLRPTM models on
continual learning tasks. It implements methods for extracting features,
calculating classification accuracy, and visualizing confusion matrices.

Author: Your Name
Email: your.email@example.com
"""

from __future__ import absolute_import, print_function
import os
import sys
import warnings
import argparse
from typing import List, Tuple, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.backends import cudnn
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import confusion_matrix
import itertools

# Local imports
from utils import setup_logger, set_random_seed
from image_folder import ImageFolder

# Suppress deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_features(model: nn.Module, data_loader: torch.utils.data.DataLoader) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract normalized feature embeddings from a dataset using the provided model.
    
    Args:
        model: Neural network model used for feature extraction
        data_loader: DataLoader containing the dataset
        
    Returns:
        Tuple containing (features, labels) as numpy arrays
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
        
        if not features:
            features = outputs
            labels = pids
        else:
            features = np.vstack((features, outputs))
            labels = np.hstack((labels, pids))
    
    return features, labels


def plot_confusion_matrix(
    true_labels: np.ndarray, 
    predicted_labels: np.ndarray, 
    classes: np.ndarray,
    normalize: bool = False,
    title: str = 'Confusion matrix',
    cmap: plt.cm = plt.cm.Blues,
    fontsize: int = 12,
    save_path: Optional[str] = None
) -> None:
    """
    Generate and plot a confusion matrix visualization.
    
    Args:
        true_labels: Ground truth labels
        predicted_labels: Predicted labels
        classes: List of class labels
        normalize: Whether to normalize the confusion matrix
        title: Title for the plot
        cmap: Colormap for the plot
        fontsize: Font size for text elements
        save_path: If provided, save the figure to this path instead of displaying
    """
    cm = confusion_matrix(true_labels, predicted_labels)
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=fontsize+2)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=fontsize)
    plt.yticks(tick_marks, classes, fontsize=fontsize)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j, i, format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
            fontsize=fontsize
        )
    
    plt.ylabel('True label', fontsize=fontsize)
    plt.xlabel('Predicted label', fontsize=fontsize)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    else:
        plt.show()


def get_transform_config(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose, str, str, int]:
    """
    Get dataset-specific configurations for transforms and paths.
    
    Args:
        dataset_name: Name of the dataset ('cifar10', 'cifar100', etc.)
        
    Returns:
        Tuple containing (transform_train, transform_test, traindir, testdir, num_classes)
    """
    if dataset_name == "cifar100":
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                (0.2023, 0.1994, 0.2010)),
        ])
        root = os.path.join('DataSet', 'cifar100')
        num_classes = 100
        
    elif dataset_name == "cifar10":
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                (0.24703233, 0.24348505, 0.26158768)),
        ])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.49139968, 0.48215827, 0.44653124),
                                (0.24703233, 0.24348505, 0.26158768)),
        ])
        root = os.path.join('DataSet', 'cifar10')
        num_classes = 10
        
    elif dataset_name == 'mini-imagenet-100':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
        ])
        root = os.path.join('DataSet', 'mini-imagenet-100')
        num_classes = 100
        
    elif dataset_name == 'tiny-imagenet-200':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406),
                                (0.229, 0.224, 0.225)),
        ])
        root = os.path.join('DataSet', 'tiny-imagenet-200')
        num_classes = 200
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    traindir = os.path.join(root, 'train')
    testdir = os.path.join(root, 'test')
    
    return transform_train, transform_test, traindir, testdir, num_classes


def main():
    """Main function to execute the testing process."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MLRPTM Testing')
    
    parser.add_argument('--data', type=str, default='tiny-imagenet-200',
                        help='Path to dataset (cifar10, cifar100, mini-imagenet-100, tiny-imagenet-200)')
    parser.add_argument('--models_dir', '-r', type=str, default='checkpoints/Tiny-imagenet-200',
                        metavar='PATH', help='Directory where trained models are saved')
    parser.add_argument('--gpu', type=str, default='0', 
                        help='GPU device ID to use')
    parser.add_argument('--seed', default=1993, type=int, metavar='N',
                        help='Random seed for reproducibility')
    parser.add_argument('--epochs', default=50, type=int, metavar='N',
                        help='Number of epochs used in training')
    parser.add_argument('--task', default=20, type=int,
                        help='Number of tasks')
    parser.add_argument('--base', default=100, type=int,
                        help='Number of classes in non-incremental state')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results and visualizations')
    parser.add_argument('--save_cm', action='store_true',
                        help='Save confusion matrices to disk')
    
    args = parser.parse_args()
    
    # Set up environment
    os.makedirs(args.output_dir, exist_ok=True)
    logger = setup_logger(os.path.join(args.output_dir, 'test_log.txt'))
    cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    set_random_seed(args.seed)
    
    # Get trained model paths
    models_paths = []
    for filename in os.listdir(args.models_dir):
        if filename.endswith(f"{args.epochs}_model.pt"):
            models_paths.append(os.path.join(args.models_dir, filename))
    
    models_paths.sort()
    # Handle special case for task > 10
    if args.task > 10:
        models_paths.append(models_paths[1])
        del models_paths[1]
    
    # Get dataset configuration
    transform_train, transform_test, traindir, testdir, num_classes = get_transform_config(args.data)
    
    # Task configuration
    num_class_per_task = int((num_classes - args.base) / (args.task - 1))
    np.random.seed(args.seed)
    random_perm = np.random.permutation(num_classes)
    
    logger.info('=== Test started ===')
    
    # Initialize metrics storage
    class_means = []
    class_stds = []
    class_labels = []
    
    # Results dataframe
    results_df = pd.DataFrame(columns=[
        'Task_ID', 'Model_ID', 'Class_Range', 'Accuracy', 'Num_Samples'
    ])
    
    # Evaluate each task
    for task_id in range(args.task):
        # Define class indices for this task
        index = random_perm[:args.base + task_id * num_class_per_task]
        
        if task_id == 0:
            index_train = random_perm[:args.base]
        else:
            index_train = random_perm[
                args.base + (task_id - 1) * num_class_per_task:
                args.base + task_id * num_class_per_task
            ]
        
        # Create data loaders
        trainfolder = ImageFolder(traindir, transform_train, index=index_train)
        testfolder = ImageFolder(testdir, transform_test, index=index)
        
        train_loader = torch.utils.data.DataLoader(
            trainfolder, batch_size=128, shuffle=False, drop_last=False)
        test_loader = torch.utils.data.DataLoader(
            testfolder, batch_size=128, shuffle=False, drop_last=False)
        
        if task_id != 0:
            logger.info(f'Testing Task {task_id}')
        
        # Load model
        model_res = models.resnet18(pretrained=False)
        model = nn.Sequential(*list(model_res.children())[:-1])
        state = torch.load(models_paths[task_id])
        model.load_state_dict(state['state_dict'])
        
        # Extract features
        train_embeddings, train_labels = extract_features(model, train_loader)
        val_embeddings, val_labels = extract_features(model, test_loader)
        
        # Calculate class means and store them
        mean_data_csv = []
        var_data_csv = []
        
        for class_idx in index_train:
            ind_cl = np.where(class_idx == train_labels)[0]
            embeddings_tmp = train_embeddings[ind_cl]
            class_labels.append(class_idx)
            
            # Calculate and store class mean and variance
            class_mean = np.mean(embeddings_tmp, axis=0)
            class_means.append(class_mean)
            
            mean_data_csv.append([class_idx] + list(class_mean.flatten()))
            var_data_csv.append([class_idx] + list(np.var(embeddings_tmp, axis=0).flatten()))
        
        # Evaluate model on all previous tasks
        embedding_mean_old = []
        embedding_std_old = []
        gt_all = []
        estimate_all = []
        
        acc_ave = 0
        for k in range(task_id + 1):
            # Get classes for task k
            if k == 0:
                task_classes = random_perm[:args.base]
            else:
                task_classes = random_perm[
                    args.base + (k - 1) * num_class_per_task:
                    args.base + k * num_class_per_task
                ]
            
            # Filter test samples for this task's classes
            gt_mask = np.isin(val_labels, task_classes)
            
            # Calculate distances to class means and classify
            pairwise_distance = euclidean_distances(val_embeddings, np.asarray(class_means))
            estimate_indices = np.argmin(pairwise_distance, axis=1)
            estimate_labels = [class_labels[j] for j in estimate_indices]
            estimate_filtered = np.asarray(estimate_labels)[gt_mask]
            
            # Save results for final task
            if task_id == args.task - 1:
                if not gt_all:
                    estimate_all = estimate_filtered
                    gt_all = val_labels[gt_mask]
                else:
                    estimate_all = np.hstack((estimate_all, estimate_filtered))
                    gt_all = np.hstack((gt_all, val_labels[gt_mask]))
            
            # Calculate accuracy
            acc = np.sum(estimate_filtered == val_labels[gt_mask]) / float(len(estimate_filtered))
            
            # Weight accuracy by class proportion
            if k == 0:
                acc_ave += acc * (float(args.base) / (args.base + task_id * num_class_per_task))
            else:
                acc_ave += acc * (float(num_class_per_task) / (args.base + task_id * num_class_per_task))
            
            # Generate confusion matrix for visualization (skip base task)
            if k != 0 and task_id != 0:
                classes = np.unique(index)
                
                if args.save_cm:
                    cm_path = os.path.join(
                        args.output_dir, 
                        f'confusion_matrix_model{task_id}_task{k}.png'
                    )
                    plot_confusion_matrix(
                        val_labels[gt_mask], 
                        estimate_filtered, 
                        classes,
                        title=f'Confusion Matrix: Model {task_id} on Task {k}',
                        save_path=cm_path
                    )
                else:
                    plot_confusion_matrix(
                        val_labels[gt_mask], 
                        estimate_filtered, 
                        classes,
                        title=f'Confusion Matrix: Model {task_id} on Task {k}'
                    )
            
            # Store results
            results_df = results_df.append({
                'Task_ID': k,
                'Model_ID': task_id,
                'Class_Range': f"{task_classes.min()}-{task_classes.max()}",
                'Accuracy': acc * 100,
                'Num_Samples': len(estimate_filtered)
            }, ignore_index=True)
            
            if task_id != 0:
                logger.info(f"Accuracy of Model {task_id} on Task {k}: {acc:.3f}")
        
        if task_id != 0:
            logger.info(f"Weighted Accuracy of Model {task_id}: {acc_ave:.3f}")
    
    # Save results
    results_df.to_csv(os.path.join(args.output_dir, 'test_results.csv'), index=False)
    
    # Generate summary statistics
    summary = results_df.groupby('Model_ID')['Accuracy'].agg(['mean', 'std', 'min', 'max'])
    summary.to_csv(os.path.join(args.output_dir, 'accuracy_summary.csv'))
    
    logger.info("=== Testing completed successfully ===")
    logger.info(f"Results saved to {args.output_dir}")


if __name__ == '__main__':
    main()
