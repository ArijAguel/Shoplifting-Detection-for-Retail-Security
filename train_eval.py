#cuda activate STG-NF
#python train_eval.py --dataset PoseLift --device cpu

""" hyperparameters

defaults:
--model_lr 5e-4 --batch_size 256 --K 8 --L 1 --R 3.0 --epochs 8 --model_weight_decay 5e-5

Faster Training (Higher LR, Larger Batch), Speed up convergence
--batch_size 512 --model_lr 1e-3 --model_lr_decay 0.95 --epochs 10

Better Generalization (Regularization Focus), Reduce overfitting
--batch_size 128 --model_lr 5e-4 --model_weight_decay 1e-4 --epochs 20

"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # This forces CPU usage
import random
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
import wandb
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from dataset import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.scoring_utils import score_dataset
from utils.train_utils import calc_num_of_params
from sklearn.metrics import precision_recall_curve, auc
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import pandas as pd
import subprocess
import sys

def main():
    parser = init_parser()
    args = parser.parse_args()

    # Initialize wandb
    wandb.init(
        project="STG-NF-PoseLift",  # Change this to your project name
        name=f"run_{args.dataset}",  # Optional: give your run a name
        config={
            "dataset": args.dataset,
            "seed": args.seed,
            "device": args.device,
            "model_lr": getattr(args, 'model_lr', None),
            "epochs": getattr(args, 'epochs', None),
            "model_optimizer": getattr(args, 'model_optimizer', None),
            "model_sched": getattr(args, 'model_sched', None),
        },
        tags=["pose-estimation", "anomaly-detection"]  # Optional tags
    )

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    # Update wandb config with the actual seed used
    wandb.config.update({"actual_seed": args.seed})

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('checkpoint', None)
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    model_args = init_model_params(args, dataset)
    
    # Log model parameters to wandb
    wandb.config.update(model_args)
    
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
    
    # Log model info
    wandb.config.update({
        "model_parameters": num_of_params,
        "train_samples": len(dataset["train"]) if "train" in dataset else 0,
        "test_samples": len(dataset["test"]) if "test" in dataset else 0
    })
    
    print(f"Model has {num_of_params} parameters")
    
    trainer = Trainer(args, model, loader['train'], loader['test'], 
                      optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                      scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    
    if pretrained:
        trainer.load_checkpoint(pretrained)
        wandb.config.update({"pretrained_checkpoint": pretrained})
    else:
        writer = SummaryWriter()
        # Pass wandb to trainer for logging during training
        trainer.train(log_writer=writer, wandb_logger=wandb)
        dump_args(args, args.ckpt_dir)

    # Testing and scoring:
    print("Starting testing...")
    normality_scores = trainer.test()
    
    auc_roc, scores_np, auc_pr, eer, eer_threshold = score_dataset(
        normality_scores, dataset["test"].metadata, args=args
    )

    # Log final metrics to wandb
    wandb.log({
        "final_auc_roc": auc_roc,
        "final_auc_pr": auc_pr,
        "final_eer": eer,
        "final_eer_threshold": eer_threshold,
        "num_test_samples": scores_np.shape[0]
    })

    # Create and log plots
    if len(normality_scores) > 0:
        # Log histogram of anomaly scores
        wandb.log({
            "anomaly_scores_histogram": wandb.Histogram(scores_np)
        })
        
    # Create and log plots
    if len(normality_scores) > 0:
        # Log histogram of anomaly scores
        wandb.log({
            "anomaly_scores_histogram": wandb.Histogram(scores_np)
        })
        
        # Try to extract labels for detailed plotting
        try:
            labels = None
            
            # Debug: Print dataset structure
            print(f"Dataset test type: {type(dataset['test'])}")
            print(f"Dataset test attributes: {[attr for attr in dir(dataset['test']) if not attr.startswith('_')]}")
            
            # Try to access metadata
            if hasattr(dataset["test"], 'metadata'):
                metadata = dataset["test"].metadata
                print(f"Metadata type: {type(metadata)}")
                
                # If metadata is a numpy array, it might be the labels directly
                if isinstance(metadata, np.ndarray):
                    print(f"Metadata is numpy array with shape: {metadata.shape}")
                    if len(metadata) == len(scores_np):
                        labels = metadata
                        print("Using metadata as labels directly")
                    else:
                        print(f"Metadata length {len(metadata)} doesn't match scores length {len(scores_np)}")
                
                # If metadata is a list
                elif isinstance(metadata, list):
                    print(f"Metadata is list with length: {len(metadata)}")
                    if len(metadata) == len(scores_np):
                        labels = np.array(metadata)
                        print("Using metadata list as labels")
                
                # If metadata is a dict
                elif isinstance(metadata, dict):
                    print(f"Metadata is dict with keys: {list(metadata.keys())}")
                    # Try common label keys
                    for key in ['labels', 'label', 'y', 'targets', 'ground_truth', 'gt']:
                        if key in metadata:
                            labels = np.array(metadata[key])
                            print(f"Found labels using key: {key}")
                            break
            
            # Try other common attributes
            if labels is None:
                for attr in ['labels', 'targets', 'y']:
                    if hasattr(dataset["test"], attr):
                        attr_value = getattr(dataset["test"], attr)
                        print(f"Found {attr} attribute of type: {type(attr_value)}")
                        if hasattr(attr_value, '__len__') and len(attr_value) == len(scores_np):
                            labels = np.array(attr_value)
                            print(f"Using {attr} as labels")
                            break
            
            # If we found labels, create plots
            if labels is not None:
                labels = np.array(labels)
                unique_labels = np.unique(labels)
                print(f"Found labels with shape: {labels.shape}")
                print(f"Unique label values: {unique_labels}")
                
                if len(labels) == len(scores_np):
                    # Log ROC curve
                    from sklearn.metrics import roc_curve
                    fpr, tpr, _ = roc_curve(labels, scores_np)
                    wandb.log({
                        "roc_curve": wandb.plot.line_series(
                            xs=fpr.tolist(),
                            ys=[tpr.tolist()],
                            keys=["ROC Curve"],
                            title="ROC Curve",
                            xname="False Positive Rate"
                        )
                    })
                    
                    # Log PR curve
                    precision, recall, _ = precision_recall_curve(labels, scores_np)
                    wandb.log({
                        "pr_curve": wandb.plot.line_series(
                            xs=recall.tolist(),
                            ys=[precision.tolist()], 
                            keys=["PR Curve"],
                            title="Precision-Recall Curve",
                            xname="Recall"
                        )
                    })
                    
                    print("Successfully created ROC and PR curves")
                else:
                    print(f"Label length {len(labels)} doesn't match scores length {len(scores_np)}")
            else:
                print("No labels found for detailed plotting")
                
        except Exception as e:
            print(f"Error in plotting section: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Always log basic score statistics regardless of labels
        wandb.log({
            "score_statistics": {
                "mean": float(np.mean(scores_np)),
                "std": float(np.std(scores_np)),
                "min": float(np.min(scores_np)),
                "max": float(np.max(scores_np)),
                "median": float(np.median(scores_np))
            }
        })

    # Save model artifacts
    if hasattr(trainer, 'model'):
        model_path = os.path.join(args.ckpt_dir, "final_model.pth")
        torch.save(trainer.model.state_dict(), model_path)
        wandb.save(model_path)

    # Logging and recording results
    print("\n-------------------------------------------------------")
    print('auc(roc): {}'.format(auc_roc))
    print('auc(pr): {}'.format(auc_pr))
    print('eer: {}'.format(eer))
    print('eer threshold: {}'.format(eer_threshold))
    print('Number of samples', scores_np.shape[0])
    
    # Create summary table
    results_table = wandb.Table(
        columns=["Metric", "Value"],
        data=[
            ["AUC-ROC", auc_roc],
            ["AUC-PR", auc_pr], 
            ["EER", eer],
            ["EER Threshold", eer_threshold],
            ["Test Samples", scores_np.shape[0]]
        ]
    )
    wandb.log({"results_summary": results_table})
    
    # Finish wandb run
    wandb.finish()

if __name__ == '__main__':
    main()