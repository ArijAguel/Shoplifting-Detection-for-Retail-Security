import random
import numpy as np
import torch
from utils.scoring_utils import score_auc
#from torch.utils.tensorboard import SummaryWriter
from models.STG_NF.model_pose import STG_NF
from models.training import Trainer
from utils.data_utils import trans_list
from utils.optim_init import init_optimizer, init_scheduler
from args import create_exp_dirs
from args import init_parser, init_sub_args
from custom_dataloader import get_dataset_and_loader
from utils.train_utils import dump_args, init_model_params
from utils.train_utils import calc_num_of_params


def main():
    parser = init_parser()
    args = parser.parse_args(
        [
    "--batch_size", "128",
    '--device', 'cpu',
    '--temporal_kernel', '11',
    '--num_workers', '2',
    '--exp_dir', 'Modeling/experiment',
    '--seg_len', '30',
    '--seg_stride', '4',
    '--checkpoint', 'Modeling/Aug11_1146__checkpoint.pth.tar'
    ]
    )

    
    #I modified the device cuda0 instead of the default device cuda2
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    
    
    

    if args.seed == 999:  # Record and init seed
        args.seed = torch.initial_seed()
        np.random.seed(0)
    else:
        random.seed(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.manual_seed(args.seed)
        np.random.seed(0)

    args, model_args = init_sub_args(args)
    args.ckpt_dir = create_exp_dirs(args.exp_dir, dirmap=args.dataset)

    pretrained = vars(args).get('checkpoint', None)
    
    dataset, loader = get_dataset_and_loader(args, trans_list=trans_list, only_test=(pretrained is not None))

    if loader['train']:
        for data, trans_index, segs_score, label in loader['train']:
            print("IN TRAIN MODE")
            print(f"SHAPE OF DATA IS {data.shape}\n")
            print(f"SHAPE OF TRANS_INDEX IS {trans_index.shape}\n")
            print(f"SHAPE OF SEGS_SCORE IS {segs_score.shape}\n")
            print(f"SHAPE OF LABEL IS {label.shape}\n")
            break
    labels=list()
    for data,trans_index,scores, label in loader['test']:
    #    print("IN TEST MODE")
    #    print(f"SHAPE OF DATA IS {data.shape}\n")
    #    print(f"SHAPE OF TRANS_INDEX IS {trans_index.shape}\n")
    #    print(f"SHAPE OF SCORES IS {scores.shape}\n")
    #    print(f"SHAPE OF LABELS IS {label.shape}\n")
        labels.extend(label)

    print(f"LENTH OF LABELS IS {len(labels)}")

    model_args = init_model_params(args, dataset)
    model = STG_NF(**model_args)
    num_of_params = calc_num_of_params(model)
        
        
    trainer = Trainer(args, model, loader['train'], loader['test'], 
                    optimizer_f=init_optimizer(args.model_optimizer, lr=args.model_lr),
                    scheduler_f=init_scheduler(args.model_sched, lr=args.model_lr, epochs=args.epochs))
    if pretrained:
        trainer.load_checkpoint(pretrained)
    else:
        #writer = SummaryWriter()
        #trainer.train(log_writer=writer)
        dump_args(args, args.ckpt_dir)
    #Testing and scoring:
    normality_scores = trainer.test()
    assert not np.all(normality_scores == 1), "Error: Normality score is all ones!"
    assert not np.all(normality_scores == 0), "Error: Normality score is all zeros!"

    print( f"SHAPE OF NORMALITY SCORES IS {normality_scores.shape}" )
    #print(f"NORMALITY SCORES IS {normality_scores}")



    #auc_roc, scores_np, auc_pr, eer, eer_threshold = score_dataset(normality_scores, dataset["test"].metadata, args=args)
    auc_roc, auc_pr= score_auc(normality_scores, labels)
    # Logging and recording results
    print("\n-------------------------------------------------------")
    print('auc(roc): {}'.format(auc_roc))
    print('auc(pr): {}'.format(auc_pr))
    #print('eer: {}'.format(eer))
    #print('eer threshold: {}'.format(eer_threshold))
    #print('Number of samples', scores_np.shape[0])





if __name__ == '__main__':
    main()
