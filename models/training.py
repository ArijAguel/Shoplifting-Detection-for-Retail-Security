"""
Train\Test helper, based on awesome previous work by https://github.com/amirmk89/gepc
"""
#trainer class

import os
import time
import shutil
import torch
import torch.optim as optim
from tqdm import tqdm

# dynamically Adjust learning rate based on epoch, decay factor, or scheduler
def adjust_lr(optimizer, epoch, lr=None, lr_decay=None, scheduler=None):
    if scheduler is not None:
        scheduler.step()
        new_lr = scheduler.get_lr()[0]
    elif (lr is not None) and (lr_decay is not None):
        new_lr = lr * (lr_decay ** epoch)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
    else:
        raise ValueError('Missing parameters for LR adjustment')
    return new_lr


def compute_loss(nll, reduction="mean", mean=0):
    if reduction == "mean":
        losses = {"nll": torch.mean(nll)}
    elif reduction == "logsumexp":
        losses = {"nll": torch.logsumexp(nll, dim=0)}
    elif reduction == "exp":
        losses = {"nll": torch.exp(torch.mean(nll) - mean)}
    elif reduction == "none":
        losses = {"nll": nll}

    losses["total_loss"] = losses["nll"]

    return losses


class Trainer: #Manages training and testing loops, checkpoints, logging
    def __init__(self, args, model, train_loader, test_loader,
                 optimizer_f=None, scheduler_f=None):
        self.model = model
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        # Loss, Optimizer and Scheduler
        if optimizer_f is None:
            self.optimizer = self.get_optimizer()
        else:
            self.optimizer = optimizer_f(self.model.parameters())
        if scheduler_f is None:
            self.scheduler = None
        else:
            self.scheduler = scheduler_f(self.optimizer)
        
        # Add wandb logger attribute
        self.wandb_logger = None

    def get_optimizer(self):
        if self.args.optimizer == 'adam':
            if self.args.lr:
                return optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adam(self.model.parameters())
        elif self.args.optimizer == 'adamx':
            if self.args.lr:
                return optim.Adamax(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
            else:
                return optim.Adamax(self.model.parameters())
        return optim.SGD(self.model.parameters(), lr=self.args.lr)

    def adjust_lr(self, epoch):
        return adjust_lr(self.optimizer, epoch, self.args.model_lr, self.args.model_lr_decay, self.scheduler)

    def save_checkpoint(self, epoch, is_best=False, filename=None):
        """
        state: {'epoch': cur_epoch + 1, 'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict()}
        """
        state = self.gen_checkpoint_state(epoch)
        if filename is None:
            filename = 'checkpoint.pth.tar'

        state['args'] = self.args

        path_join = os.path.join(self.args.ckpt_dir, filename)
        torch.save(state, path_join)
        if is_best:
            shutil.copy(path_join, os.path.join(self.args.ckpt_dir, 'checkpoint_best.pth.tar'))
        
        # Log checkpoint to wandb
        if self.wandb_logger:
            self.wandb_logger.save(path_join)

    def load_checkpoint(self, filename):
        filename = filename
        try:
            checkpoint = torch.load(filename)
            self.start_epoch = checkpoint['epoch']
            self.model.load_state_dict(checkpoint['state_dict'], strict=False)
            self.model.set_actnorm_init()
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("Checkpoint loaded successfully from '{}' at (epoch {})\n"
                  .format(filename, checkpoint['epoch']))
        except FileNotFoundError:
            print("No checkpoint exists from '{}'. Skipping...\n".format(self.args.ckpt_dir))

    def train(self, log_writer=None, wandb_logger=None, clip=100):
        # Store wandb logger
        self.wandb_logger = wandb_logger
        
        time_str = time.strftime("%b%d_%H%M_")
        checkpoint_filename = time_str + '_checkpoint.pth.tar'
        start_epoch = 0
        num_epochs = self.args.epochs
        self.model.train()
        self.model = self.model.to('cpu')
        key_break = False
        
        for epoch in range(start_epoch, num_epochs):
            if key_break:
                break
            print("Starting Epoch {} / {}".format(epoch + 1, num_epochs))
            
            # Track epoch metrics
            epoch_losses = []
            
            pbar = tqdm(self.train_loader)
            for itern, data_arr in enumerate(pbar):
                try:
                    data = [data.to('cpu', non_blocking=True) for data in data_arr]
                    score = data[-2].amin(dim=-1)
                    label = data[-1]
                    if self.args.model_confidence:
                        samp = data[0]
                    else:
                        samp = data[0][:, :2]
                    z, nll = self.model(samp.float(), label=label, score=score)
                    if nll is None:
                        continue
                    if self.args.model_confidence:
                        nll = nll * score
                    losses = compute_loss(nll, reduction="mean")["total_loss"]
                    losses.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # Store loss for epoch averaging
                    epoch_losses.append(losses.item())
                    
                    pbar.set_description("Loss: {:.6f}".format(losses.item()))
                    
                    # Log to tensorboard (existing)
                    if log_writer:
                        log_writer.add_scalar('NLL Loss', losses.item(), epoch * len(self.train_loader) + itern)
                    
                    # Log batch metrics to wandb (every 100 iterations to avoid spam)
                    if self.wandb_logger and itern % 100 == 0:
                        self.wandb_logger.log({
                            "batch_loss": losses.item(),
                            "epoch": epoch,
                            "iteration": epoch * len(self.train_loader) + itern,
                            "learning_rate": self.optimizer.param_groups[0]['lr']
                        })

                except KeyboardInterrupt:
                    print('Keyboard Interrupted. Save results? [yes/no]')
                    choice = input().lower()
                    if choice == "yes":
                        key_break = True
                        break
                    else:
                        exit(1)

            # Calculate and log epoch metrics
            if epoch_losses:
                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                
                # Log epoch metrics to wandb
                if self.wandb_logger:
                    self.wandb_logger.log({
                        "epoch": epoch,
                        "epoch_avg_loss": avg_epoch_loss,
                        "epoch_total_batches": len(epoch_losses),
                        "learning_rate": self.optimizer.param_groups[0]['lr']
                    })

            self.save_checkpoint(epoch, filename=checkpoint_filename)
            new_lr = self.adjust_lr(epoch)
            print('Checkpoint Saved. New LR: {0:.3e}'.format(new_lr))

    def test(self):
        self.model.eval()
        self.model.to('cpu')
        pbar = tqdm(self.test_loader)
        probs = torch.empty(0).to('cpu')
        print("Starting Test Eval")
        
        test_losses = []  # Track test losses for logging
        
        for itern, data_arr in enumerate(pbar):
            data = [data.to('cpu', non_blocking=True) for data in data_arr]
            score = data[-2].amin(dim=-1)
            if self.args.model_confidence:
                samp = data[0]
            else:
                samp = data[0][:, :2]
            with torch.no_grad():
                z, nll = self.model(samp.float(), label=torch.ones(data[0].shape[0]), score=score)
            if self.args.model_confidence:
                nll = nll * score
            
            # Store individual losses for analysis
            test_losses.extend(nll.cpu().detach().numpy().tolist())
            
            probs = torch.cat((probs, -1 * nll), dim=0)
        
        prob_mat_np = probs.cpu().detach().numpy().squeeze().copy(order='C')
        
        # Log test statistics to wandb
        if self.wandb_logger and test_losses:
            import numpy as np
            self.wandb_logger.log({
                "test_samples": len(test_losses),
                "test_loss_mean": np.mean(test_losses),
                "test_loss_std": np.std(test_losses),
                "test_loss_min": np.min(test_losses),
                "test_loss_max": np.max(test_losses)
            })
        
        return prob_mat_np

    def gen_checkpoint_state(self, epoch):
        checkpoint_state = {'epoch': epoch + 1,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(), }
        return checkpoint_state