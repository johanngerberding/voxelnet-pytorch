import os 
import argparse
import torch 
import datetime 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

from config import get_cfg_defaults
from model import RPN3D
from dataset import KITTIDataset, collate_fn


def save_checkpoint(model, is_best, checkpoint_dir, epoch):
    out = os.path.join(checkpoint_dir, f"{str(epoch).zfill(3)}.pth")
    torch.save(model, out) 
    if is_best:
        out = os.path.join(checkpoint_dir, "best.pth")
        torch.save(model, out) 


def main():
    # argparse stuff  
    summary_interval = 10 # iterations!  
    print_interval = 10 # iterations 
    val_epoch = 10 
    start_epoch = 0 
    global_counter = 0 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = get_cfg_defaults()
    cfg.freeze()
    
    data_dir = cfg.DATA.DIR 
    train_data_dir = os.path.join(data_dir, 'training')
    val_data_dir = os.path.join(data_dir, 'test')

    train_dataset = KITTIDataset(train_data_dir, True, False)
    val_dataset = KITTIDataset(val_data_dir, False, True)     

    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=cfg.TRAIN.NUM_WORKERS,
        pin_memory=False,
    ) 

    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=cfg.VAL.NUM_WORKERS,
        pin_memory=False,
    )

    model = RPN3D('Car', cfg.TRAIN.ALPHA, cfg.TRAIN.BETA).to(device)
    model_checkpoint = None 
    resume = False
    exps_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exps")
    if not os.path.isdir(exps_dir):
        os.makedirs(exps_dir)

    date = datetime.datetime.now().strftime("%Y-%m-%d")
    exp_dir = os.path.join(exps_dir, date + "-000")
    i = 1 
    while os.path.isdir(exp_dir):
        exp_dir = os.path.join(exps_dir, date + f"-{str(i).zfill(3)}")
        i += 1  
    os.makedirs(exp_dir) 

    checkpoints_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.isdir(checkpoints_dir):
        os.makedirs(checkpoints_dir)

    log_dir = os.path.join(exp_dir, "logs")
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)

    if model_checkpoint and resume: 
        raise NotImplementedError
    
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [150])
    summary = SummaryWriter() 
    
    for epoch in range(start_epoch, cfg.TRAIN.NUM_EPOCHS):
        counter = 0  
        lr_scheduler.step()
        for (i, data) in enumerate(train_dataloader):
            counter += 1  
            global_counter += 1  
            model.train(True)
            _, _, loss, cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(data, device)
            loss.backward()
            # gradient clipping 
            clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            optimizer.zero_grad()

            if counter % print_interval == 0:
                print("Train: {} @ epoch: {}/{} - Loss: {:.4f} | Reg Loss: {:.4f} | Cls Loss: {:.4f}".format(
                    counter, epoch + 1, cfg.TRAIN.NUM_EPOCHS, loss.item(), reg_loss.item(), cls_loss.item()
                ))
            
            if counter % summary_interval == 0:
                summary.add_scalars(
                    str(epoch + 1), {
                        'train/loss': loss.item(),
                        'train/reg_loss': reg_loss.item(),
                        'train/cls_loss': cls_loss.item(),
                    }, global_counter
                )


    print("Training finished.")
    summary.close()






if __name__ == "__main__":
    main()