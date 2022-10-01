import os 
import argparse
import sys
import warnings 
import time  
import torch 
import datetime 
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tensorboardX import SummaryWriter

import cv2

from config import get_cfg_defaults
from model import RPN3D
from dataset import KITTIDataset, collate_fn
from utils import box3d_to_label

warnings.simplefilter("ignore")


def save_checkpoint(model, is_best, checkpoint_dir, epoch):
    out = os.path.join(checkpoint_dir, f"{str(epoch).zfill(3)}.pth")
    torch.save(model, out) 
    if is_best:
        out = os.path.join(checkpoint_dir, "best.pth")
        torch.save(model, out) 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_interval", type=int, default=100)
    parser.add_argument("--print_interval", type=int, default=100)
    parser.add_argument("--val_epoch", type=int, default=1)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--global_counter", type=int, default=0)
    parser.add_argument("--summary_val_interval", type=int, default=100)
    parser.add_argument("--vis", type=bool, default=True)
    parser.add_argument("--num_vis_dump", type=int, default=50)
    parser.add_argument("--gpu", type=bool, default=True)
    parser.add_argument("--model_checkpoint", type=str, default="")
    parser.add_argument("--resume", type=bool, default=False)
    parser.add_argument("--cfg", type=str, default=None) 

    args = parser.parse_args()
    global_counter = args.global_counter 
    
    device = torch.device(
            "cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    min_loss = sys.float_info.max
    cfg = get_cfg_defaults()
    if args.cfg: 
        cfg.merge_from_file(args.cfg)
    cfg.freeze()
    print(cfg)

    data_dir = cfg.DATA.DIR 
    train_data_dir = os.path.join(data_dir, 'training')
    val_data_dir = os.path.join(data_dir, 'validation')
    
    train_dataset = KITTIDataset(
        data_dir=train_data_dir, 
        shuffle=True, 
        augment=False, 
        test=False, 
    )
    print(f"Len Train Dataset: {len(train_dataset)}") 
    val_dataset = KITTIDataset(
        data_dir=val_data_dir, 
        shuffle=False, 
        augment=False, 
        test=False,
    )     
    print(f"Len Val Dataset: {len(val_dataset)}")

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

    val_dataloader_iter = iter(val_dataloader)

    model = RPN3D(cfg.OBJECT.NAME, cfg.TRAIN.ALPHA, cfg.TRAIN.BETA).to(device)
    
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
    os.makedirs(checkpoints_dir, exist_ok=True)

    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    vis_output_dir = os.path.join(exp_dir, "vis") 
    os.makedirs(vis_output_dir, exist_ok=True) 
    
    # model predictions output dir
    preds_dir = os.path.join(exp_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "config.yaml"), 'w') as fp:
        fp.write(cfg.dump())

    if args.model_checkpoint and args.resume: 
        raise NotImplementedError
    
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, [cfg.TRAIN.LR_SCHEDULER_STEP])
    summary = SummaryWriter() 

    for epoch in range(args.start_epoch, cfg.TRAIN.NUM_EPOCHS):
        epoch_start = time.time() 
        print("-" * 30) 
        print(f"Epoch {epoch+1}") 
        print("-" * 30) 
        counter = 0 

        tot_val_loss = 0
        tot_val_times = 0
        print_time_start = time.time() 
        for (i, data) in enumerate(train_dataloader):
            counter += 1  
            global_counter += 1  
            model.train(True)

            _, _, loss, cls_loss, reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(data, device)
            loss.backward()
            # gradient clipping 
            clip_grad_norm_(model.parameters(), cfg.TRAIN.GRADIENT_CLIP)
            optimizer.step()
            optimizer.zero_grad()

            if counter % args.print_interval == 0:
                print_time_elapsed = time.time() - print_time_start 
                print("Train: {} @ epoch: {}/{} - Loss: {:.4f} | Reg Loss: {:.4f} | Cls Loss: {:.4f} | Time: {} mins".format(
                    counter, epoch + 1, cfg.TRAIN.NUM_EPOCHS, loss.item(), reg_loss.item(), cls_loss.item(), int(print_time_elapsed / 60)
                ))
                print_time_start = time.time()
            
            if counter % args.summary_interval == 0:
                summary.add_scalars(
                    str(epoch + 1), {
                        'train/loss': loss.item(),
                        'train/reg_loss': reg_loss.item(),
                        'train/cls_loss': cls_loss.item(),
                    }, global_counter
                )

            if counter % args.summary_val_interval == 0:
                
                with torch.no_grad():
                    model.eval()
                    val_data = next(val_dataloader_iter)
                    probs, deltas, val_loss, val_cls_loss, val_reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(val_data, device)

                    summary.add_scalars(str(epoch + 1), {
                        'validate/loss': loss.item(), 
                        'validate/reg_loss': reg_loss.item(),
                        'validate/cls_loss': cls_loss.item(),
                    })

                    try: 
                        tags, ret_box3d_scores, ret_summary = model.predict(
                            val_data, probs, deltas, summary=True, visual=True)

                        for (tag, img) in ret_summary:
                            img = img[0].transpose(2, 0 ,1)
                            summary.add_image(tag, img, global_counter)
                    except:
                        raise Exception("Prediction skipped due to an error!")
                    
                    tot_val_loss += val_loss.item()
                    tot_val_times += 1 

        avg_val_loss = tot_val_loss / float(tot_val_times)
        is_best = avg_val_loss < min_loss 
        min_loss = min(avg_val_loss, min_loss)
        save_checkpoint(model, is_best, checkpoints_dir, epoch)
        
        vis_count = 0
        if (epoch + 1) % args.val_epoch == 0:   # Time consuming

            model.train(False)  # Validation mode
            
            with torch.no_grad():
                for (i, val_data) in enumerate(val_dataloader):
                    # Forward pass for validation and prediction
                    probs, deltas, val_loss, val_cls_loss, val_reg_loss, cls_pos_loss_rec, cls_neg_loss_rec = model(val_data, device)

                    front_images, bird_views, heatmaps = None, None, None
                    if args.vis:
                        tags, ret_box3d_scores, front_images, bird_views, heatmaps = \
                            model.predict(val_data, probs, deltas, summary=False, visual=True)
                    else:
                        tags, ret_box3d_scores = model.predict(
                            val_data, probs, deltas, summary=False, visual=False)

                    # tags: (N)
                    # ret_box3d_scores: (N, N'); (class, x, y, z, h, w, l, rz, score)
                    for tag, score in zip(tags, ret_box3d_scores):
                        output_path = os.path.join(preds_dir, str(epoch + 1), 'data', tag + '.txt')
                        vis_outdir = os.path.split(output_path)[0]
                        os.makedirs(vis_outdir, exist_ok=True) 
                        
                        with open(output_path, 'w+') as f:
                            labels = box3d_to_label([score[:, 1:8]], [score[:, 0]], [score[:, -1]], coordinate = 'lidar')[0]
                            for line in labels:
                                f.write(line)
                    
                    # Dump visualizations
                    if args.vis and vis_count < args.num_vis_dump:
                        for tag, front_image, bird_view, heatmap in zip(tags, front_images, bird_views, heatmaps): 
                            vis_count += 1 
                            front_img_path = os.path.join(
                                vis_output_dir, 
                                str(epoch + 1), 
                                tag + '_front.jpg',
                            )
                            bird_view_path = os.path.join(
                                vis_output_dir, 
                                str(epoch + 1), 
                                tag + '_bv.jpg',
                            )
                            heatmap_path = os.path.join(
                                vis_output_dir, 
                                str(epoch + 1), 
                                tag + '_heatmap.jpg',
                            )
                            os.makedirs(
                                os.path.split(heatmap_path)[0], 
                                exist_ok=True,
                            ) 
                            cv2.imwrite(front_img_path, front_image)
                            cv2.imwrite(bird_view_path, bird_view)
                            cv2.imwrite(heatmap_path, heatmap)
                    
        lr_scheduler.step()
        epoch_elapsed = time.time() - epoch_start
        print("Epoch {} time: {} seconds".format(epoch + 1, epoch_elapsed))

    print("Training finished.")
    summary.close()


if __name__ == "__main__":
    main()
