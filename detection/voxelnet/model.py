import os 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

from config import get_cfg_defaults
from utils import (
    center_to_corner_box_2d,
    colorize, 
    corner_to_standup_box2d,
    draw_lidar_box_3d_on_birdview, 
    label_to_gt_box_3d,
    lidar_to_bird_view_image, 
    nms, 
    generate_anchors, 
    generate_targets, 
    deltas_to_boxes_3d,
    load_calib, 
    draw_lidar_box_3d_on_image,
)
from loss import smooth_L1_loss 


cfg = get_cfg_defaults()

class VFELayer(nn.Module):
    def __init__(self, cin: int, cout: int):
        super(VFELayer, self).__init__()

        self.in_channels = cin 
        self.out_channels = cout
        self.local_agg_features = cout // 2 

        self.fcn = nn.Sequential(
            nn.Linear(self.in_channels, self.local_agg_features),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(self.local_agg_features)


    def forward(self, inputs, mask):
        temp = self.fcn(inputs).transpose(1,2)
        pointwise_input = self.bn(temp).transpose(1,2) 
        agg, _ = torch.max(pointwise_input, dim=1, keepdim=True)
        repeat = agg.expand(-1, cfg.OBJECT.POINTS_PER_VOXEL, -1)
        concat = torch.cat([pointwise_input, repeat], dim=2)
        mask = mask.expand(-1, -1, 2 * self.local_agg_features) 
        concat = concat * mask.float()
        return concat 


class FeatureLearningNet(nn.Module):
    def __init__(self):
        super(FeatureLearningNet, self).__init__()
        self.vfe_1 = VFELayer(7, 32)
        self.vfe_2 = VFELayer(32, 128)
    

    def forward(self, feature: list, coordinate: list):
        bs = len(feature)
        feature = torch.cat(feature, dim=0)
        coordinate = torch.cat(coordinate, dim=0)
        vmax, _ = torch.max(feature, dim=2, keepdim=True)
        mask = (vmax != 0) 
        x = self.vfe_1(feature, mask)
        x = self.vfe_2(x, mask)

        voxelwise, _ = torch.max(x, dim=1)
        # use pytorch sparse tensor for efficient memory usage 
        outs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size(
            [bs, cfg.OBJECT.DEPTH, cfg.OBJECT.HEIGHT, cfg.OBJECT.WIDTH, 128]
        ))

        outs = outs.to_dense()

        return outs



class ConvMD(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        cin: int, 
        cout: int, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        bn: bool = True, 
        activation: bool = True,
    ):
        super(ConvMD, self).__init__()
        self.input_dim = input_dim
        self.cin = cin 
        self.cout = cout 
        self.kernel_size = kernel_size 
        self.stride = stride  
        self.padding = padding 
        self.bn = bn 
        self.activation = activation 

        if self.input_dim == 2:
            self.conv = nn.Conv2d(
                self.cin, 
                self.cout, 
                self.kernel_size, 
                self.stride, 
                self.padding,
            )
            if self.bn: 
                self.batch_norm = nn.BatchNorm2d(self.cout)
        
        elif self.input_dim == 3:
            self.conv = nn.Conv3d(
                self.cin,
                self.cout,
                self.kernel_size,
                self.stride,
                self.padding,
            )
            if self.bn: 
                self.batch_norm = nn.BatchNorm3d(self.cout)

        else: 
            raise ValueError("Choose between 2D and 3D input.")


    def forward(self, x):
        x = self.conv(x) 

        if self.bn: 
            x = self.batch_norm(x) 

        if self.activation:
            x = F.relu(x)

        return x


class DeConv2d(nn.Module):
    def __init__(
        self, 
        cin: int, 
        cout: int, 
        kernel_size: int, 
        stride: int, 
        padding: int, 
        bn: bool = True,
    ):
        super(DeConv2d, self).__init__()
        self.cin = cin 
        self.cout = cout 
        self.kernel_size = kernel_size
        self.stride = stride 
        self.padding = padding 
        self.bn = bn 
        
        self.deconv = nn.ConvTranspose2d(
            self.cin, self.cout, self.kernel_size, 
            self.stride, self.padding)

        if self.bn: 
            self.batch_norm = nn.BatchNorm2d(self.cout)


    def forward(self, x):
        x = self.deconv(x)
        if self.bn: 
            x = self.batch_norm(x)
        return F.relu(x) 


class MiddleConvNet(nn.Module):
    def __init__(self):
        super(MiddleConvNet, self).__init__()
        
        self.middle_layer = nn.Sequential(
            ConvMD(3, 128, 64, 3, (2, 1, 1,), (1, 1, 1)),
            ConvMD(3, 64, 64, 3, (1, 1, 1,), (0, 1, 1)),
            ConvMD(3, 64, 64, 3, (2, 1, 1,), (1, 1, 1)),
        )

        if cfg.OBJECT.NAME == 'Car':
            self.block1 = nn.Sequential(
                ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            )
        else: 
            self.block1 = nn.Sequential(
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
                ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            )

        self.deconv1 = DeConv2d(128, 256, 3, (1, 1), (1, 1))

        self.block2 = nn.Sequential(
            ConvMD(2, 128, 128, 3, (2, 2), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
            ConvMD(2, 128, 128, 3, (1, 1), (1, 1)),
        )

        self.deconv2 = DeConv2d(128, 256, 2, (2, 2), (0, 0))

        self.block3 = nn.Sequential(
            ConvMD(2, 128, 256, 3, (2, 2), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
            ConvMD(2, 256, 256, 3, (1, 1), (1, 1)),
        )

        self.deconv3 = DeConv2d(256, 256, 4, (4, 4), (0, 0))

        self.prob_conv = ConvMD(2, 768, 2, 1, (1, 1), (0, 0), bn=False, activation=False)
        self.reg_conv = ConvMD(2, 768, 14, 1, (1, 1), (0, 0), bn=False, activation=False)
        self.output_shape = [cfg.OBJECT.FEATURE_HEIGHT, cfg.OBJECT.FEATURE_WIDTH]


    def forward(self, x):
        batch_size, _, height, width, _ = x.shape 
        x = x.permute(0, 4, 1, 2, 3) # (B, D, H, W, C) -> (B, C, D, H, W)
        
        x = self.middle_layer(x)
        #print(f"x shape: {x.size()}")
        #x = x.view(batch_size, -1, height, width)
        x = x.reshape((batch_size, -1, height, width)) 
        x = self.block1(x) 
        tmp_deconv_1 = self.deconv1(x)

        x = self.block2(x)
        tmp_deconv_2 = self.deconv2(x)

        x = self.block3(x)
        tmp_deconv_3 = self.deconv3(x)
        x = torch.cat(
            [tmp_deconv_3, tmp_deconv_2, tmp_deconv_1], dim=1,
        )

        # probability score map (batch, 2, 200/100, 176/120)
        probs_map = self.prob_conv(x) 

        # regression map (batch, 14, 200/100, 176/120) 
        reg_map = self.reg_conv(x)

        return torch.sigmoid(probs_map), reg_map 


class RPN3D(nn.Module):
    def __init__(self, cls_name: str = 'Car', alpha=1.5, beta=1, sigma=3):
        super(RPN3D, self).__init__()
        self.cls_name = cls_name
        self.alpha = alpha 
        self.beta = beta 
        self.sigma = sigma 

        self.feature_net = FeatureLearningNet()
        self.middle_rpn = MiddleConvNet()
        #self.reg_loss_fn = torch.nn.SmoothL1Loss(reduction='sum')

        self.anchors = generate_anchors()
        self.rpn_output_shape = self.middle_rpn.output_shape


    def forward(self, x, device):
        label = x[1]
        voxel_features = x[2]
        #voxel_numbers = x[3]
        voxel_coordinates = x[4]
        voxel_features = [f.to(device) for f in voxel_features]
        voxel_coordinates = [c.to(device) for c in voxel_coordinates]

        features = self.feature_net(voxel_features, voxel_coordinates)
        prob_out, delta_out = self.middle_rpn(features)

        # calculate the ground truth
        pos_equal_one, neg_equal_one, targets = generate_targets(label, self.rpn_output_shape, self.anchors) 
        pos_equal_one_for_reg = np.concatenate(
            [np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1
        )

        pos_equal_one_sum = np.clip(
            np.sum(
                pos_equal_one, axis=(1, 2, 3)
            ).reshape(-1, 1, 1, 1), a_min=1, a_max=None,
        )
        neg_equal_one_sum = np.clip(
            np.sum(
                neg_equal_one, axis=(1, 2, 3)
            ).reshape(-1, 1, 1, 1), a_min=1, a_max=None,
        )

        # move everything to gpu   
        device = features.device 
        pos_equal_one = torch.from_numpy(pos_equal_one).to(device).float()
        neg_equal_one = torch.from_numpy(neg_equal_one).to(device).float() 
        targets = torch.from_numpy(targets).to(device).float() 
        pos_equal_one_for_reg = torch.from_numpy(pos_equal_one_for_reg).to(device).float()
        pos_equal_one_sum = torch.from_numpy(pos_equal_one_sum).to(device).float()
        neg_equal_one_sum = torch.from_numpy(neg_equal_one_sum).to(device).float()

        # reshape 
        pos_equal_one = pos_equal_one.permute(0, 3, 1, 2)
        neg_equal_one = neg_equal_one.permute(0, 3, 1, 2)
        targets = targets.permute(0, 3, 1, 2) 
        pos_equal_one_for_reg = pos_equal_one_for_reg.permute(0, 3, 1, 2) 
        
        # calc loss 
        cls_pos_loss = (-pos_equal_one * torch.log(prob_out + 1e-6)) / pos_equal_one_sum
        cls_neg_loss = (-neg_equal_one * torch.log(1 - prob_out + 1e-6)) / neg_equal_one_sum
        cls_loss = torch.sum(self.alpha * cls_pos_loss + self.beta * cls_neg_loss)

        cls_pos_loss_rec = torch.sum(cls_pos_loss)
        cls_neg_loss_rec = torch.sum(cls_neg_loss)

        #reg_loss = self.reg_loss_fn(delta_out * pos_equal_one_for_reg, targets * pos_equal_one_for_reg) 
        reg_loss = smooth_L1_loss(delta_out * pos_equal_one_for_reg, targets * pos_equal_one_for_reg, self.sigma) / \
            pos_equal_one_sum
        reg_loss = torch.sum(reg_loss)
        
        loss = cls_loss + reg_loss

        return (
            prob_out, 
            delta_out, 
            loss, 
            cls_loss, 
            reg_loss, 
            cls_pos_loss_rec, 
            cls_neg_loss_rec,
        )


    def predict(self, data, probs, deltas, summary=False, visual=False):

        device = probs.device   
        tag = data[0]      
        label = data[1]
        
        rgb = data[5]
        raw_lidar = data[6]

        batch_size = probs.shape[0]
        batch_gt_boxes_3d = None 

        if summary or visual: 
            batch_gt_boxes_3d = label_to_gt_box_3d(label, cls_name='Car', coordinate='lidar')

        probs = probs.cpu().detach().numpy()
        deltas = deltas.cpu().detach().numpy()

        batch_boxes_3d = deltas_to_boxes_3d(deltas, self.anchors) 
        batch_boxes_2d = batch_boxes_3d[:, :, [0, 1, 4, 5, 6]] 
        batch_probs = probs.reshape((batch_size, -1))

        # NMS 
        ret_box_3d = []
        ret_score = []

        for batch_id in range(batch_size):
            # remove boxes with low scores 
            idx = np.where(batch_probs[batch_id, :] >= cfg.RPN.SCORE_THRES)[0]
            tmp_boxes_3d = batch_boxes_3d[batch_id, idx, ...]
            tmp_boxes_2d = batch_boxes_2d[batch_id, idx, ...]
            tmp_scores = batch_probs[batch_id, idx]

            boxes_2d = corner_to_standup_box2d(
                center_to_corner_box_2d(tmp_boxes_2d, coordinate='lidar')
            )

            idx, cnt = nms(
                torch.from_numpy(boxes_2d).to(device), 
                torch.from_numpy(tmp_scores).to(device), 
                cfg.RPN.NMS_THRES, 
                cfg.RPN.NMS_POST_TOPK,
            )  

            idx = idx[:cnt].cpu().detach().numpy()

            tmp_boxes_3d = tmp_boxes_3d[idx, ...]
            tmp_scores = tmp_scores[idx]
            ret_box_3d.append(tmp_boxes_3d)
            ret_score.append(tmp_scores)

        ret_box_3d_score = []
        for boxes_3d, scores in zip(ret_box_3d, ret_score):
            ret_box_3d_score.append(np.concatenate(
                [np.tile(self.cls_name, len(boxes_3d))[:, np.newaxis], 
                boxes_3d, scores[:, np.newaxis]], axis=-1
            ))
        
        if summary:
            P, Tr, R = load_calib(os.path.join(cfg.DATA.CALIB_DIR, tag[0] + '.txt'))

            front_image = draw_lidar_box_3d_on_image(
                rgb[0], ret_box_3d[0],  batch_gt_boxes_3d[0], 
                P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)

            birdview = lidar_to_bird_view_image(raw_lidar[0], factor=1) 
            birdview = draw_lidar_box_3d_on_birdview(
                birdview, ret_box_3d[0], batch_gt_boxes_3d[0], factor=1, 
                P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
            
            heatmap = colorize(probs[0, ...], 1)

            ret_summary = [
                ['predict/front_view_rgb', front_image[np.newaxis, ...]],
                ['predict/bird_view-lidar', birdview[np.newaxis, ...]],
                ['predict/bird_view_heatmap', heatmap[np.newaxis, ...]],
            ] 

            return tag, ret_box_3d_score, ret_summary 
        
        if visual:
            front_images, bird_views, heatmaps = [], [], []
            for i in range(len(rgb)):
                cur_tag = tag[i]
                P, Tr, R = load_calib(os.path.join(cfg.DATA.CALIB_DIR, cur_tag + '.txt'))
                
                front_image = draw_lidar_box_3d_on_image(
                    rgb[i], ret_box_3d[i],  batch_gt_boxes_3d[i], 
                    P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)

                birdview = lidar_to_bird_view_image(raw_lidar[i], factor=1) 
                birdview = draw_lidar_box_3d_on_birdview(
                    birdview, ret_box_3d[i], batch_gt_boxes_3d[i], factor=1, 
                    P2=P, T_VELO_2_CAM=Tr, R_RECT_0=R)
                
                heatmap = colorize(probs[i, ...], 1)

                front_images.append(front_image)
                bird_views.append(birdview)
                heatmaps.append(heatmap)

            return tag, ret_box_3d_score, front_images, bird_views, heatmaps

        return tag, ret_box_3d_score 





def test():
    from dataset import KITTIDataset
    import numpy as np  
    from dataset import collate_fn
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device) 
    dataset = KITTIDataset(cfg.DATA.DIR, False)
    print(len(dataset))
    model = RPN3D().cuda()
    print(model) 
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=True, collate_fn=collate_fn, pin_memory=False)

    for data in dataloader:
        print(type(data)) 
        label, voxel_features, voxel_numbers, voxel_coordinates, rgb, raw_lidar = data 
        voxel_features = [f.to(device) for f in voxel_features]
        voxel_coordinates = [c.to(device) for c in voxel_coordinates] 
        print(type(label)) 
        print(len(voxel_coordinates))
        print(len(voxel_features)) 
        
        (prob_out, delta_out, loss, cls_loss, reg_loss, _, _) = model(data, device) 
        print(f"Prob out: {prob_out.shape}") 
        print(f"Delta out: {delta_out.shape}")
        print(f"Loss: {loss}")
        print(f"Cls loss: {cls_loss}")
        print(f"Reg loss: {reg_loss}")
        break




if __name__ == "__main__":
    test()