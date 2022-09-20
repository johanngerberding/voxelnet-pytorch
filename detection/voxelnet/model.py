import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import numpy as np 

from config import get_cfg_defaults
from utils import generate_anchors, generate_targets 


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
    def __init__(
        self, 
        alpha: float = 1.5, 
        beta: int = 1, 
        sigma: int = 3, 
        training: bool = True, 
        name: str = '',
    ):
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
        self.middle_rpn = MiddleConvNet(alpha, beta, sigma)

        self.anchors = generate_anchors(cfg)
        self.rpn_output_shape = self.middle_rpn.output_shape


    def forward(self, x, device):

        label = x[0]
        voxel_features = x[1]
        voxel_numbers = x[2]
        voxel_coordinates = x[3]
        voxel_features = [f.to(device) for f in voxel_features]
        voxel_coordinates = [c.to(device) for c in voxel_coordinates]

        features = self.feature_net(voxel_features, voxel_coordinates)
        print(features.size()) 
        prob_out, delta_out = self.middle_rpn(features)
        print(prob_out.size())
        print(delta_out.size())

        # calculate the ground truth
        pos_equal_one, neg_equal_one, targets = generate_targets(label, self.rpn_output_shape, self.anchors, cfg) 
        print(f"pos equal one shape: {pos_equal_one.shape}")
        print(f"neg equal one shape: {neg_equal_one.shape}") 
        print(f"targets shape: {targets.shape}")
        pos_equal_one_for_reg = np.concatenate(
            [np.tile(pos_equal_one[..., [0]], 7), np.tile(pos_equal_one[..., [1]], 7)], axis=-1
        )
        print(f"pos equal one for reg shape: {pos_equal_one_for_reg.shape}") 

        pos_equal_one_sum = np.clip(
            np.sum(
                pos_equal_one, axis=(1, 2, 3)
            ).reshape(-1, 1, 1, 1), a_min=1, a_max=None,
        )
        print(f"pos equal one sum shape: {pos_equal_one_sum.shape}")
        print(pos_equal_one_sum[:, :, :, :]) 
        neg_equal_one_sum = np.clip(
            np.sum(
                neg_equal_one, axis=(1, 2, 3)
            ).reshape(-1, 1, 1, 1), a_min=1, a_max=None,
        )
        print(f"neg equal one sum shape: {neg_equal_one_sum.shape}")

        # move everything to gpu   
        device = features.device 
        print(device)
        pos_equal_one = torch.from_numpy(pos_equal_one).to(device).float()
        neg_equal_one = torch.from_numpy(neg_equal_one).to(device).float() 
        
        
        # calc loss 



        return None 


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
        
        out = model(data, device) 
        break




if __name__ == "__main__":
    test()