import torch 
import torch.nn as nn 

from config import get_cfg_defaults

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
    

    def forward(self, feature, coordinate):
        bs = len(feature)
        feature = torch.cat(feature)
        print(feature.size())
        coordinate = torch.cat(coordinate)
        print(coordinate.size())
        vmax, _ = torch.max(feature, dim=2, keepdim=True)
        print(vmax.size()) 
        mask = (vmax != 0) 
        print(mask.shape)
        x = self.vfe_1(feature, mask)
        print(x.size())

        x = self.vfe_2(x, mask)
        print(x.size())

        voxelwise, _ = torch.max(x, dim=1)
        print(voxelwise.size())
        outs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size(
            [bs, cfg.OBJECT.DEPTH, cfg.OBJECT.HEIGHT, cfg.OBJECT.WIDTH, 128]
        ))

        outs = outs.to_dense()

        return outs


def test():
    from dataset import KITTIDataset
     
    dataset = KITTIDataset(cfg.DATA.DIR, False)
    print(len(dataset)) 

if __name__ == "__main__":
    test()