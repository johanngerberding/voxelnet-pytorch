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
    

    def forward(self, feature: list, coordinate: list):
        bs = len(feature)
        print(f"Batch size: {bs}")
        feature = torch.cat(feature, dim=0)
        print(f"Feature size: {feature.size()}")
        coordinate = torch.cat(coordinate, dim=0)
        print(f"Coordinate shape: {coordinate.size()}")
        vmax, _ = torch.max(feature, dim=2, keepdim=True)
        print(f"Vmax shape: {vmax.size()}") 
        mask = (vmax != 0) 
        print(f"Mask shape: {mask.shape}")
        x = self.vfe_1(feature, mask)
        print(f"Shape after first layer: {x.size()}")

        x = self.vfe_2(x, mask)
        print(f"Shape after second layer: {x.size()}")

        voxelwise, _ = torch.max(x, dim=1)
        print(f"Voxelwise shape: {voxelwise.size()}")
        outs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size(
            [bs, cfg.OBJECT.DEPTH, cfg.OBJECT.HEIGHT, cfg.OBJECT.WIDTH, 128]
        ))

        outs = outs.to_dense()

        return outs


def test():
    from dataset import KITTIDataset
    import numpy as np  
    dataset = KITTIDataset(cfg.DATA.DIR, False)
    print(len(dataset))

    feature_learning_net = FeatureLearningNet()

    #TODO iterate through dataset and prepare input for model

    for img, pcl, labels, voxels in dataset:
        feature = voxels['feature_buffer']   
        print(feature.shape) 
        coordinate = voxels['coordinate_buffer'] 
        print(coordinate.shape)
        print(coordinate[0, :]) 
        feature = [torch.tensor(feature)]
        coordinate = [
            torch.from_numpy(
                np.pad(coordinate, ((0,0), (1,0)), mode='constant', constant_values=0)
            )
        ]
        print(f"coordinate shape after padding: {coordinate[0].size()}") 
        print(coordinate[0][0, :]) 
        out = feature_learning_net(feature, coordinate) 
        print(out.size()) 
        break  


if __name__ == "__main__":
    test()