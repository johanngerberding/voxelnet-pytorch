import torch 
import torch.nn as nn 

voxel_pcl = 35
input_depth = int((1 - -3 ) / 0.4) 
input_height = int((40 - -40) / 0.2) 
input_width = int((70.4 - 0) / 0.2) 


class VFELayer(nn.Module):
    def __init__(self, cin: int, cout: int):
        super(VFELayer, self).__init__()

        self.in_channels = cin 
        self.out_channels = cout
        self.local_agg_features = cout / 2 

        self.fcn = nn.Sequential(
            nn.Linear(self.in_channels, self.local_agg_features),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm1d(self.local_agg_features)


    def forward(self, inputs, mask):
        temp = self.fcn(inputs).transpose(1,2)
        pointwise_input = self.bn(temp).transpose(1,2) 
        agg, _ = torch.max(pointwise_input, dim=1, keepdim=True)
        repeat = agg.expand(-1, voxel_pcl, -1)
        concat = torch.cat([pointwise_input, repeat], dim=2)
        mask = mask.expand(-1, -1, 2 * self.local_agg_features) 
        concat = concat * mask.float()
        return concat 


class FeatureLearningNet(nn.Module):
    def __init__(self):
        super(FeatureLearningNet, self).__init__()
        self.vfe_1 = VFELayer(7, 32)
        self.vfe_2 = VFELayer(32, 128)
    

    def forward(self, feature, number, coordinate):
        bs = len(feature)
        feature = torch.cat(feature, dim=0)
        print(feature.size())
        coordinate = torch.cat(coordinate, dim=0)
        print(coordinate.size())
        vmax, _ = torch.max(feature, dim=2, keepdim=True)
        mask = (vmax != 0) 
        print(mask.shape)
        x = self.vfe_1(feature, mask)
        print(x.size())

        x = self.vfe_2(x, mask)
        print(x.size())

        voxelwise, _ = torch.max(x, dim=1)
        outs = torch.sparse.FloatTensor(coordinate.t(), voxelwise, torch.Size(
            [bs, input_depth, input_height, input_width, 128]
        ))

        outs = outs.to_dense()

        return outs


def test():
    pcl_path = ""


if __name__ == "__main__":
    test()
