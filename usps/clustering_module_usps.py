import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter


class auto_encoder(nn.Module):
    def __init__(self):
        super(auto_encoder, self).__init__()
        self.encoder1 = nn.Linear(256, 500)
        self.encoder2 = nn.Linear(500, 500)
        self.encoder3 = nn.Linear(500, 1000)
        self.encoder4 = nn.Linear(1000, 10)
        self.decoder1 = nn.Linear(10, 1000)
        self.decoder2 = nn.Linear(1000, 500)
        self.decoder3 = nn.Linear(500, 500)
        self.decoder4 = nn.Linear(500, 256)

    def forward(self, x):
        h = F.relu(self.encoder1(x))
        h = F.relu(self.encoder2(h))
        h = F.relu(self.encoder3(h))
        latent = self.encoder4(h)
        r = F.relu(self.decoder1(latent))
        r = F.relu(self.decoder2(r))
        r = F.relu(self.decoder3(r))
        r = self.decoder4(r)
        return latent


class clustering_model(nn.Module):
    def __init__(self, fuzzifier, class_number, device):
        super(clustering_model, self).__init__()
        self.feat_extractor = auto_encoder()
        self.clustering_layer = Parameter(torch.Tensor(10, class_number), requires_grad=True)
        self.fuzzifier = fuzzifier
        self.device = device
        self.class_number = class_number

    def load_init_weight(self, pretrain_path, gpu):
        state_dict = torch.load(pretrain_path, map_location='cuda:{}'.format(gpu))
        self.feat_extractor.load_state_dict(state_dict)

    def forward(self, x):
        latent = self.feat_extractor(x)
        dis = torch.sqrt(torch.sum(torch.square(torch.unsqueeze(latent, dim=1) - self.clustering_layer), dim=2)).t()
        dis = dis ** (-2. / (self.fuzzifier - 1.))
        c = torch.unsqueeze(torch.sum(dis, dim=0), dim=0)
        u_matrix = (dis / torch.mul(torch.ones(size=(self.class_number, 1)).to(self.device), c)).t()
        return u_matrix