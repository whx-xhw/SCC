import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models.resnet import resnet18
from thop import profile, clever_format


class model_simple(nn.Module):
    def __init__(self):
        super(model_simple, self).__init__()
        self.f = []
        temp_model = resnet18().named_children()
        embedding_size = 512
        for name, module in temp_model:
            if name == 'conv1':
                module = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            if not isinstance(module, nn.Linear):
                self.f.append(module)
        self.f = nn.Sequential(*self.f)
        self.g = nn.Sequential(nn.Linear(embedding_size, 512, bias=False), nn.BatchNorm1d(512),
                               nn.ReLU(inplace=True), nn.Linear(512, 1024, bias=True))

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        return F.normalize(feature, dim=-1)


def get_res_simple(self_supervised_pretrain_path):
    model = model_simple()
    flops, params = profile(model, inputs=(torch.randn(1, 3, 64, 64),))
    flops, params = clever_format([flops, params])
    model.load_state_dict(torch.load(self_supervised_pretrain_path))
    return model


class clustering_model(nn.Module):
    def __init__(self, fuzzifier, class_number, device):
        super(clustering_model, self).__init__()
        self.feat_extractor = get_res_simple(self_supervised_pretrain_path='./weight/init_weight.pth')
        self.clustering_layer = Parameter(torch.Tensor(512, class_number), requires_grad=True)
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
