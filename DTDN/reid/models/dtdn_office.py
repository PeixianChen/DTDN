<<<<<<< HEAD
from __future__ import absolute_import

import torch
from torch import cat
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
from . import resnet
from . import mobilenet




__all__ = ['resnet50']

class DynamicNet(nn.Module):

    def __init__(self, channels=512, reduction=16):
        super(DynamicNet, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        # self.a = nn.Parameter(torch.tensor(-5.0))
        # self.b = nn.Parameter(torch.tensor(-0.1))
        self.shift = lambda x: -5 * (x - 0.1)
        #self.shift = lambda x:self.a * x + self.b
        self.sigmoid = nn.Sigmoid()
        # self.init_params()

    def forward(self, x):
        # module_input = x
        # x = self.avg_pool(x)
        x = self.max_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.shift(x)
        # x = self.a * x + self.b
        x = self.sigmoid(x)
        # if x.device.index ==0:
        #     print (x.mean().item(), x.sum().item())
        return x
    def init_params(self):
        print("...DynamicNet init_param...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

class Encoder(nn.Module):
    __factory = {
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: torchvision.models.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, target_num=0, cut_layer=None):
        super(Encoder, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.cut_layer = cut_layer

        # Construct base (pretrained) resnet
        if depth not in Encoder.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = Encoder.__factory[depth](pretrained=pretrained)
    
        if not self.pretrained:
            self.init_params()

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == self.cut_layer:
                break
            x = module(x)
            #print(name)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True))

    def forward(self, x):
        return self.main(x)

class TaskNet(nn.Module):
    __factory = {
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: torchvision.models.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, target_num=0, cut_layer=None):
        super(TaskNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.cut_layer = cut_layer
        self.target_num = target_num

        # Construct base (pretrained) resnet
        if depth not in TaskNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = TaskNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)
            if self.target_num > 0:
                self.target_classifier = nn.Linear(self.num_features, self.target_num)
                init.normal_(self.target_classifier.weight, std=0.001)
                init.constant_(self.target_classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None, tgt_output_feature=None, domain='source'):
        Flag = False
        # Flag = True
        for name, module in self.base._modules.items():
            if name == self.cut_layer:
                Flag = True
            if name == 'avgpool':
                break
            if Flag:
                x = module(x)

        if self.cut_at_pooling:
            return x

        if output_feature == 'pool5':
            x = F.max_pool2d(x, x.size()[2:])
            x = x.view(x.size(0), -1)
            x = F.normalize(x)
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if tgt_output_feature == 'pool5':
            tgt_feat = x
            tgt_feat = F.normalize(tgt_feat, p=2)
            tgt_feat = self.drop(tgt_feat)
            return x, tgt_feat
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        # triplet feature
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x_class = self.classifier(x)
            # x_class = nn.Softmax(dim=-1)(x_class)
        if domain == 'target':
            x_class = self.target_classifier(x)
        # two outputs
        return x_class, x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
def resnet50(**kwargs):
    return Encoder(50, **kwargs), TaskNet(50, **kwargs), DynamicNet()
=======
from __future__ import absolute_import

import torch
from torch import cat
from torch import nn
from torch.nn import functional as F
from torch.nn import init
import torchvision
import pdb
from . import resnet
from . import mobilenet




__all__ = ['resnet50']

class DynamicNet(nn.Module):

    def __init__(self, channels=512, reduction=16):
        super(DynamicNet, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.shift = lambda x: -5 * (x - 0.1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.max_pool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.shift(x)
        # x = self.a * x + self.b
        x = self.sigmoid(x)
        return x
    def init_params(self):
        print("...DynamicNet init_param...")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

class Encoder(nn.Module):
    __factory = {
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: torchvision.models.resnet50,
        # 50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, target_num=0, cut_layer=None):
        super(Encoder, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.cut_layer = cut_layer

        # Construct base (pretrained) resnet
        if depth not in Encoder.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = Encoder.__factory[depth](pretrained=pretrained)
        # self.base_b = ResNet.__factory[depth](pretrained=pretrained)

    
        if not self.pretrained:
            self.init_params()

    def forward(self, x, output_feature=None):
        for name, module in self.base._modules.items():
            if name == self.cut_layer:
                break
            x = module(x)
            #print(name)
        return x

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

class ResidualBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ResidualBlock, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim_out, dim_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dim_out, affine=True))

    def forward(self, x):
        return self.main(x)

class TaskNet(nn.Module):
    __factory = {
        18: resnet.resnet18,
        34: resnet.resnet34,
        50: torchvision.models.resnet50,
        # 50: resnet.resnet50,
        101: resnet.resnet101,
        152: resnet.resnet152,
    }

    def __init__(self, depth, pretrained=True, cut_at_pooling=False,
                 num_features=0, norm=False, dropout=0, num_classes=0, target_num=0, cut_layer=None):
        super(TaskNet, self).__init__()

        self.depth = depth
        self.pretrained = pretrained
        self.cut_at_pooling = cut_at_pooling
        self.cut_layer = cut_layer
        self.target_num = target_num

        # Construct base (pretrained) resnet
        if depth not in TaskNet.__factory:
            raise KeyError("Unsupported depth:", depth)
        self.base = TaskNet.__factory[depth](pretrained=pretrained)

        if not self.cut_at_pooling:
            self.num_features = num_features
            self.norm = norm
            self.dropout = dropout
            self.has_embedding = num_features > 0
            self.num_classes = num_classes
            out_planes = self.base.fc.in_features

            # Append new layers
            if self.has_embedding:
                self.feat = nn.Linear(out_planes, self.num_features)
                self.feat_bn = nn.BatchNorm1d(self.num_features)
                init.kaiming_normal_(self.feat.weight, mode='fan_out')
                init.constant_(self.feat.bias, 0)
                init.constant_(self.feat_bn.weight, 1)
                init.constant_(self.feat_bn.bias, 0)
            else:
                # Change the num_features to CNN output channels
                self.num_features = out_planes
            if self.dropout > 0:
                self.drop = nn.Dropout(self.dropout)
            if self.num_classes > 0:
                self.classifier = nn.Linear(self.num_features, self.num_classes)
                init.normal_(self.classifier.weight, std=0.001)
                init.constant_(self.classifier.bias, 0)
            if self.target_num > 0:
                self.target_classifier = nn.Linear(self.num_features, self.target_num)
                init.normal_(self.target_classifier.weight, std=0.001)
                init.constant_(self.target_classifier.bias, 0)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x, output_feature=None, tgt_output_feature=None, domain='source'):
        Flag = False
        # Flag = True
        for name, module in self.base._modules.items():
            if name == self.cut_layer:
                Flag = True
            if name == 'avgpool':
                break
            if Flag:
                #print(name)
                x = module(x)

        if self.cut_at_pooling:
            return x

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(x.size(0), -1)

        if tgt_output_feature == 'pool5':
            tgt_feat = x
            tgt_feat = F.normalize(tgt_feat, p=2)
            tgt_feat = self.drop(tgt_feat)
            return x, tgt_feat
        if self.norm:
            x = F.normalize(x)
        elif self.has_embedding:
            x = F.relu(x)
        # triplet feature
        if self.dropout > 0:
            x = self.drop(x)
        if self.num_classes > 0:
            x_class = self.classifier(x)
            # x_class = nn.Softmax(dim=-1)(x_class)
        if domain == 'target':
            x_class = self.target_classifier(x)
        # two outputs
        return x_class, x

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)
def resnet50(**kwargs):
    return Encoder(50, **kwargs), TaskNet(50, **kwargs), DynamicNet()
>>>>>>> f0906cafd587b9f863e29ed0904c7c6f81d0db32
