import torch.nn as nn
import torch.nn.functional as F
import torch

from torchvision import datasets
from torchvision import transforms
from radialbnn.radial_layer import RadialLayer
from radialbnn.utils.elbo_approximator import elbo_approximator



def get_net(name, lossType=None, modelType = None,in_dim = None, out_dim = None):
    if name == 'MNIST':
        if modelType=='bnn':
            return RadialBayesianNetwork(in_dim,out_dim)
        else:
            return Net1
    elif name == 'FashionMNIST':
        if modelType=='bnn':
            return RadialBayesianNetwork(in_dim,out_dim)
        else:
            return Net1
    elif name == 'SVHN':
        if modelType=='bnn':
            return RadialBayesianNetwork(in_dim,out_dim)
        else:
            return Net2
    elif name == 'CIFAR10':
        if modelType=='bnn':
            return RadialBayesianNetwork(in_dim,out_dim)
        if modelType=='res':
            return resnet18()
        else:
            return Net3

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.7)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training,p=0.7)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(1152, 400)
        self.fc2 = nn.Linear(400, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3_drop(self.conv3(x)), 2))
        x = x.view(-1, 1152)
        x = F.relu(self.fc1(x))
        e1 = F.relu(self.fc2(x))
        x = F.dropout(e1, training=self.training)
        x = self.fc3(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5)
        self.fc1 = nn.Linear(1024, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = x.view(-1, 1024)
        e1 = F.relu(self.fc1(x))
        x = F.dropout(e1, training=self.training,p=0.7)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50

class NetDeep1(nn.Module):#MNIST dataset
    def __init__(self,numClass=10):
        super(NetDeep1, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 640)
        self.fc2 = nn.Linear(640, 120)
        self.numClass=numClass
        self.svmHeads=[]
        for k in range(self.numClass):
            self.svmHeads.append(nn.Linear(120,1).cuda())
        #self.fc2 = nn.Linear(50, 10)
        #create head layers.

    # def forward(self, x):
    #     x = F.relu(F.max_pool2d(self.conv1(x), 2))
    #     x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    #     x = x.view(-1, 320)
    #     self.temX=x
    #     x = F.relu(self.fc1(x))
    #     #x = F.dropout(e1, training=self.training,p=0.5)
    #     x = F.relu(self.fc2(x))
    #     # x = torch.nn.Sigmoid(self.fc2(x))
    #     x = torch.cat((x,torch.ones(x.size(dim=0),1).cuda()),dim=1)
    #     self.svmHeads=[]
    #     for k in range(self.numClass):
    #         self.svmHeads.append(nn.Linear(121,1).cuda())#it is not clear why this cannot be assigned to device with the outter call net().to(self.device)
    #     res=[]
    #     w=[]
    #     for k in range(self.numClass):
    #         res.append(self.svmHeads[k](x))
    #         w.append(self.svmHeads[k].weight.data)
    #     self.temRes=res
    #     self.temW=w
        
    #     return x, res, w #h,a,w
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        self.temX=x
        x = F.relu(self.fc1(x))
        #x = F.dropout(e1, training=self.training,p=0.5)
        x = F.tanh(self.fc2(x))
        # x = F.dropout(x, training=self.training,p=0.7)

        # x = torch.nn.Sigmoid(self.fc2(x))
        # x = torch.cat((x,torch.ones(x.size(dim=0),1).cuda()),dim=1)
        # self.svmHeads=[]
        # for k in range(self.numClass):
        #     self.svmHeads.append(nn.Linear(120,1).cuda())#it is not clear why this cannot be assigned to device with the outter call net().to(self.device)
        res=[]
        w=[]
        for k in range(self.numClass):
            res.append(self.svmHeads[k](x))
            w.append(self.svmHeads[k].weight.data)
        self.temRes=res
        self.temW=w
        
        return x, res, w #h,a,w

    def get_embedding_dim(self):
        return 50
    
    
class Cifar10CnnModel(nn.Module):
    def __init__(self,numClass=10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 16 x 16

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 8 x 8

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 256 x 4 x 4

            nn.Flatten(), 
            nn.Linear(256*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10))
        
    def forward(self, xb):
        return self.network(xb)    
    
class NetDeepSVM(nn.Module):#MNIST dataset
    def __init__(self,NClass = 5):
        super(NetDeepSVM, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 1000)
        self.svmHeads=[]
        self.NClass = NClass
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")
        for k in range(NClass):
            self.svmHeads.append(nn.Linear(1000,1).to(self.device))
        #self.fc2 = nn.Linear(50, 10)
        #create head layers.

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        
        x = F.relu(self.fc1(x))
        self.temX0=x
        x = F.dropout(x, training=self.training,p=0.5)
        x = F.relu(self.fc2(x))
        self.temX=x
        #x = self.fc3(x)
        
#it is not clear why this cannot be assigned to device with the outter call net().to(self.device)
        res=[]
        w=[]
        for k in range(self.NClass):
            res.append(self.svmHeads[k](x))
            #res.append(self.fc3(x))

            w.append(self.svmHeads[k].weight.data)
        self.temRes=res
        self.temW=w
        #res.append(x)
        return x, res, w #h,a,w    
    def get_embedding_dim(self):
        return 1000
    
    
    
    
    
    
    
    
class NetDeepBin(nn.Module):#MNIST dataset
    def __init__(self):
        super(NetDeepBin, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.3)
        self.fc1 = nn.Linear(320, 100)
        self.fc2 = nn.Linear(100, 30)
        
        #self.fc2 = nn.Linear(50, 10)
        #create head layers.

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        self.temX=x
        x = F.relu(self.fc1(x))
        #x = F.dropout(e1, training=self.training,p=0.5)
        x = F.tanh(self.fc2(x))
        
        self.svmHeads=[]
        for k in range(2):
            self.svmHeads.append(nn.Linear(100,1).to('cuda'))#it is not clear why this cannot be assigned to device with the outter call net().to(self.device)
        res=[]
        w=[]
        for k in range(2):
            res.append(self.svmHeads[k](x))
            w.append(self.svmHeads[k].weight.data)
        self.temRes=res
        self.temW=w
        
        return x, res, w #h,a,w

    def get_embedding_dim(self):
        return 50



@elbo_approximator
class RadialBayesianNetwork(nn.Module):

    def __init__(self, input_dim, output_dim):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rl1 = RadialLayer(input_dim, 1200)
        self.rl2 = RadialLayer(1200, 1200)
        self.rl3 = RadialLayer(1200, output_dim)

    def forward(self, x):

        x = x.view(-1, self.input_dim)

        x = F.relu(self.rl1(x))
        e1 = F.relu(self.rl2(x))
        x = self.rl3(e1)

        return x, e1
    def get_embedding_dim(self):
        return 1200


class Net_ftest(nn.Module):
    def __init__(self):
        super(Net_ftest, self).__init__()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        e1 = x
        x = F.dropout(e1, training=self.training,p=0.7)
        x = self.fc2(x)
        return x, e1

    def get_embedding_dim(self):
        return 50




from collections import OrderedDict

import torch
import torch.nn as nn


__all__ = ["ResNet", "resnet18", "resnet32_grasp", "resnet32_eigendamage"]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layer_channels,
        channels,
        num_classes=10,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.zero_init_residual = zero_init_residual

        self.inplanes = channels[0]
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group

        # Customised for CIFAR-10, needs to be large conv for imagenet
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, self.inplanes, layer_channels[0])
        self.layer2 = self._make_layer(
            block,
            channels[1],
            layer_channels[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            channels[2],
            layer_channels[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )

        if len(layer_channels) == 4:
            self.layer4 = self._make_layer(
                block,
                channels[3],
                layer_channels[3],
                stride=2,
                dilate=replace_stride_with_dilation[2],
            )
            self.fc = nn.Linear(channels[3] * block.expansion, num_classes)
        else:
            # only three layers
            self.fc = nn.Linear(channels[2] * block.expansion, num_classes)
        print(channels[2] * block.expansion)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming_uniform_ in default conv2d
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                # better accuracy and more stable pruning? - 94.7 accuracy
                nn.init.normal_(m.weight, 0, 0.01)
                # better pruning? - 94.4 accuracy
                # nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
            name = "ds_block"
        else:
            name = "n_block"

        layers = OrderedDict()
        layers[name + "0"] = block(
            self.inplanes,
            planes,
            stride,
            downsample,
            self.groups,
            self.base_width,
            previous_dilation,
            norm_layer,
        )
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers["n_block" + str(i)] = block(
                self.inplanes,
                planes,
                groups=self.groups,
                base_width=self.base_width,
                dilation=self.dilation,
                norm_layer=norm_layer,
            )

        return nn.Sequential(layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # x = self.maxpool(x) - Remove for now, back in with ImageNet

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        if hasattr(self, "layer4"):
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        e1 = x
        x = self.fc(e1)

        return x,e1

    def forward(self, x):
        return self._forward_impl(x)

      
    def get_embedding_dim(self):
        return 50  


def _resnet(arch, block, layer_channels, channels, **kwargs):
    model = ResNet(block, layer_channels, channels, **kwargs)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    layer_channels = [64, 128, 256, 512]
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], layer_channels, **kwargs)


def resnet32_grasp(**kwargs):
    # https://github.com/alecwangcq/GraSP/blob/master/models/base/resnet.py#L55
    layer_channels = [32, 64, 128]

    depth = 32
    n = (depth - 2) // 6

    return _resnet("resnet32", BasicBlock, [n] * 3, layer_channels, **kwargs)


def resnet32_eigendamage(**kwargs):
    # https://github.com/alecwangcq/EigenDamage-Pytorch/blob/master/models/resnet.py#L111

    layer_channels = [64, 128, 256]

    depth = 32
    n = (depth - 2) // 6

    return _resnet("resnet32", BasicBlock, [n] * 3, layer_channels, **kwargs)