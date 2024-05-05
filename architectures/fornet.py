"""
Simple helper file for defining network architectures.

Authors:
Edoardo Daniele Cannas - edoardodaniele.cannas@polimi.it
Sara Mandelli - sara.mandelli@polimi.it
Paolo Bestagini - paolo.bestagini@polimi.it
Stefano Tubaro - stefano.tubaro@polimi.it
"""


"""
Libraries import
"""


from collections import OrderedDict
import torch
from torchvision import transforms
from torchvision.models import alexnet, densenet, resnet
import antialiased_cnns
import torch.nn as nn
import torch.nn.functional as F
from antialiased_cnns import BlurPool
import torch.utils.model_zoo as model_zoo
import re


"""
Feature Extractor
"""


class FeatureExtractor(nn.Module):
    """
    Abstract class to be extended when supporting features extraction.
    It also provides standard normalized and parameters
    """

    def features(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def get_trainable_parameters(self):
        return self.parameters()

    @staticmethod
    def get_normalizer():
        return transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


"""
BondiNet architecture
"""


class BondiNet(FeatureExtractor):
    """
    Implementation of the Bondi et al. architecture for camera model identification.
    Reference: https://ieeexplore.ieee.org/abstract/document/7786852
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 2, **kwargs):
        super(BondiNet, self).__init__()

        # --- Extract parameters from the constructor

        first_layer_stride = kwargs.get('first_layer_stride')
        if first_layer_stride is None:
            raise ValueError('The first_layer_stride parameter is mandatory for BondiNet')

        # --- Build the network

        self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4,
                                     stride=first_layer_stride, padding=0)
        self.lrelu1 = torch.nn.LeakyReLU()
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.maxpool1 = torch.nn.MaxPool2d(kernel_size=2, padding=0)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,
                                     stride=1, padding=0)
        self.lrelu2 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=48)
        self.maxpool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5,
                                     stride=1, padding=0)
        self.lrelu3 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.maxpool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                                     stride=1, padding=0)
        self.lrelu4 = torch.nn.LeakyReLU()
        self.dense1 = torch.nn.LazyLinear(out_features=128)
        self.dense2 = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lrelu1(self.conv1(x))
        x = self.maxpool1(self.bn1(x))
        x = self.lrelu2(self.conv2(x))
        x = self.maxpool2(self.bn2(x))
        x = self.lrelu3(self.conv3(x))
        x = self.maxpool3(self.bn3(x))
        x = self.lrelu4(self.conv4(x))
        x = torch.flatten(x, 1)
        return x

    def getConvsOnly(self) -> torch.nn.Module:
        """
        An helper function that returns a Sequential module with only the convolutional layers (or feature extractor)
        :return: torch.nn.Module
        """
        return torch.nn.Sequential(OrderedDict([('conv1', self.conv1),
                                                ('lrelu1', self.lrelu1),
                                                ('bn1', self.bn1),
                                                ('maxpool1', self.maxpool1),
                                                ('conv2', self.conv2),
                                                ('lrelu2', self.lrelu2),
                                                ('bn2', self.bn2),
                                                ('maxpool2', self.maxpool2),
                                                ('conv3', self.conv3),
                                                ('lrelu3', self.lrelu3),
                                                ('bn3', self.bn3),
                                                ('maxpool3', self.maxpool3),
                                                ('conv4', self.conv4),
                                                ('lrelu4', self.lrelu4)]))

    def forward(self, x):
        x = F.leaky_relu(self.dense1(self.features(x)))
        x = self.dropout(x)
        x = self.dense2(x)
        return x


class AABondiNet(FeatureExtractor):
    """
    Anti-aliased version of BondiNet architecture.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 2, **kwargs):
        super(AABondiNet, self).__init__()

        # --- Extract parameters from the constructor

        first_layer_stride = kwargs.get('first_layer_stride')
        if first_layer_stride is None:
            raise ValueError('The first_layer_stride parameter is mandatory for AABondiNet')
        pool_only = kwargs.get('pool_only')
        if pool_only is None:
            raise ValueError('The pool_only parameter is mandatory for AABondiNet')

        # --- Build the network

        # Insert a blurpool layer after the first convolutional layer if the stride != 1 and we are blurring everything
        if first_layer_stride == 1 or pool_only:
            self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=4,
                                         stride=first_layer_stride, padding=0)
            self.lrelu1 = torch.nn.LeakyReLU()
        else:
            self.conv1 = torch.nn.Conv2d(in_channels=in_channels, out_channels=32,
                                             kernel_size=4, stride=1, padding=0)
            self.lrelu1 = torch.nn.Sequential(torch.nn.LeakyReLU(),
                                              BlurPool(32, filt_size=4, stride=first_layer_stride))
        self.bn1 = torch.nn.BatchNorm2d(num_features=32)
        self.maxpool1 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                            BlurPool(32, filt_size=2, stride=2, pad_off=0))
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=48, kernel_size=5,
                                     stride=1, padding=0)
        self.lrelu2 = torch.nn.LeakyReLU()
        self.bn2 = torch.nn.BatchNorm2d(num_features=48)
        self.maxpool2 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                            BlurPool(48, filt_size=2, stride=2, pad_off=0))
        self.conv3 = torch.nn.Conv2d(in_channels=48, out_channels=64, kernel_size=5,
                                     stride=1, padding=0)
        self.lrelu3 = torch.nn.LeakyReLU()
        self.bn3 = torch.nn.BatchNorm2d(num_features=64)
        self.maxpool3 = torch.nn.Sequential(torch.nn.MaxPool2d(kernel_size=2, stride=1, padding=0),
                                            BlurPool(64, filt_size=2, stride=2, pad_off=0))
        self.conv4 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4,
                                     stride=1, padding=0)
        self.lrelu4 = torch.nn.LeakyReLU()
        self.dense1 = torch.nn.LazyLinear(out_features=128)
        self.dense2 = torch.nn.Linear(128, num_classes)
        self.dropout = torch.nn.Dropout(p=0.2)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.maxpool1(self.bn1((self.conv1(x))))  # the activation is already inside the first convolutional layer
        x = self.maxpool1(self.bn1(self.lrelu1(self.conv1(x))))  # the activation is already inside the first convolutional layer
        x = self.maxpool2(self.bn2(self.lrelu2(self.conv2(x))))
        x = self.maxpool3(self.bn3(self.lrelu3(self.conv3(x))))
        x = self.lrelu4(self.conv4(x))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = F.leaky_relu(self.dense1(self.features(x)))
        x = self.dropout(x)
        x = self.dense2(x)
        return x


"""
ISPL DenseNet
"""

class ISPLDenseNet121(FeatureExtractor):
    """
    Custom implementation of the DenseNet121 from ISPL, where we can change the number of output classes even
    with pretrained models.
    """
    def __init__(self, num_classes: int, pretrained: bool = False, **kwargs):
        super(ISPLDenseNet121, self).__init__()

        # --- Build the network

        # Instantiate the feature extractor
        self.densenet = densenet.densenet121(pretrained=pretrained)
        # Instantiate the final classifier
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                              torch.nn.Linear(in_features=self.densenet.classifier.in_features,
                                                              out_features=num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.densenet.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # x = F.softmax(x, dim=-1)
        return x


class AAISPLDenseNet121(FeatureExtractor):
    """
    Custom implementation of the DenseNet121 from ISPL, where we can change the number of output classes even
    with pretrained models.
    AA = anti-aliasing, we use the modifications introduced in the paper "Making CNNs Shift-Invariant Again"
    (https://github.com/adobe/antialiased-cnns/tree/master)
    """
    def __init__(self, num_classes: int, pretrained: bool = True, **kwargs):
        super(AAISPLDenseNet121, self).__init__()

        # --- Parse the necessary parameters
        pool_only = kwargs.get('pool_only')
        if pool_only is None:
            raise ValueError('The pool_only parameter is mandatory for AADenseNet121')

        # --- Build the network

        # Instantiate the feature extractor
        if pool_only:
            self.densenet = antialiased_cnns.densenet121(pretrained=pretrained, pool_only=pool_only)
        else:
            # Create a model with no pool_only
            self.densenet = antialiased_cnns.densenet121(pretrained=False, pool_only=pool_only)
            if pretrained:
                # Load the pretrained weights with strict=False
                # This code is taken straight from the anti-aliased CNNs repository
                pattern = re.compile(
                    r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
                # state_dict = model_zoo.load_url(model_url)
                state_dict = model_zoo.load_url(antialiased_cnns.densenet.model_urls['densenet121_lpf4'],
                                                map_location='cpu', check_hash=True)['state_dict']
                for key in list(state_dict.keys()):
                    res = pattern.match(key)
                    if res:
                        new_key = res.group(1) + res.group(2)
                        state_dict[new_key] = state_dict[key]
                        del state_dict[key]
                self.densenet.load_state_dict(state_dict, strict=False)  # strict=False allows to load only the matching weights

        # Instantiate the final classifier
        self.classifier = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                              torch.nn.Linear(in_features=self.densenet.classifier.in_features,
                                                              out_features=num_classes))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.densenet.features(x)
        x = F.relu(x, inplace=True)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        # x = F.softmax(x, dim=-1)
        return x


"""
ISPL ResNet
"""

class ISPLResNet50(FeatureExtractor):
    """
    Custom implementation of the ResNet from ISPL, where we can change the number of output classes even
    with pretrained models.
    """
    def __init__(self, num_classes: int, pretrained: bool = False, **kwargs):
        super(ISPLResNet50, self).__init__()

        # --- Build the network

        # Instantiate the feature extractor
        self.resnet = resnet.resnet50(pretrained=pretrained)
        # Instantiate the final classifier
        self.fc = torch.nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_classes)
        del self.resnet.fc

    def forward_resnet_conv(self, x, upto: int = 4):
        """
        Forward ResNet only in its convolutional part
        :param net:
        :param x:
        :param upto:
        :return:
        """
        x = self.resnet.conv1(x)  # N / 2
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # N / 4

        if upto >= 1:
            x = self.resnet.layer1(x)  # N / 4
        if upto >= 2:
            x = self.resnet.layer2(x)  # N / 8
        if upto >= 3:
            x = self.resnet.layer3(x)  # N / 16
        if upto >= 4:
            x = self.resnet.layer4(x)  # N / 32
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_resnet_conv(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        # x = F.softmax(x, dim=-1)
        return x

class ISPLAAResNet50(FeatureExtractor):
    """
        Custom implementation of the AAResNet from ISPL, where we can change the number of output classes even
        with pretrained models.
        """

    def __init__(self, num_classes: int, pretrained: bool = False, **kwargs):
        super(ISPLAAResNet50, self).__init__()

        # --- Parse the necessary parameters
        pool_only = kwargs.get('pool_only')
        if pool_only is None:
            raise ValueError('The pool_only parameter is mandatory for AADenseNet121')

        # --- Build the network

        # Instantiate the feature extractor
        self.resnet = antialiased_cnns.resnet50(pretrained=pretrained, pool_only=pool_only)
        # Instantiate the final classifier
        self.fc = torch.nn.Linear(in_features=self.resnet.fc.in_features, out_features=num_classes)
        del self.resnet.fc

    def forward_resnet_conv(self, x, upto: int = 4):
        """
        Forward ResNet only in its convolutional part
        :param net:
        :param x:
        :param upto:
        :return:
        """
        x = self.resnet.conv1(x)  # N / 2
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)  # N / 4

        if upto >= 1:
            x = self.resnet.layer1(x)  # N / 4
        if upto >= 2:
            x = self.resnet.layer2(x)  # N / 8
        if upto >= 3:
            x = self.resnet.layer3(x)  # N / 16
        if upto >= 4:
            x = self.resnet.layer4(x)  # N / 32
        return x

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_resnet_conv(x)
        # Cannot use "squeeze" as batch-size can be 1
        x = self.resnet.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        # x = F.softmax(x, dim=-1)
        return x


"""
Model factory
"""

def create_model(model_name, params, device):
    if model_name == 'BondiNet':
        return BondiNet(**params).to(device)
    elif model_name == 'AABondiNet':
        return AABondiNet(**params).to(device)
    elif model_name == 'DenseNet121':
        return ISPLDenseNet121(**params).to(device)
    elif model_name == 'AADenseNet121':
        return AAISPLDenseNet121(**params).to(device)
    elif model_name == 'ResNet50':
        return ISPLResNet50(**params).to(device)
    elif model_name == 'AAResNet50':
        return ISPLAAResNet50(**params).to(device)
    else:
        raise ValueError(f"Invalid model name '{model_name}'")