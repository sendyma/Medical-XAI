import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from efficientnet.model import EfficientNet
import copy


def efficientb0_features(pretrained=True, **kwargs):
    r"""EfficientNet_b0 model from
    `"efficientnet: rethinking model scaling for convolutional neural networks"`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        # model = EfficientNet.from_pretrained("efficientnet_-b0")
        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2, image_size=768)
        # inputs = torch.rand(1, 3, 224, 224)
        # outputs = model(inputs)
        # print(outputs.shape)
        # aaa = model.conv_info()
    else:
        model = EfficientNet.from_name('efficientnet-b0')

    return model



base_architecture_to_features = {'efficientnet_b0': efficientb0_features}



class GlobalNet(nn.Module):

    def __init__(self, features, num_classes):

        super(GlobalNet, self).__init__()

        # this has to be named features to allow the precise loading
        self._conv_stem_backbone = copy.deepcopy(features._conv_stem)
        self._bn0_backbone = copy.deepcopy(features._bn0)
        self._swish_backbone = copy.deepcopy(features._swish)

        features._conv_stem = nn.Identity()
        features._bn0 = nn.Identity()
        features._swish = nn.Identity()
        self.features = features
        # self.features = nn.Sequential(*list(features.children())[2:])

        first_add_on_layer_in_channels = 1280   # efficientnet-b0
        # first_add_on_layer_in_channels = 1024    # densenet-121
        # first_add_on_layer_in_channels = 1664    # densenet-169

        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._max_pooling = nn.AdaptiveMaxPool2d(1)
        self._dropout = nn.Dropout(0.2)
        self.global_classification_layers = nn.Sequential(nn.Linear(first_add_on_layer_in_channels, 4096),
                                                          nn.ReLU(),
                                                          nn.Dropout(0.5),
                                                          nn.Linear(4096, num_classes)
                                                          )

    def forward(self, x):

        x_backbone = self.features(x)
        x_global = self._avg_pooling(x_backbone)
        x_global = x_global.flatten(start_dim=1)
        x_global = self._dropout(x_global)
        x_global = self.global_classification_layers(x_global)

        return x_global


    def forward_backbone(self, inputs):

        # Stem
        x = self._swish_backbone(self._bn0_backbone(self._conv_stem_backbone(inputs)))

        return x


def construct_GlobalNet(base_architecture, pretrained=True, num_classes=2):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    return GlobalNet(features=features, num_classes=num_classes)
