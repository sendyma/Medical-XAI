import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from utils_model.receptive_field import compute_proto_layer_rf_info_v2
from efficientnet.model import EfficientNet


def efficientb0_features(pretrained=True, **kwargs):
    r"""EfficientNet_b0 model from
    `"efficientnet: rethinking model scaling for convolutional neural networks"`

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """

    if pretrained:
        # model = EfficientNet.from_pretrained("efficientnet-b0")
        model = EfficientNet.from_pretrained("efficientnet-b0", num_classes=2, image_size=768)
        # inputs = torch.rand(1, 3, 224, 224)
        # outputs = model(inputs)
        # print(outputs.shape)
        # aaa = model.conv_info()
    else:
        model = EfficientNet.from_name('efficientnet-b0')

    return model



base_architecture_to_features = {'efficientnet_b0': efficientb0_features}



def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index)


class PPNet(nn.Module):

    def __init__(self, features, img_size, prototype_shape,
                 proto_layer_rf_info, num_classes, init_weights=True,
                 prototype_activation_function='log',
                 add_on_layers_type='bottleneck'):

        super(PPNet, self).__init__()
        self.img_size = img_size
        self.prototype_shape = prototype_shape
        self.num_prototypes = prototype_shape[0]
        self.num_classes = num_classes
        self.epsilon = 1e-4

        # prototype_activation_function could be 'log', 'linear',
        # or a generic function that converts distance to similarity score
        self.prototype_activation_function = prototype_activation_function  # log

        '''
        Here we are initializing the class identities of the prototypes
        Without domain specific knowledge we allocate the same number of
        prototypes for each class
        '''
        assert (self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(self.num_prototypes, self.num_classes)  # 2000*200

        # num_prototypes_per_class = self.num_prototypes // self.num_classes  # 10
        # for j in range(self.num_prototypes):
        #     self.prototype_class_identity[j, j // num_prototypes_per_class] = 1

        self.num_pos_proto = self.num_prototypes // 2
        self.num_neg_proto = self.num_prototypes - self.num_pos_proto
        self.prototype_class_identity[0:self.num_neg_proto, 0] = 1
        self.prototype_class_identity[self.num_neg_proto:, 1] = 1

        self.proto_layer_rf_info = proto_layer_rf_info  # [7, 32, 268, 16]

        # this has to be named features to allow the precise loading
        features._conv_stem = nn.Identity()
        features._bn0 = nn.Identity()
        features._swish = nn.Identity()
        self.features = features

        first_add_on_layer_in_channels = 1280

        if add_on_layers_type == 'bottleneck':
            add_on_layers = []
            current_in_channels = first_add_on_layer_in_channels
            while (current_in_channels > self.prototype_shape[1]) or (len(add_on_layers) == 0):
                current_out_channels = max(self.prototype_shape[1], (current_in_channels // 2))
                add_on_layers.append(nn.Conv2d(in_channels=current_in_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                add_on_layers.append(nn.ReLU())
                add_on_layers.append(nn.Conv2d(in_channels=current_out_channels,
                                               out_channels=current_out_channels,
                                               kernel_size=1))
                if current_out_channels > self.prototype_shape[1]:
                    add_on_layers.append(nn.ReLU())
                else:
                    assert (current_out_channels == self.prototype_shape[1])
                    add_on_layers.append(nn.Sigmoid())
                current_in_channels = current_in_channels // 2
            self.add_on_layers = nn.Sequential(*add_on_layers)
        else:
            self.add_on_layers = nn.Sequential(
                nn.Dropout(0.2),  ######################################################
                nn.Conv2d(in_channels=first_add_on_layer_in_channels, out_channels=self.prototype_shape[1], kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=self.prototype_shape[1], out_channels=self.prototype_shape[1], kernel_size=1),
                nn.Sigmoid()  ############################################################
            )  # 512-->128-->128

        self.prototype_vectors = nn.Parameter(torch.rand(self.prototype_shape), requires_grad=True)

        # do not make this just a tensor,
        # since it will not be moved automatically to gpu
        self.ones = nn.Parameter(torch.ones(self.prototype_shape), requires_grad=False)

        # self.last_layer = nn.Linear(self.num_prototypes, self.num_classes, bias=False)  # do not use bias
        self.last_layer_neg = nn.Linear(self.num_prototypes//2, 1, bias=False)  # do not use bias
        self.last_layer_pos = nn.Linear(self.num_prototypes//2, 1, bias=False)  # do not use bias
        self.last_layer_dropout = nn.Dropout(0.0)

        if init_weights:
            self._initialize_weights()  ####################

    def conv_features(self, x):
        '''
        the feature input to prototype layer
        '''
        x = self.features(x)
        x = self.add_on_layers(x)
        return x

    @staticmethod
    def _weighted_l2_convolution(input, filter, weights):
        '''
        input of shape N * c * h * w
        filter of shape P * c * h1 * w1
        weight of shape P * c * h1 * w1
        '''
        input2 = input ** 2
        input_patch_weighted_norm2 = F.conv2d(input=input2, weight=weights)

        filter2 = filter ** 2
        weighted_filter2 = filter2 * weights
        filter_weighted_norm2 = torch.sum(weighted_filter2, dim=(1, 2, 3))
        filter_weighted_norm2_reshape = filter_weighted_norm2.view(-1, 1, 1)

        weighted_filter = filter * weights
        weighted_inner_product = F.conv2d(input=input, weight=weighted_filter)

        # use broadcast
        intermediate_result = \
            - 2 * weighted_inner_product + filter_weighted_norm2_reshape
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(input_patch_weighted_norm2 + intermediate_result)

        return distances

    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        x2 = x ** 2  # [91, 128, 7, 7]
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)  # [91, 2000, 7, 7]

        p2 = self.prototype_vectors ** 2  # [2000, 128, 1, 1]
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)  # [2000, 1, 1]

        xp = F.conv2d(input=x, weight=self.prototype_vectors)  # [91, 2000, 7, 7]
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def prototype_distances(self, x):
        '''
        x is the raw input
        '''
        conv_features = self.conv_features(x)  # [91, 128, 7, 7] after sigmoid
        distances = self._l2_convolution(conv_features)  # [91, 2000, 7, 7]
        return distances

    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)

    def distance_2_similarity_exp(self, distances):
        return torch.exp(-distances / 128.0)  # 128.0

    def distance_2_similarity_linear(self, distances):
        return -distances

    def forward(self, x, return_full=False):

        x_backbone = self.features(x)
        conv_output = self.add_on_layers(x_backbone)

        distances = self._l2_convolution(conv_output)  # [91, 2000, 7, 7], 2000 prototypes, 2000 similarity(activation) score maps
        similarity_maps = self.distance_2_similarity_exp(distances)

        # distances = self.prototype_distances(x)  # [91, 2000, 7, 7], 2000 prototypes, 2000 similarity(activation) score maps
        '''
        we cannot refactor the lines below for similarity scores
        because we need to return min_distances 
        '''
        # global max pooling
        minus_min_distances, spatial_index = F.max_pool2d(-distances,
                                                          kernel_size=(distances.size()[2], distances.size()[3]),
                                                          return_indices=True)
        min_distances = -minus_min_distances.view(-1, self.num_prototypes)  # [b, num_proto, 1, 1]
        spatial_index = spatial_index.view(-1, self.num_prototypes)  # [b, num_proto]

        prototype_activations = self.distance_2_similarity_exp(min_distances)  # [b, num_proto]

        # logits = self.last_layer(prototype_activations)
        logits_neg = self.last_layer_neg(self.last_layer_dropout(prototype_activations[:, 0:self.num_prototypes//2]))
        logits_pos = self.last_layer_pos(self.last_layer_dropout(prototype_activations[:, self.num_prototypes//2:]))
        logits = torch.cat((logits_neg, logits_pos), dim=1)
        if return_full:
            return logits, min_distances, similarity_maps, prototype_activations, conv_output, spatial_index
        else:
            return logits, min_distances, similarity_maps

    def push_forward(self, x):
        '''this method is needed for the pushing operation'''

        x_backbone = self.features(x)
        conv_output = self.add_on_layers(x_backbone)
        distances = self._l2_convolution(conv_output)
        return conv_output, distances

    def prune_prototypes(self, prototypes_to_prune):
        '''
        prototypes_to_prune: a list of indices each in
        [0, current number of prototypes - 1] that indicates the prototypes to
        be removed
        '''
        prototypes_to_keep = list(set(range(self.num_prototypes)) - set(prototypes_to_prune))

        self.prototype_vectors = nn.Parameter(self.prototype_vectors.data[prototypes_to_keep, ...],
                                              requires_grad=True)

        self.prototype_shape = list(self.prototype_vectors.size())
        self.num_prototypes = self.prototype_shape[0]

        # changing self.last_layer in place
        # changing in_features and out_features make sure the numbers are consistent
        self.last_layer.in_features = self.num_prototypes
        self.last_layer.out_features = self.num_classes
        self.last_layer.weight.data = self.last_layer.weight.data[:, prototypes_to_keep]

        # self.ones is nn.Parameter
        self.ones = nn.Parameter(self.ones.data[prototypes_to_keep, ...], requires_grad=False)
        # self.prototype_class_identity is torch tensor
        # so it does not need .data access for value update
        self.prototype_class_identity = self.prototype_class_identity[prototypes_to_keep, :]

    def __repr__(self):
        # PPNet(self, features, img_size, prototype_shape,
        # proto_layer_rf_info, num_classes, init_weights=True):
        rep = (
            'PPNet(\n'
            '\tfeatures: {},\n'
            '\timg_size: {},\n'
            '\tprototype_shape: {},\n'
            '\tproto_layer_rf_info: {},\n'
            '\tnum_classes: {},\n'
            '\tepsilon: {}\n'
            ')'
        )

        return rep.format(self.features,
                          self.img_size,
                          self.prototype_shape,
                          self.proto_layer_rf_info,
                          self.num_classes,
                          self.epsilon)

    def set_last_layer_incorrect_connection(self, incorrect_strength):
        '''
        the incorrect strength will be actual strength if -0.5 then input -0.5
        '''
        positive_one_weights_locations = torch.t(self.prototype_class_identity)
        negative_one_weights_locations = 1 - positive_one_weights_locations

        correct_class_connection = 1
        incorrect_class_connection = incorrect_strength
        self.last_layer.weight.data.copy_(
            correct_class_connection * positive_one_weights_locations
            + incorrect_class_connection * negative_one_weights_locations)

    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)
        self.last_layer_neg.weight.data.fill_(1.0)
        self.last_layer_pos.weight.data.fill_(1.0)


def construct_ProtoPNet(base_architecture, pretrained=True, img_size=224,
                        prototype_shape=(2000, 512, 1, 1), num_classes=200,
                        prototype_activation_function='log',
                        add_on_layers_type='bottleneck'):
    features = base_architecture_to_features[base_architecture](pretrained=pretrained)
    # layer_filter_sizes, layer_strides, layer_paddings = features.conv_info()
    # proto_layer_rf_info = compute_proto_layer_rf_info_v2(img_size=img_size,
    #                                                      layer_filter_sizes=layer_filter_sizes,
    #                                                      layer_strides=layer_strides,
    #                                                      layer_paddings=layer_paddings,
    #                                                      prototype_kernel_size=prototype_shape[2])
    proto_layer_rf_info = [7, 32, 268, 16]  # for efficient_b0
    return PPNet(features=features,
                 img_size=img_size,
                 prototype_shape=prototype_shape,
                 proto_layer_rf_info=proto_layer_rf_info,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type)