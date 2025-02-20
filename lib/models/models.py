import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50
import torchvision.models as models


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).

    from https://github.com/facebookresearch/ConvNeXt
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

class PrototypeLayer(nn.Module):
    def __init__(
            self, 
            num_classes, 
            paf='linear',
            init_weights=False,
            prototype_shape = (10, 1024, 1, 1),
            falc=2048
        ):
        super().__init__()
        self.prototype_shape = prototype_shape
        self.num_classes = num_classes
        self.prototype_activation_function = paf
        self.epsilon = 1e-4
        self.num_prototypes = self.prototype_shape[0]

        assert(self.num_prototypes % self.num_classes == 0)
        # a onehot indication matrix for each prototype's class identity
        self.prototype_class_identity = torch.zeros(
            self.num_prototypes,
            self.num_classes,
            device='cuda'
        )

        num_prototypes_per_class = self.num_prototypes // self.num_classes
        for j in range(self.num_prototypes):
            self.prototype_class_identity[j, j // num_prototypes_per_class] = 1
        
        # Initialize prototypes
        first_add_on_layer_in_channels = falc
        self.add_on_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=first_add_on_layer_in_channels, 
                out_channels=self.prototype_shape[1], 
                kernel_size=1
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=self.prototype_shape[1], 
                out_channels=self.prototype_shape[1], 
                kernel_size=1
            ),
            # nn.Softmax(dim=1)
            nn.Sigmoid()
            # nn.ReLU()
        )
        
        self.prototypes = nn.Parameter(
            torch.rand(self.prototype_shape),
            requires_grad=True
        )
        self.ones = nn.Parameter(
            torch.ones(self.prototype_shape),
            requires_grad=False
        )
        self.last_layer = nn.Linear(
            self.num_prototypes, 
            self.num_classes,
            bias=True
        ) 

        if init_weights:
            self._initialize_weights()
    
    def _l2_convolution(self, x):
        '''
        apply self.prototype_vectors as l2-convolution filters on input x
        '''
        
        x2 = x ** 2
        x2_patch_sum = F.conv2d(input=x2, weight=self.ones)

        p2 = self.prototypes ** 2
        p2 = torch.sum(p2, dim=(1, 2, 3))
        # p2 is a vector of shape (num_prototypes,)
        # then we reshape it to (num_prototypes, 1, 1)
        p2_reshape = p2.view(-1, 1, 1)

        xp = F.conv2d(input=x, weight=self.prototypes)
        intermediate_result = - 2 * xp + p2_reshape  # use broadcast
        # x2_patch_sum and intermediate_result are of the same shape
        distances = F.relu(x2_patch_sum + intermediate_result)

        return distances

    def compute_prototype_distances(self, x):
        """Compute distances between input features and prototypes."""

        x = self.add_on_layers(x)
        distances = self._l2_convolution(x)

        return distances
    
    def distance_2_similarity(self, distances):
        if self.prototype_activation_function == 'log':
            return torch.log((distances + 1) / (distances + self.epsilon))
        elif self.prototype_activation_function == 'linear':
            return -distances
        else:
            return self.prototype_activation_function(distances)
    
    def _initialize_weights(self):
        for m in self.add_on_layers.modules():
            if isinstance(m, nn.Conv2d):
                # every init technique has an underscore _ in the name
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        self.set_last_layer_incorrect_connection(incorrect_strength=-0.5)

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
    
    def forward(self, x):
        """
        Returns:
            classifier_output: shape (N, in_features)
            distances: shape (N, num_prototypes)
        """
        distances = self.compute_prototype_distances(x)

        min_distances = -F.max_pool2d(
            -distances,
            kernel_size=(distances.size()[2], distances.size()[3])
        )
        min_distances = min_distances.view(-1, self.num_prototypes)
        prototype_activations = self.distance_2_similarity(min_distances)
        logits = self.last_layer(prototype_activations)
        return logits, min_distances

class AdapterModule(nn.Module):
    def __init__(self, input_dim, reduction_factor=16, init_weights=False):
        super(AdapterModule, self).__init__()
        hidden_dim = input_dim // reduction_factor
        self.adapter = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, input_dim, kernel_size=1),
            # nn.ReLU()
        )

        if init_weights:
            for layer in self.adapter:
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(
                        layer.weight, 
                        mode='fan_out', 
                        nonlinearity='relu'
                    )
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        return x + self.adapter(x)

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        
        self.base_model = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V1
        )

        if not args.fine_tune:
            # Freeze all parameters of the base ResNet
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.base_model.fc = nn.Identity()

        # Add adapters to specific layers
        self.adapters = nn.ModuleDict(
            {
                "layer1": AdapterModule(
                    256,
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer2": AdapterModule(
                    512, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer3": AdapterModule(
                    1024, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer4": AdapterModule(
                    2048, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
            }
        )

        self.prototype_layer = PrototypeLayer(
            num_classes=2,
            paf=args.proto_activation,
            prototype_shape=args.proto_shape,
            init_weights=args.init_weight_proto,
            falc=2048
        )

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        # maybe add dropout and maxpool here

        x = self.base_model.layer1(x)
        x = self.adapters["layer1"](x)
        
        x = self.base_model.layer2(x)
        x = self.adapters["layer2"](x)
        
        x = self.base_model.layer3(x)
        x = self.adapters["layer3"](x)
        
        x = self.base_model.layer4(x)
        x = self.adapters["layer4"](x) 

        class_logits, proto_distance = self.prototype_layer(x)
        
        return class_logits, proto_distance
    

class DenseNet(nn.Module):
    def __init__(self, args):
        super(DenseNet, self).__init__()
        
        self.base_model = models.densenet121(
            weights=models.DenseNet121_Weights.IMAGENET1K_V1
        ).features

        if not args.fine_tune:
            # Freeze all parameters of the base ResNet
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Add adapters to specific layers
        self.adapters = nn.ModuleDict(
            {
                "layer1": AdapterModule(
                    128,
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer2": AdapterModule(
                    256, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer3": AdapterModule(
                    512, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer4": AdapterModule(
                    1024, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
            }
        )

        self.prototype_layer = PrototypeLayer(
            num_classes=2,
            paf=args.proto_activation,
            prototype_shape=args.proto_shape,
            init_weights=args.init_weight_proto,
            falc=1024
        )
    
    def forward(self, x):
        x = self.base_model.conv0(x)
        x = self.base_model.norm0(x)
        x = self.base_model.relu0(x)
        x = self.base_model.pool0(x)
        x = self.base_model.denseblock1(x)
        x = self.base_model.transition1(x)
        x = self.adapters["layer1"](x)

        x = self.base_model.denseblock2(x)
        x = self.base_model.transition2(x)
        x = self.adapters["layer2"](x)

        x = self.base_model.denseblock3(x)
        x = self.base_model.transition3(x)
        x = self.adapters["layer3"](x)

        x = self.base_model.denseblock4(x)
        x = self.base_model.norm5(x)
        x = self.adapters["layer4"](x)

        class_logits, proto_distance = self.prototype_layer(x)
        
        return class_logits, proto_distance
    

class ConvNext(nn.Module):
    def __init__(self, args):
        super(ConvNext, self).__init__()
        
        self.base_model = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        ).features

        if not args.fine_tune:
            # Freeze all parameters of the base ResNet
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.base_model.fc = nn.Identity()
        self.Fconv = nn.Sequential(
            LayerNorm(768, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(
                in_channels=768, 
                out_channels=1536, 
                kernel_size=(2, 2), 
                stride=2
            )
        )

        # Add adapters to specific layers
        self.adapters = nn.ModuleDict(
            {
                "layer1": AdapterModule(
                    192,
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer2": AdapterModule(
                    384, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer3": AdapterModule(
                    768, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer4": AdapterModule(
                    1536, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
            }
        )

        self.prototype_layer = PrototypeLayer(
            num_classes=2,
            paf=args.proto_activation,
            prototype_shape=args.proto_shape,
            init_weights=args.init_weight_proto,
            falc=1536
        )
    
    def forward(self, x):
        x = self.base_model[0](x)
        x = self.base_model[1](x)
        x = self.base_model[2](x)
        x = self.adapters["layer1"](x)

        x = self.base_model[3](x)
        x = self.base_model[4](x)
        x = self.adapters["layer2"](x)

        x = self.base_model[5](x)
        x = self.base_model[6](x)
        x = self.adapters["layer3"](x)

        x = self.base_model[7](x)
        x = self.Fconv(x)
        x = self.adapters["layer4"](x)

        class_logits, proto_distance = self.prototype_layer(x)
        
        return class_logits, proto_distance
    

class RegNet(nn.Module):
    def __init__(self, args):
        super(RegNet, self).__init__()
        
        self.base_model = models.regnet_y_8gf(
            weights=models.RegNet_Y_8GF_Weights.IMAGENET1K_V1
        )

        if not args.fine_tune:
            # Freeze all parameters of the base ResNet
            for param in self.base_model.parameters():
                param.requires_grad = False

        self.base_model.fc = nn.Identity()

        # Add adapters to specific layers
        self.adapters = nn.ModuleDict(
            {
                "layer1": AdapterModule(
                    224,
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer2": AdapterModule(
                    448, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer3": AdapterModule(
                    896, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer4": AdapterModule(
                    2016, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
            }
        )

        self.prototype_layer = PrototypeLayer(
            num_classes=2,
            paf=args.proto_activation,
            prototype_shape=args.proto_shape,
            init_weights=args.init_weight_proto,
            falc=2016
        )
    
    def forward(self, x):
        x = self.base_model.stem(x)
        x = self.base_model.trunk_output.block1(x)
        x = self.adapters["layer1"](x)

        x = self.base_model.trunk_output.block2(x)
        x = self.adapters["layer2"](x)

        x = self.base_model.trunk_output.block3(x)
        x = self.adapters["layer3"](x)

        x = self.base_model.trunk_output.block4(x)
        x = self.adapters["layer4"](x)

        class_logits, proto_distance = self.prototype_layer(x)
        
        return class_logits, proto_distance
    

class VGG(nn.Module):
    def __init__(self, args):
        super(VGG, self).__init__()
        
        self.base_model = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features

        if not args.fine_tune:
            # Freeze all parameters of the base ResNet
            for param in self.base_model.parameters():
                param.requires_grad = False

        # self.base_model.fc = nn.Identity()

        # Add adapters to specific layers
        self.adapters = nn.ModuleDict(
            {
                "layer1": AdapterModule(
                    128,
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer2": AdapterModule(
                    256, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer3": AdapterModule(
                    512, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
                "layer4": AdapterModule(
                    512, 
                    args.red_factor, 
                    args.init_weight_adapter
                ),
            }
        )

        self.prototype_layer = PrototypeLayer(
            num_classes=2,
            paf=args.proto_activation,
            prototype_shape=args.proto_shape,
            init_weights=args.init_weight_proto,
            falc=512
        )
    
    def forward(self, x):
        x = self.base_model[0](x)
        x = self.base_model[1](x)
        x = self.base_model[2](x)
        x = self.base_model[3](x)
        x = self.base_model[4](x)
        x = self.base_model[5](x)
        x = self.base_model[6](x)
        x = self.base_model[7](x)
        x = self.base_model[8](x)
        x = self.base_model[9](x)
        x = self.adapters["layer1"](x)

        x = self.base_model[10](x)
        x = self.base_model[11](x)
        x = self.base_model[12](x)
        x = self.base_model[13](x)
        x = self.base_model[14](x)
        x = self.base_model[15](x)
        x = self.base_model[16](x)
        x = self.adapters["layer2"](x)

        x = self.base_model[17](x)
        x = self.base_model[18](x)
        x = self.base_model[19](x)
        x = self.base_model[20](x)
        x = self.base_model[21](x)
        x = self.base_model[22](x)
        x = self.base_model[23](x)
        x = self.adapters["layer3"](x)

        x = self.base_model[24](x)
        x = self.base_model[25](x)
        x = self.base_model[26](x)
        x = self.base_model[27](x)
        x = self.base_model[28](x)
        x = self.base_model[29](x)
        x = self.base_model[30](x)
        x = self.adapters["layer4"](x)

        class_logits, proto_distance = self.prototype_layer(x)
        
        return class_logits, proto_distance