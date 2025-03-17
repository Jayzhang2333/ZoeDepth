import torch
import torch.nn as nn
from zoedepth.models.layers.fusion_layers import BasicBlockFusion, conv_bn_relu, BasicBlockEarlyFusion, JointConvTransBlock
import torch.nn.functional as F
from zoedepth.models.layers.dat import DAT

def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True):
    if backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48 , 136 , 384], features, groups=groups, expand=expand)  # efficientnet_lite3     
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained, scratch

def _make_encoder_conv_trans(backbone, features, use_pretrained, groups=1, expand=False, exportable=True):
    if backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3_conv_trans(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48 , 136 , 384], features, groups=groups, expand=expand)  # efficientnet_lite3     
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained, scratch

def _make_encoder_original(backbone, features, use_pretrained, groups=1, expand=False, exportable=True):
    if backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3_original(use_pretrained, exportable=exportable)
        scratch = _make_scratch([32, 48 , 136 , 384], features, groups=groups, expand=expand)  # efficientnet_lite3     
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained, scratch


def _make_encoder_conv_cbam( features, groups=1, expand=False):
    
    encoder = ResNetEncoder(ratios=[4, 8, 16, 32],
                          channels=[48, 96, 192, 384],
                          enable_fusion=True)
    scratch = _make_scratch([ 48, 96, 192, 384], features, groups=groups, expand=expand)  # efficientnet_lite3     
    
        
    return encoder, scratch

def _make_encoder_conv_cbam_early_fusion( features, groups=1, expand=False):
    
    encoder = ResNetEncoderEarlyFusion(ratios=[4, 8, 8, 8],
                          channels=[96, 160, 224, 288],
                          enable_fusion=True)
    scratch = _make_scratch([ 96, 160, 224, 288], features, groups=groups, expand=expand)  # efficientnet_lite3     
    
        
    return encoder, scratch

def _make_encoder_conv_cbam_input_fusion( features, groups=1, expand=False):
    
    encoder = ResNetEncoderFusedInput(ratios=[4, 8, 8, 8],
                          channels=[128, 256, 384, 512],
                          enable_fusion=False)
    scratch = _make_scratch([ 128, 256, 384, 512], features, groups=groups, expand=expand)  # efficientnet_lite3     
    
        
    return encoder, scratch

def _make_encoder_joint( features, groups=1, expand=False):
    
    encoder = JointConvTransEncoder(ratios=[8, 8, 8, 8],
                          channels=[128, 196, 256, 384],
                          enable_fusion=False)
    scratch = _make_scratch([ 128, 196, 256, 384], features, groups=groups, expand=expand)  # efficientnet_lite3     
    
        
    return encoder, scratch

def _make_encoder_DAT( features, groups=1, expand=False):
    
    encoder = DAT()
    scratch = _make_scratch([ 96, 128, 256, 512], features, groups=groups, expand=expand)  # efficientnet_lite3     
    
        
    return encoder, scratch



def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand==True:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )

    return scratch


def _make_pretrained_efficientnet_lite3_original(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable
    )
    return _make_efficientnet_backbone(efficientnet)

def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
        in_chans=33
    )
    return _make_efficientnet_backbone(efficientnet)

def _make_pretrained_efficientnet_lite3_conv_trans(use_pretrained, exportable=False):
    efficientnet = torch.hub.load(
        "rwightman/gen-efficientnet-pytorch",
        "tf_efficientnet_lite3",
        pretrained=use_pretrained,
        exportable=exportable,
        in_chans=65
    )
    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained

class ProgressiveFusionMulti(nn.Module):
    def __init__(self, num_features, channels, final_dim = None, final_size = None, ):
        """
        Args:
            num_features (int): Number of feature maps to fuse.
            channels (int): Number of channels in each feature map.
        """
        super(ProgressiveFusionMulti, self).__init__()
        # We need (num_features - 1) fusion stages.
        self.fusion_convs = nn.ModuleList([
            nn.Conv2d(channels * 2, channels, kernel_size=1)
            for _ in range(num_features - 1)
        ])
        self.final_size = final_size
        self.final_dim = final_dim
        if final_size is not None:
            if final_dim is not None:
                self.extra_conv = nn.Conv2d(channels, final_dim, kernel_size=3,padding=1)
            else:
                self.extra_conv = nn.Conv2d(channels, channels, kernel_size=3,padding=1)
        elif final_dim is not None:
            self.extra_conv = nn.Conv2d(channels, final_dim, kernel_size=3,padding=1)

    def forward(self, features):
        """
        Args:
            features (list of Tensors): A list of feature maps ordered from highest to lowest resolution.
                                         Each tensor has shape [B, C, H, W] (different H and W).
        Returns:
            Tensor: The fused feature map at the highest resolution with shape [B, C, H_high, W_high].
        """
        # Start with the lowest resolution feature map.
        fused = features[-1]
        fusion_idx = 0

        # Fuse progressively from lower resolution to higher resolution.
        # Iterate from the second-to-last feature to the first.
        for i in range(len(features) - 2, -1, -1):
            # Upsample the current fused map to the resolution of the next higher feature.
            fused = F.interpolate(fused, size=features[i].shape[2:], mode='bilinear', align_corners=False)
            # Concatenate along the channel dimension.
            fused = torch.cat([features[i], fused], dim=1)
            # Fuse via a 1x1 conv to maintain the number of channels.
            fused = self.fusion_convs[fusion_idx](fused)
            fusion_idx += 1

        if self.final_size is not None:
            fused = F.interpolate(fused, size=self.final_size, mode='bilinear', align_corners=False)

        if self.final_size is not None or self.final_dim is not None:    
            fused = self.extra_conv(fused)
           

        return fused


class ResNetEncoderFusedInput(nn.Module):
        
    # originally
    def __init__(self, block=BasicBlockFusion, layers=[2, 2, 2, 2],
                 ratios=[16, 16, 16, 16], channels=[128, 256, 384, 512],
                 enable_fusion=False):
        """
        Args:
            block (nn.Module): Block type to use (here, the customized BasicBlock).
            layers (list): Number of blocks in each of the 4 layers.
            ratios (list): List of attention ratio values for each layer.
            channels (list): List of output channels for each layer.
            enable_fusion (bool): Whether to enable fusion in the last block of each layer.
        """
        super(ResNetEncoderFusedInput, self).__init__()

        self.conv1_ga = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.conv1_scale = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        self.conv1 = conv_bn_relu(96, 96, kernel=3, stride=1, padding=1,
                                      bn=False)
        
        self.layer1 = self._make_layer(block,96, channels[0], layers[0],
                                       stride=2, ratio=ratios[0], enable_fusion=enable_fusion)
        self.layer2 = self._make_layer(block, channels[0], channels[1], layers[1],
                                       stride=2, ratio=ratios[1], enable_fusion=enable_fusion)
        self.layer3 = self._make_layer(block,channels[1], channels[2], layers[2],
                                       stride=2, ratio=ratios[2], enable_fusion=enable_fusion)
        self.layer4 = self._make_layer(block,channels[2], channels[3], layers[3],
                                       stride=2, ratio=ratios[3], enable_fusion=enable_fusion)
    # originally
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, ratio=16, enable_fusion=False):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = []
        # If only one block exists, then it is also the last block and uses fusion if enabled.
        if blocks == 1:
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=enable_fusion))
           
        else:
            # First block: fusion disabled.
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=False))
            
            # Intermediate blocks: fusion disabled.
            for i in range(1, blocks - 1):
                layers.append(block(planes, planes, ratio=ratio, fusion=False))
            # Last block: fusion enabled if requested.
            layers.append(block(planes, planes, ratio=ratio, fusion=enable_fusion))
        return nn.Sequential(*layers)


    def forward(self, ga, scale, fuse_features=None):
        """
        Args:
            x (Tensor): Input image tensor.
            fuse_features (list or None): Optional list of 4 feature maps to fuse, one per residual layer.
                Each fusion feature must match the dimensions of the last block output of that layer.
        Returns:
            List[Tensor]: List of feature maps from the initial conv block and each of the 4 layers.
        """
        features = []

        # Initial conv block.
        ga_embedding = self.conv1_ga(ga)
        scale_embedding = self.conv1_scale(scale)
        x = torch.cat((ga_embedding, scale_embedding, fuse_features), dim=1)
        x = self.conv1(x)

     

        # Process layer1.
        for i, block in enumerate(self.layer1):
                x = block(x)
        features.append(x)

        # Process layer2.
        for i, block in enumerate(self.layer2):
                x = block(x)
        features.append(x)

        # Process layer3.
        for i, block in enumerate(self.layer3):
                x = block(x)
        features.append(x)

        # Process layer4.
        for i, block in enumerate(self.layer4):
                x = block(x)
        
        features.append(x)

        return features


class ResNetEncoder(nn.Module):
        
    # originally
    def __init__(self, block=BasicBlockFusion, layers=[2, 2, 2, 2],
                 ratios=[16, 16, 16, 16], channels=[48, 96, 192, 384],
                 enable_fusion=False):
        """
        Args:
            block (nn.Module): Block type to use (here, the customized BasicBlock).
            layers (list): Number of blocks in each of the 4 layers.
            ratios (list): List of attention ratio values for each layer.
            channels (list): List of output channels for each layer.
            enable_fusion (bool): Whether to enable fusion in the last block of each layer.
        """
        super(ResNetEncoder, self).__init__()

        self.conv1_ga = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.conv1_scale = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        self.conv1 = conv_bn_relu(32, 32, kernel=3, stride=1, padding=1,
                                      bn=False)
        
        self.layer1 = self._make_layer(block, 32, channels[0], layers[0],
                                       stride=2, ratio=ratios[0], enable_fusion=enable_fusion)
        self.layer2 = self._make_layer(block, channels[0], channels[1], layers[1],
                                       stride=2, ratio=ratios[1], enable_fusion=enable_fusion)
        self.layer3 = self._make_layer(block,channels[1], channels[2], layers[2],
                                       stride=2, ratio=ratios[2], enable_fusion=enable_fusion)
        self.layer4 = self._make_layer(block,channels[2], channels[3], layers[3],
                                       stride=2, ratio=ratios[3], enable_fusion=enable_fusion)
    # originally
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, ratio=16, enable_fusion=False):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = []
        # If only one block exists, then it is also the last block and uses fusion if enabled.
        if blocks == 1:
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=enable_fusion))
           
        else:
            # First block: fusion disabled.
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=False))
            
            # Intermediate blocks: fusion disabled.
            for i in range(1, blocks - 1):
                layers.append(block(planes, planes, ratio=ratio, fusion=False))
            # Last block: fusion enabled if requested.
            layers.append(block(planes, planes, ratio=ratio, fusion=enable_fusion))
        return nn.Sequential(*layers)


    def forward(self, ga, scale, fuse_features=None):
        """
        Args:
            x (Tensor): Input image tensor.
            fuse_features (list or None): Optional list of 4 feature maps to fuse, one per residual layer.
                Each fusion feature must match the dimensions of the last block output of that layer.
        Returns:
            List[Tensor]: List of feature maps from the initial conv block and each of the 4 layers.
        """
        features = []

        # Initial conv block.
        ga_embedding = self.conv1_ga(ga)
        scale_embedding = self.conv1_scale(scale)
        x = torch.cat((ga_embedding, scale_embedding), dim=1)
        x = self.conv1(x)

     

        # Process layer1.
        for i, block in enumerate(self.layer1):
            if fuse_features is not None and len(fuse_features) > 0 and i == len(self.layer1) - 1:
                # print('layer1 fuse')
                x = block(x, fuse=fuse_features[0])
            else:
                x = block(x)
        features.append(x)

        # Process layer2.
        for i, block in enumerate(self.layer2):
            if fuse_features is not None and len(fuse_features) > 1 and i == len(self.layer2) - 1:
                # print('layer2 fuse')
                x = block(x, fuse=fuse_features[1])
            else:
                x = block(x)
        features.append(x)

        # Process layer3.
        for i, block in enumerate(self.layer3):
            if fuse_features is not None and len(fuse_features) > 2 and i == len(self.layer3) - 1:
                # print('layer3 fuse')
                x = block(x, fuse=fuse_features[2])
            else:
                x = block(x)
        features.append(x)

        # Process layer4.
        for i, block in enumerate(self.layer4):
            if fuse_features is not None and len(fuse_features) > 3 and i == len(self.layer4) - 1:
                # print('layer4 fuse')
                x = block(x, fuse=fuse_features[3])
            else:
                x = block(x)
        
        features.append(x)

        return features
    


class ResNetEncoderEarlyFusion(nn.Module):
        
    # originally
    def __init__(self, block=BasicBlockEarlyFusion, layers=[2, 2, 2, 2],
                 ratios=[16, 16, 16, 16], channels=[96, 160, 224, 288],
                 enable_fusion=False):
        """
        Args:
            block (nn.Module): Block type to use (here, the customized BasicBlock).
            layers (list): Number of blocks in each of the 4 layers.
            ratios (list): List of attention ratio values for each layer.
            channels (list): List of output channels for each layer.
            enable_fusion (bool): Whether to enable fusion in the last block of each layer.
        """
        super(ResNetEncoderEarlyFusion, self).__init__()

        self.conv1_ga = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.conv1_scale = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        self.conv1 = conv_bn_relu(32, 32, kernel=3, stride=1, padding=1,
                                      bn=False)
        
        self.layer1 = self._make_layer(block, channels[0], channels[0], layers[0],
                                       stride=2, ratio=ratios[0], enable_fusion=enable_fusion)
        self.layer2 = self._make_layer(block, channels[1], channels[1], layers[1],
                                       stride=2, ratio=ratios[1], enable_fusion=enable_fusion)
        self.layer3 = self._make_layer(block,channels[2], channels[2], layers[2],
                                       stride=2, ratio=ratios[2], enable_fusion=enable_fusion)
        self.layer4 = self._make_layer(block,channels[3], channels[3], layers[3],
                                       stride=2, ratio=ratios[3], enable_fusion=enable_fusion)
    # originally
    def _make_layer(self, block, inplanes, planes, blocks, stride=1, ratio=16, enable_fusion=False):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = []
        # If only one block exists, then it is also the last block and uses fusion if enabled.
        if blocks == 1:
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=enable_fusion))
           
        else:
            # First block: fusion disabled.
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=enable_fusion))
            
            # Intermediate blocks: fusion disabled.
            for i in range(1, blocks):
                layers.append(block(planes, planes, ratio=ratio, fusion=False))

        return nn.Sequential(*layers)


    def forward(self, ga, scale, fuse_features=None):
        """
        Args:
            x (Tensor): Input image tensor.
            fuse_features (list or None): Optional list of 4 feature maps to fuse, one per residual layer.
                Each fusion feature must match the dimensions of the last block output of that layer.
        Returns:
            List[Tensor]: List of feature maps from the initial conv block and each of the 4 layers.
        """
        features = []

        # Initial conv block.
        ga_embedding = self.conv1_ga(ga)
        scale_embedding = self.conv1_scale(scale)
        x = torch.cat((ga_embedding, scale_embedding), dim=1)
        x = self.conv1(x)

     

        # Process layer1.
        for i, block in enumerate(self.layer1):
            if fuse_features is not None and len(fuse_features) > 0 and i == 0:
                # print('layer1 fuse')
                # print(fuse_features[0].shape)
                x = block(x, fuse=fuse_features[0])
            else:
                x = block(x)
        features.append(x)

        # Process layer2.
        for i, block in enumerate(self.layer2):
            if fuse_features is not None and len(fuse_features) > 1 and i == 0:
                # print('layer2 fuse')
                # print(fuse_features[1].shape)
                x = block(x, fuse=fuse_features[1])
            else:
                x = block(x)
        features.append(x)

        # Process layer3.
        for i, block in enumerate(self.layer3):
            if fuse_features is not None and len(fuse_features) > 2 and i == 0:
                # print('layer3 fuse')
                # print(fuse_features[2].shape)
                x = block(x, fuse=fuse_features[2])
            else:
                x = block(x)
        features.append(x)

        # Process layer4.
        for i, block in enumerate(self.layer4):
            if fuse_features is not None and len(fuse_features) > 3 and i == 0:
                # print('layer4 fuse')
                # print(fuse_features[3].shape)
                x = block(x, fuse=fuse_features[3])
            else:
                x = block(x)
        
        features.append(x)

        return features


class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock_custom_original(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom_original, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size = None):
        """Forward pass.

        Returns:
            tensor: output
        """

        if (size is None):
            modifier = {"scale_factor": 2}
        else:
            modifier = {"size": size}
        

        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        
        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )
        

        output = self.out_conv(output)

        return output


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, out_features, activation, deconv=False, bn=False, expand=False, align_corners=True):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        # out_features = features
        # if self.expand==True:
        #     out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output
    

class FeatureFusionBlock_DA(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size = None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_DA, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1
        self.size = size

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs, size = None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        # output = nn.functional.interpolate(
        #     output, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        # )

        output = self.out_conv(output)

        return output


class OutputConv(nn.Module):
    """Output conv block.
    """

    def __init__(self, features, groups, activation, non_negative):

        super(OutputConv, self).__init__()

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            # activation, # originally, this is an input activation function, not the leakyrelu below
            nn.LeakyReLU(0.2, inplace = False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def forward(self, x):        
        return self.output_conv(x)
    
class OutputConvInterpolation(nn.Module):
    """Output conv block.
    """

    def __init__(self, features, activation, non_negative):

        super(OutputConvInterpolation, self).__init__()

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

    def forward(self, x):        
        return self.output_conv(x)
    
class ConfOutputConvInterpolation(nn.Module):
    """Output conv block.
    """

    def __init__(self, features, activation):

        super(ConfOutputConvInterpolation, self).__init__()

        self.output_conv = nn.Sequential(
            nn.Conv2d(features, 32, kernel_size=3, stride=1, padding=1),
            activation,
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x):        
        return self.output_conv(x)
    
class DepthUncertaintyHead(nn.Module):
    def __init__(self, features, groups, activation, non_negative):
        super(DepthUncertaintyHead, self).__init__()
        
        # Head for the per-pixel scale correction factor (depth output)
        self.depth_head = nn.Sequential(
            nn.Conv2d(features, features//2, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(features//2, 32, kernel_size=3, stride=1, padding=1),
            # activation, # originally, this is an input activation function, not the leakyrelu below
            nn.LeakyReLU(0.2, inplace = False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )
        
        # Head for the uncertainty map output
        self.uncertainty_head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Softplus()  # alternative to ReLU; outputs are strictly positive
        )
    
    def forward(self, x):
        depth = self.depth_head(x)
        uncertainty = self.uncertainty_head(x)
        return depth, uncertainty
    

def weights_init(m):
    import math
    # initialize from normal (Gaussian) distribution
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2.0 / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    

class FeatureFusionBlock_mine(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, out_features, activation = nn.ReLU(False),  bn=False):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_mine, self).__init__()

        self.groups=1
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit = ResidualConvUnit_custom(features, activation, bn)

        self.out_conv.apply(weights_init)
        self.resConfUnit.apply(weights_init)
        
        
    def forward(self, x):
        """Forward pass.

        Returns:
            tensor: output
        """


        output = self.resConfUnit(x)

        output = self.out_conv(output)

        return output



class JointConvTransEncoder(nn.Module):
        
    # originally
    def __init__(self, layers=[2, 2, 2, 2],
                 ratios=[16, 16, 16, 16], channels=[128, 256, 384, 512],
                 enable_fusion=False):
        """
        Args:
            block (nn.Module): Block type to use (here, the customized BasicBlock).
            layers (list): Number of blocks in each of the 4 layers.
            ratios (list): List of attention ratio values for each layer.
            channels (list): List of output channels for each layer.
            enable_fusion (bool): Whether to enable fusion in the last block of each layer.
        """
        super(JointConvTransEncoder, self).__init__()

        self.conv1_ga = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        
        self.conv1_scale = conv_bn_relu(1, 16, kernel=3, stride=1, padding=1,
                                          bn=False)
        self.conv1 = conv_bn_relu(96, 96, kernel=3, stride=1, padding=1,
                                      bn=False)
        
        self.layer1 = self._make_residual_layer(BasicBlockFusion,96, channels[0], layers[0],
                                       stride=2, ratio=ratios[0], enable_fusion=enable_fusion)
        self.layer2 = self._make_residual_layer(BasicBlockFusion, channels[0], channels[1], layers[1],
                                       stride=2, ratio=ratios[1], enable_fusion=enable_fusion)
        self.layer3 = JointConvTransBlock(channels[1], channels[2], patch_size=2, num_layer=1, depth=layers[2])
        self.layer4 = JointConvTransBlock(channels[2], channels[3], patch_size=2, num_layer=1, depth=layers[3])
    # originally
    def _make_residual_layer(self, block, inplanes, planes, blocks, stride=1, ratio=16, enable_fusion=False):
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
        layers = []
        # If only one block exists, then it is also the last block and uses fusion if enabled.
        if blocks == 1:
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=enable_fusion))
           
        else:
            # First block: fusion disabled.
            layers.append(block(inplanes, planes, stride, downsample,
                                ratio=ratio, fusion=False))
            
            # Intermediate blocks: fusion disabled.
            for i in range(1, blocks - 1):
                layers.append(block(planes, planes, ratio=ratio, fusion=False))
            # Last block: fusion enabled if requested.
            layers.append(block(planes, planes, ratio=ratio, fusion=enable_fusion))
        return nn.Sequential(*layers)

    


    def forward(self, ga, scale, fuse_features=None):
        """
        Args:
            x (Tensor): Input image tensor.
            fuse_features (list or None): Optional list of 4 feature maps to fuse, one per residual layer.
                Each fusion feature must match the dimensions of the last block output of that layer.
        Returns:
            List[Tensor]: List of feature maps from the initial conv block and each of the 4 layers.
        """
        features = []

        # Initial conv block.
        ga_embedding = self.conv1_ga(ga)
        scale_embedding = self.conv1_scale(scale)
        # print(ga_embedding.shape)
        # print(scale_embedding.shape)
        # print(fuse_features.shape)
        x = torch.cat((ga_embedding, scale_embedding, fuse_features), dim=1)
        x = self.conv1(x)

     

        # Process layer1.
        for i, block in enumerate(self.layer1):
                x = block(x)
        features.append(x)

        # Process layer2.
        for i, block in enumerate(self.layer2):
                x = block(x)
        features.append(x)

        # Process layer3.
        
        x = self.layer3(x)
        features.append(x)

        # Process layer4.
        x = self.layer4(x)
        features.append(x)

        return features