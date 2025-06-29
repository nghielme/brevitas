# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import List

from torch import Tensor
import torch.nn as nn

import brevitas.nn as qnn
from brevitas.nn.mixin import act
from brevitas.quant import Int8WeightPerChannelFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Int32Bias
from brevitas.quant import TruncTo8bit
from brevitas.quant.experimental.float_quant_ocp import Fp8e4m3OCPAct, Fp8e4m3OCPWeight
from brevitas.quant_tensor.base_quant_tensor import QuantTensor


def make_quant_conv2d(
        in_channels,
        out_channels,
        kernel_size,
        weight_bit_width,
        weight_quant,
        input_quant=None,
        output_quant=None,
        stride=1,
        padding=0,
        bias=False):
    return qnn.QuantConv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        bias=bias,
        weight_quant=weight_quant,
        weight_bit_width=weight_bit_width,
        input_quant=input_quant,
        output_quant=output_quant
        )


class QuantBasicBlock(nn.Module):
    """
    Quantized BasicBlock implementation with extra relu activations to respect FINN constraints on the sign of residual
    adds. Ok to train from scratch, but doesn't lend itself to e.g. retrain from torchvision.
    """
    expansion = 1

    def __init__(
            self,
            in_planes,
            planes,
            stride=1,
            bias=False,
            shared_quant_act=None,
            act_bit_width=8,
            weight_bit_width=8,
            weight_quant=Int8WeightPerChannelFloat,
            act_quant=None,
            quant_type='FIXED'):
        super(QuantBasicBlock, self).__init__()
        self.quant_type = quant_type
        self.conv1 = make_quant_conv2d(
            in_planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            output_quant=act_quant)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = qnn.QuantReLU(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True)
        self.conv2 = make_quant_conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            weight_bit_width=weight_bit_width,
            weight_quant=weight_quant,
            output_quant=act_quant)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                make_quant_conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=bias,
                    weight_bit_width=weight_bit_width,
                    weight_quant=weight_quant,
                    output_quant=act_quant),
                nn.BatchNorm2d(self.expansion * planes),
                # We add a ReLU activation here because FINN requires the same sign along residual adds
                qnn.QuantReLU(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True))
            # Redefine shared_quant_act whenever shortcut is performing downsampling
            shared_quant_act = self.downsample[-1]
        if shared_quant_act is None:
            shared_quant_act = qnn.QuantReLU(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True)
        # We add a ReLU activation here because FINN requires the same sign along residual adds
        self.relu2 = shared_quant_act
        self.relu_out = qnn.QuantReLU(act_quant=act_quant, return_quant_tensor=True, bit_width=act_bit_width)

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        if len(self.downsample):
            x = self.downsample(x)
        # Check that the addition is made explicitly among QuantTensor structures
        if self.quant_type != 'FLOAT':
            assert isinstance(out, QuantTensor), "Perform add among QuantTensors"
            assert isinstance(x, QuantTensor), "Perform add among QuantTensors"
        out = out + x
        out = self.relu_out(out)
        return out


class QuantResNet(nn.Module):

    def __init__(
            self,
            block_impl,
            num_blocks: List[int],
            first_maxpool=False,
            zero_init_residual=False,
            num_classes=10,
            act_bit_width=8,
            weight_bit_width=8,
            round_average_pool=False,
            last_layer_bias_quant=Int32Bias,
            weight_quant=Int8WeightPerChannelFloat,
            first_layer_weight_quant=Int8WeightPerChannelFloat,
            last_layer_weight_quant=Int8WeightPerTensorFloat,
            act_quant=None,
            quant_type='FIXED'):
        super(QuantResNet, self).__init__()
        
        self.quant_type = quant_type
        self.in_planes = 64
        self.conv1 = make_quant_conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            weight_bit_width=weight_bit_width,
            weight_quant=first_layer_weight_quant,
            input_quant=act_quant, # first layer, add quant to input
            output_quant=act_quant)
        self.bn1 = nn.BatchNorm2d(64)
        shared_quant_act = qnn.QuantReLU(act_quant=act_quant, bit_width=act_bit_width, return_quant_tensor=True)
        self.relu = shared_quant_act
        # MaxPool is typically present for ImageNet but not for CIFAR10
        if first_maxpool:
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.Identity()

        self.layer1, shared_quant_act = self._make_layer(
            block_impl, 64, num_blocks[0], 1, shared_quant_act, weight_bit_width, act_bit_width, weight_quant, act_quant, quant_type)
        self.layer2, shared_quant_act = self._make_layer(
            block_impl, 128, num_blocks[1], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant, act_quant, quant_type)
        self.layer3, shared_quant_act = self._make_layer(
            block_impl, 256, num_blocks[2], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant, act_quant, quant_type)
        self.layer4, _ = self._make_layer(
            block_impl, 512, num_blocks[3], 2, shared_quant_act, weight_bit_width, act_bit_width, weight_quant, act_quant, quant_type)

        if self.quant_type == 'FLOAT':
            self.final_pool = nn.AvgPool2d(
                kernel_size=4)
        else:
            # Performs truncation to 8b (without rounding), which is supported in FINN
            avgpool_float_to_int_impl_type = 'ROUND' if round_average_pool else 'FLOOR'
            self.final_pool = qnn.TruncAvgPool2d(
                kernel_size=4,
                trunc_quant=TruncTo8bit,
                float_to_int_impl_type=avgpool_float_to_int_impl_type)
        # Keep last layer at 8b
        self.linear = qnn.QuantLinear(
            512 * block_impl.expansion,
            num_classes,
            weight_bit_width=weight_bit_width,
            bias=True,
            bias_quant=last_layer_bias_quant,
            weight_quant=last_layer_weight_quant,
            input_quant=act_quant)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, QuantBasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
            self,
            block_impl,
            planes,
            num_blocks,
            stride,
            shared_quant_act,
            weight_bit_width,
            act_bit_width,
            weight_quant,
            act_quant,
            quant_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            block = block_impl(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                bias=False,
                shared_quant_act=shared_quant_act,
                act_bit_width=act_bit_width,
                weight_bit_width=weight_bit_width,
                weight_quant=weight_quant,
                act_quant=act_quant,
                quant_type=quant_type)
            layers.append(block)
            shared_quant_act = layers[-1].relu_out
            self.in_planes = planes * block_impl.expansion
        return nn.Sequential(*layers), shared_quant_act

    def forward(self, x: Tensor):
        # There is no input quantizer, we assume the input is already 8b RGB
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.final_pool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def float_weight_act_class_factory(
    weight_bit_width, act_bit_width,
    act_e_bits, act_m_bits,
    weight_e_bits, weight_m_bits
):
    # Determine weight class
    weight_quant = Fp8e4m3OCPWeight
    weight_quant_class = weight_quant.let(**{'exponent_bit_width' : weight_e_bits, 'mantissa_bit_width' : weight_m_bits, 'bit_width' : weight_bit_width})
    
    act_quant = Fp8e4m3OCPAct
    act_quant_class = act_quant.let(**{'exponent_bit_width' : act_e_bits, 'mantissa_bit_width' : act_m_bits, 'bit_width' : act_bit_width})

    return weight_quant_class, act_quant_class


def get_params_from_config(cfg):
    
    def get_float_params(cfg):
        weight_bit_width = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
        weight_exp_bits = cfg.getint('FLOAT', 'WEIGHT_EXPONENT_BITS')
        weight_mant_bits = cfg.getint('FLOAT', 'WEIGHT_MANTISSA_BITS')
        act_bit_width = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
        act_exp_bits = cfg.getint('FLOAT', 'ACT_EXPONENT_BITS')
        act_mant_bits = cfg.getint('FLOAT', 'ACT_MANTISSA_BITS')
        
        assert weight_bit_width in [4, 6, 8], "Weight bit width must be one of [4, 6, 8]"
        assert act_bit_width in [4, 6, 8], "Activation bit width must be one of [4, 6, 8]"
        assert weight_exp_bits + weight_mant_bits + 1 == weight_bit_width, "Weight exponent and mantissa bits must sum to weight bit width"
        assert act_exp_bits + act_mant_bits + 1 == act_bit_width, "Activation exponent and mantissa bits must sum to activation bit width"
        
        return weight_bit_width, weight_exp_bits, weight_mant_bits, act_bit_width, act_exp_bits, act_mant_bits

    quant_type = cfg.get('QUANT', 'TYPE', fallback='FIXED')
    num_classes = cfg.getint('MODEL', 'NUM_CLASSES')
    kwargs = {
        'block_impl':QuantBasicBlock,
        'num_blocks':[2, 2, 2, 2],
        'quant_type': quant_type,
    }
    kwargs['num_classes'] = num_classes
    if quant_type == 'FLOAT':
        weight_bit_width, weight_exp_bits, weight_mant_bits, act_bit_width, act_exp_bits, act_mant_bits = get_float_params(cfg)
        weight_quant_class, act_quant_class = float_weight_act_class_factory(
            weight_bit_width, act_bit_width, act_exp_bits, act_mant_bits, weight_exp_bits, weight_mant_bits
        )
        kwargs['weight_bit_width'] = weight_bit_width
        kwargs['act_bit_width'] = act_bit_width
        kwargs['weight_quant'] = weight_quant_class
        kwargs['act_quant'] = act_quant_class
        kwargs['first_layer_weight_quant'] = weight_quant_class
        kwargs['last_layer_weight_quant'] = weight_quant_class
        kwargs['last_layer_bias_quant'] = None
    elif quant_type == 'FIXED':
        # use the defaults, do not write the parameters
        kwargs['weight_bit_width'] = cfg.getint('QUANT', 'WEIGHT_BIT_WIDTH')
        kwargs['act_bit_width'] = cfg.getint('QUANT', 'ACT_BIT_WIDTH')
    else:
        raise ValueError('No valid QUANT value defined in the config')
    
    return kwargs


def quant_resnet18(cfg) -> QuantResNet:
    kwargs = get_params_from_config(cfg)
    model = QuantResNet(**kwargs)
    return model