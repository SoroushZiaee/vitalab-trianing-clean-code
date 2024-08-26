import torch
import torch.nn.functional as F
import numpy as np


class BlurPoolConv2d(torch.nn.Module):

    # Purpose: This class creates a convolutional layer that first applies a blurring filter to the input before performing the convolution operation.
    # Condition: The function apply_blurpool iterates over all layers of the model and replaces convolution layers (ch.nn.Conv2d) with BlurPoolConv2d if they have a stride greater than 1 and at least 16 input channels.
    # Preventing Aliasing: Blurring the output of convolution layers (especially those with strides greater than 1) helps to reduce aliasing effects. Aliasing occurs when high-frequency signals are sampled too sparsely, leading to incorrect representations.
    # Smooth Transitions: Applying a blur before downsampling ensures that transitions between pixels are smooth, preserving important information in the feature maps.
    # Stabilizing Training: Blurring can help stabilize training by reducing high-frequency noise, making the model less sensitive to small changes in the input data.
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (
            np.max(child.stride) > 1 and child.in_channels >= 16
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)
