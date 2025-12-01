"""
Compatibility patch for WavTokenizer + PEFT LoRA

PEFT wraps Conv modules, breaking attribute access in WavTokenizer's conv.py
This patch makes attribute access work correctly with wrapped modules.
"""

import torch.nn as nn


def patch_streamable_conv_for_lora():
    """
    Monkey-patch SConv1d and SConvTranspose1d to handle LoRA-wrapped modules
    """
    import sys
    sys.path.insert(0, '/home/sbplab/ruizi/WavTokenizer-main')
    from encoder.modules.conv import SConv1d, SConvTranspose1d

    # Store original forward methods
    _original_sconv1d_forward = SConv1d.forward
    _original_sconvtrans_forward = SConvTranspose1d.forward

    def _get_conv_attr(conv_module, attr_name):
        """
        Safely get attribute from potentially LoRA-wrapped Conv module

        Args:
            conv_module: Could be StreamableConv1d or its .conv attribute
            attr_name: 'kernel_size', 'stride', 'dilation', etc.

        Returns:
            The attribute value
        """
        # Try direct access first (original module)
        if hasattr(conv_module, attr_name):
            return getattr(conv_module, attr_name)

        # If wrapped with PEFT, try accessing base_layer or original_module
        if hasattr(conv_module, 'base_layer'):
            return getattr(conv_module.base_layer, attr_name)
        if hasattr(conv_module, 'original_module'):
            return getattr(conv_module.original_module, attr_name)
        if hasattr(conv_module, 'module'):
            return getattr(conv_module.module, attr_name)

        # Handle NormConv1d (has self.conv) and NormConvTranspose1d (has self.convtr)
        if hasattr(conv_module, 'conv') and hasattr(conv_module.conv, attr_name):
            return getattr(conv_module.conv, attr_name)
        if hasattr(conv_module, 'convtr') and hasattr(conv_module.convtr, attr_name):
            return getattr(conv_module.convtr, attr_name)

        # Fallback: access _parameters or _modules
        if hasattr(conv_module, '_modules') and 'original_module' in conv_module._modules:
            return getattr(conv_module._modules['original_module'], attr_name)

        raise AttributeError(f"Cannot access {attr_name} from {type(conv_module)}")

    def patched_sconv1d_forward(self, x):
        """Patched SConv1d.forward that handles LoRA wrapping"""
        B, C, T = x.shape

        # Original code tries: self.conv.conv.kernel_size[0]
        # But after LoRA wrapping, self.conv might be wrapped
        # We need to access attributes safely

        try:
            # Try accessing conv.conv (nested Conv1d)
            inner_conv = self.conv.conv
        except AttributeError:
            # If wrapped, conv.conv might not exist, use conv directly
            inner_conv = self.conv
            if hasattr(inner_conv, 'base_layer'):
                inner_conv = inner_conv.base_layer

        kernel_size = _get_conv_attr(inner_conv, 'kernel_size')[0]
        stride = _get_conv_attr(inner_conv, 'stride')[0]
        dilation = _get_conv_attr(inner_conv, 'dilation')[0]

        # Rest of the original forward logic
        from encoder.modules.conv import get_extra_padding_for_conv1d, pad1d

        kernel_size = (kernel_size - 1) * dilation + 1  # effective kernel size with dilations
        padding_total = kernel_size - stride
        extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
        if self.causal:
            # Left padding for causal
            x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
        else:
            # Asymmetric padding required for odd strides
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
        return self.conv(x)

    def patched_sconvtrans_forward(self, x):
        """Patched SConvTranspose1d.forward that handles LoRA wrapping"""
        # SConvTranspose1d 使用 convtr 而不是 conv
        conv_attr = self.convtr if hasattr(self, 'convtr') else self.conv
        try:
            inner_conv = conv_attr.conv
        except AttributeError:
            inner_conv = conv_attr
            if hasattr(inner_conv, 'base_layer'):
                inner_conv = inner_conv.base_layer

        kernel_size = _get_conv_attr(inner_conv, 'kernel_size')[0]
        stride = _get_conv_attr(inner_conv, 'stride')[0]
        padding_total = kernel_size - stride

        y = conv_attr(x)

        # Remove padding
        from encoder.modules.conv import unpad1d
        if self.causal:
            # Remove left padding
            y = unpad1d(y, (padding_total, 0))
        else:
            # Remove asymmetric padding
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))

        return y

    # Apply patches
    SConv1d.forward = patched_sconv1d_forward
    SConvTranspose1d.forward = patched_sconvtrans_forward

    print("✓ Applied WavTokenizer-LoRA compatibility patch")


def apply_lora_patch():
    """Apply all necessary patches"""
    patch_streamable_conv_for_lora()
