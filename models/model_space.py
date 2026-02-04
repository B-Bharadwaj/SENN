import random
import torch
import torch.nn as nn

ACTS = ["relu", "gelu", "leaky_relu"]
POOLS = ["none", "max", "avg"]

def act_layer(name: str):
    if name == "relu": return nn.ReLU(inplace=True)
    if name == "gelu": return nn.GELU()
    if name == "leaky_relu": return nn.LeakyReLU(0.1, inplace=True)
    return nn.ReLU(inplace=True)

def random_genome():
    num_blocks = random.randint(2, 4)
    blocks = []
    out_choices = [16, 32, 48, 64, 96, 128]
    k_choices = [3, 5]
    for _ in range(num_blocks):
        blocks.append({
            "out": random.choice(out_choices),
            "k": random.choice(k_choices),
            "act": random.choice(ACTS),
            "bn": random.choice([True, False]),
            "drop": random.choice([0.0, 0.1, 0.2, 0.3]),
            "pool": random.choice(POOLS),
        })
    genome = {
        "blocks": blocks,
        "head": {"dense": random.choice([0, 64, 128, 256]), "drop": random.choice([0.0, 0.2, 0.3, 0.5])},
        "opt": {"lr": 10 ** random.uniform(-4, -2.5), "weight_decay": random.choice([0.0, 1e-4, 5e-4])}
    }
    return genome

class GenomeCNN(nn.Module):
    def __init__(self, genome, num_classes=10):
        super().__init__()
        in_ch = 3
        layers = []
        for b in genome["blocks"]:
            layers.append(nn.Conv2d(in_ch, b["out"], kernel_size=b["k"], padding=b["k"]//2, bias=not b["bn"]))
            if b["bn"]:
                layers.append(nn.BatchNorm2d(b["out"]))
            layers.append(act_layer(b["act"]))
            if b["drop"] > 0:
                layers.append(nn.Dropout2d(p=b["drop"]))
            if b["pool"] == "max":
                layers.append(nn.MaxPool2d(2))
            elif b["pool"] == "avg":
                layers.append(nn.AvgPool2d(2))
            in_ch = b["out"]

        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        head_layers = []
        dense = genome["head"]["dense"]
        head_drop = genome["head"]["drop"]
        if dense and dense > 0:
            head_layers += [nn.Linear(in_ch, dense), nn.ReLU(inplace=True)]
            if head_drop > 0: head_layers.append(nn.Dropout(p=head_drop))
            head_layers.append(nn.Linear(dense, num_classes))
        else:
            if head_drop > 0: head_layers.append(nn.Dropout(p=head_drop))
            head_layers.append(nn.Linear(in_ch, num_classes))

        self.head = nn.Sequential(*head_layers)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return self.head(x)

def build_model(genome, num_classes=10):
    return GenomeCNN(genome, num_classes=num_classes)

def count_params(model) -> int:
    """Calculate total trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_size):
    """Calculate FLOPs for the model"""
    total_flops = 0

    def conv_flops(input_size, kernel_size, in_channels, out_channels, stride=1, padding=0):
        # Calculate output size for Conv2D layer manually (considering stride and padding)
        output_height = (input_size[1] + 2 * padding - kernel_size) // stride + 1
        output_width = (input_size[2] + 2 * padding - kernel_size) // stride + 1
        return 2 * (out_channels * kernel_size * kernel_size * in_channels * output_height * output_width)

    def fc_flops(input, output):
        # FLOPs for Fully Connected Layers
        return 2 * (input * output)

    # Iterate over layers
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            # Consider stride and padding for Conv2D layers
            total_flops += conv_flops(input_size, layer.kernel_size[0], layer.in_channels, layer.out_channels, stride=layer.stride[0], padding=layer.padding[0])
            # Update input size for next layer (accounting for stride and padding)
            input_size = (layer.out_channels, input_size[1] // layer.stride[0], input_size[2] // layer.stride[0])  # assuming square kernels and square strides
        elif isinstance(layer, nn.Linear):
            total_flops += fc_flops(layer.in_features, layer.out_features)

    return total_flops