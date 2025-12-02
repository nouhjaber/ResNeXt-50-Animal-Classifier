import torch
import torch.nn as nn
import logging
import os
from collections import OrderedDict


# ============================================================================
# DEVICE MANAGEMENT
# ============================================================================

def get_device(verbose: bool = True) -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        if verbose:
            print(f"[INFO] Using GPU: {torch.cuda.get_device_name(0)}")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device = torch.device("mps")
        if verbose:
            print("[INFO] Using Apple MPS GPU")
    else:
        device = torch.device("cpu")
        if verbose:
            print("[INFO] Using CPU")

    return device


def clear_gpu_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ============================================================================
# MODEL SIZE AND PARAMETER COUNTING
# ============================================================================

def get_model_size(model, readable: bool = True):
    total_params = sum(p.numel() for p in model.parameters())

    if not readable:
        return total_params
    
    if total_params >= 1_000_000:
        return f"{total_params / 1_000_000:.2f}M"
    elif total_params >= 1_000:
        return f"{total_params / 1_000:.2f}K"
    else:
        return str(total_params)


def count_parameters(model, readable: bool = True):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total = trainable + frozen

    if not readable:
        return {
            "trainable": trainable,
            "frozen": frozen,
            "total": total
        }
    
    def fmt(n):
        if n >= 1_000_000:
            return f"{n / 1_000_000:.2f}M"
        elif n >= 1_000:
            return f"{n / 1_000:.2f}K"
        return str(n)

    return {
        "trainable": fmt(trainable),
        "frozen": fmt(frozen),
        "total": fmt(total),
    }


# ============================================================================
# MODEL SUMMARY AND ARCHITECTURE VISUALIZATION
# ============================================================================

def print_model_summary(model: nn.Module, input_size=(1, 3, 224, 224), device=None):

    was_training = model.training
    model.eval()

    if device is None:
        device = get_device(verbose=False)

    # Prepare dummy input
    dummy_input = torch.zeros(input_size).to(device)

    summary = OrderedDict()

    def register_hook(module):
        def hook(module, inp, out):
            class_name = module.__class__.__name__
            m_key = f"{class_name}-{len(summary)+1}"

            # Input shape
            if isinstance(inp, (list, tuple)):
                input_shape = list(inp[0].size())
            else:
                input_shape = list(inp.size())

            # Output shape
            if isinstance(out, (list, tuple)):
                output_shape = list(out[0].size())
            else:
                output_shape = list(out.size())

            params = sum(p.numel() for p in module.parameters())
            trainable = any(p.requires_grad for p in module.parameters())

            summary[m_key] = {
                "input": input_shape,
                "output": output_shape,
                "params": params,
                "trainable": trainable,
            }

        # Skip containers (only hook leaf modules)
        if not list(module.children()):
            module.register_forward_hook(hook)

    # Register hooks
    model.apply(register_hook)

    # Forward pass
    with torch.no_grad():
        model(dummy_input)

    # Restore training mode
    model.train(was_training)

    # Print summary
    print("-" * 100)
    print(f"{'Layer (type)':<30} {'Input shape':<20} {'Output shape':<20} {'Params':<12} {'Trainable'}")
    print("-" * 100)

    for layer, info in summary.items():
        trainable_str = "✓" if info['trainable'] else "✗"
        print(f"{layer:<30} "
              f"{str(info['input']):<20} "
              f"{str(info['output']):<20} "
              f"{info['params']:<12} "
              f"{trainable_str}")

    print("-" * 100)
    total_params = sum(info["params"] for info in summary.values())
    trainable_params = sum(info["params"] for info in summary.values() if info["trainable"])
    frozen_params = total_params - trainable_params
    
    print(f"Total params:     {total_params:,}")
    print(f"Trainable:        {trainable_params:,}")
    print(f"Frozen:           {frozen_params:,}")
    print("-" * 100)


# ============================================================================
# WEIGHT INITIALIZATION
# ============================================================================

def init_weights_he(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_fc_layer(in_features=2048, out_features=50):
    layer = nn.Linear(in_features, out_features)
    init_weights_xavier(layer)
    return layer


# ============================================================================
# LAYER FREEZING/UNFREEZING (Transfer Learning)
# ============================================================================

def freeze_layers(model):
    for name, param in model.named_parameters():
        if ("layer4" not in name) and ("fc" not in name):
            param.requires_grad = False


def unfreeze_layer4(model):
    for name, param in model.named_parameters():
        if "layer4" in name:
            param.requires_grad = True


def freeze_batch_norm(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.eval()
            for param in module.parameters():
                param.requires_grad = False


def get_trainable_params(model):
    return [n for n, p in model.named_parameters() if p.requires_grad]


def get_frozen_params(model):
    return [n for n, p in model.named_parameters() if not p.requires_grad]


# ============================================================================
# LOGGING
# ============================================================================

def setup_logging(log_file="train.log", log_level=logging.INFO):
    # Create logs directory if needed
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)
    
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger()


# ============================================================================
# CHECKPOINTING
# ============================================================================

def save_checkpoint(model, optimizer, epoch, metrics, filename="checkpoint.pth"):
    checkpoint = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "metrics": metrics
    }
    
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    torch.save(checkpoint, filename)


def load_checkpoint(model, optimizer=None, filename="checkpoint.pth"):
    ckpt = torch.load(filename, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])

    if optimizer:
        optimizer.load_state_dict(ckpt["optimizer_state"])

    return ckpt["epoch"], ckpt.get("metrics", {})


# ============================================================================
# MEMORY AND PERFORMANCE
# ============================================================================

def estimate_memory_usage(model, batch_size=8, input_size=(3, 224, 224)):
    params_mb = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    activations_mb = batch_size * torch.zeros(input_size).numel() * 4 / 1e6
    # 3x for forward + backward + optimizer state
    total_mb = params_mb + activations_mb * 3
    return round(total_mb, 2)


def get_model_flops(model, input_size=(1, 3, 224, 224)):
    try:
        from thop import profile
        dummy = torch.randn(*input_size)
        flops, params = profile(model, inputs=(dummy,), verbose=False)
        return flops
    except ImportError:
        print("[WARNING] Install 'thop' for FLOPs calculation: pip install thop")
        return None


# ============================================================================
# LEARNING RATE MANAGEMENT
# ============================================================================

def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_learning_rates_by_layer(optimizer, lr_dict):
    for param_group in optimizer.param_groups:
        param_names = param_group['params']
        for layer_name, lr in lr_dict.items():
            if layer_name in str(param_names):
                param_group['lr'] = lr
                break