from prettytable import PrettyTable


def format_params(num_params):
    """Convert parameter count to human-readable format (M or B)"""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.2f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.2f}M"
    else:
        return f"{int(num_params)}"


def print_model_params(model):
    """Print model parameters in table format

    Args:
        model: Model instance, automatically iterates through first-level submodules
    """
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_frozen = total_params - total_trainable
    trainable_ratio = (total_trainable / total_params * 100) if total_params > 0 else 0

    table = PrettyTable()
    table.title = "Model Parameters Summary"
    table.field_names = ["Component", "Total", "Trainable", "Frozen", "Trainable %"]

    for name, module in model.named_children():
        module_params = sum(p.numel() for p in module.parameters())
        module_trainable = sum(
            p.numel() for p in module.parameters() if p.requires_grad
        )
        module_frozen = module_params - module_trainable
        module_ratio = (
            (module_trainable / module_params * 100) if module_params > 0 else 0
        )

        table.add_row(
            [
                name,
                format_params(module_params),
                format_params(module_trainable),
                format_params(module_frozen),
                f"{module_ratio:.2f}%",
            ]
        )

    table.add_row(
        [
            "Total",
            format_params(total_params),
            format_params(total_trainable),
            format_params(total_frozen),
            f"{trainable_ratio:.2f}%",
        ]
    )

    table.align["Component"] = "l"
    table.align["Total"] = "r"
    table.align["Trainable"] = "r"
    table.align["Frozen"] = "r"
    table.align["Trainable %"] = "r"

    print(f"\n{table}\n")
