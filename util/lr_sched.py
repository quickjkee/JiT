import math


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup."""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        if args.lr_schedule == "constant":
            lr = args.lr
        elif args.lr_schedule == "cosine":
            lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
                1.0 + math.cos(
                    math.pi * (epoch - args.warmup_epochs)
                    / (args.epochs - args.warmup_epochs)
                )
            )
        else:
            raise NotImplementedError

    for param_group in optimizer.param_groups:
        # Keep classifier/LSEP head LR constant.
        if param_group.get("constant_lr", False):
            param_group["lr"] = param_group["base_lr"]
        else:
            param_group["lr"] = lr * param_group.get("lr_scale", 1.0)

    return lr