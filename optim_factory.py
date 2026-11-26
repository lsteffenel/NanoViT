from torch import optim as optim


def create_optimizer(args, model, skip_list=None, get_num_layer=None, get_layer_scale=None,
                     filter_bias_and_bn=True):
    if get_num_layer is not None or get_layer_scale is not None:
        raise NotImplementedError("layer decay is not supported by this optim_factory")
    skip = set(skip_list or [])
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad or name in skip:
            continue
        if filter_bias_and_bn and (p.ndim <= 1 or name.endswith(".bias")):
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": no_decay, "weight_decay": 0.0, "lr_scale": 1.0},
        {"params": decay, "weight_decay": args.weight_decay, "lr_scale": 1.0},
    ]
    betas = tuple(args.opt_betas) if args.opt_betas else (0.9, 0.999)
    opt = args.opt.lower()
    if opt == "adamw":
        return optim.AdamW(groups, lr=args.lr, eps=args.opt_eps, betas=betas)
    if opt == "adam":
        return optim.Adam(groups, lr=args.lr, eps=args.opt_eps, betas=betas)
    if opt == "sgd":
        return optim.SGD(groups, lr=args.lr, momentum=args.momentum, nesterov=True)
    raise ValueError(f"Unsupported optimizer: {args.opt}")
