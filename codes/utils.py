

def set_device(args):
    if args.cpu is not None:
        if args.gpu is not None:
            raise ArgumentError("Can't use CPU and GPU at the same time")
    elif args.gpu is None:
        args.gpu = os.environ["CUDA_VISIBLE_DEVICES"]