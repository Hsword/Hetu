from megatron.initialize import initialize_megatron
from megatron import get_args as get_megatron_args
import argparse

def initialize_galvatron(model_args = None, mode="train_dist"):
    use_megatron = False
    if mode in ["train_dist", "train"]:
        use_megatron = (mode == "train_dist")
        extra_args_provider = [lambda parser: galvatron_training_args(parser, use_megatron)]
    if model_args is not None:
        extra_args_provider.append(model_args)
    if use_megatron:
        initialize_megatron(extra_args_provider)
        args = get_args()
    else:
        args = parse_args(extra_args_provider)
    if 'allow_tf32' in args and args.allow_tf32:
        import torch
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
    return args

def parse_args(extra_args_provider):
    parser = argparse.ArgumentParser()
    # Custom arguments.
    if extra_args_provider is not None:
        if isinstance(extra_args_provider, list):
            for extra_args in extra_args_provider:
                parser = extra_args(parser)
        else:
            parser = extra_args_provider(parser)
    args = parser.parse_args()
    return args

def get_args():
    return get_megatron_args()

def galvatron_training_args(parser, use_megatron=True):
    group = parser.add_argument_group(title="Galvatron Training Arguments")

    group.add_argument(
        "--set_model_config_manually", type=int, default=0, help="Whether to set model config manually. If set to 1, model config set by 'model_size' will be overwritten."
    )
    group.add_argument(
        "--set_layernum_manually", type=int, default=0, help="Whether to set layernum config manually (doesn't overwrite other model configs)."
    )
    group.add_argument(
        "--initialize_on_meta", type=int, default=0, help="Whether to initialize parameters on meta device.", choices=[0, 1]
    )
    group.add_argument(
        "--global_train_batch_size", type=int, default=32, help="Global training batch size"
    )
    group.add_argument(
        "--dropout_prob", type=float, default=0.1, help="Dropout rate."
    )
    group.add_argument("-e", "--epochs", type=int,
                        default=10, help="Number of epochs")
    group.add_argument(
        "--adam_weight_decay", type=float, default=0.01, help="Weight_decay of adam"
    )
    group.add_argument(
        "--check_loss", type=int, default=0, help="Whether to check model correctness."
    )
    group.add_argument(
        "--profile", type=int, default=0, help="Whether to profile model GPU memory."
    )
    group.add_argument(
        "--save_profiled_memory", type=int, default=0, help="Whether to save profiled memory."
    )
    group.add_argument(
        "--profile_type", type=str, default="allocated", help="Profile allocated memory or reserved memory.",
        choices = ["allocated", "reserved"],
    )
    group.add_argument(
        "--load_params", type=int, default=0, help="Whether to load saved init params."
    )
    group.add_argument(
        "--pp_deg", type=int, default=2, help="Pipeline parallel degree.", choices=[1,2,4,8,16,32,64,128,256,512],
    )
    group.add_argument(
        "--global_tp_deg", type=int, default=-1, help="Global tensor parallel degree.", choices=[-1,1,2,4,8,16,32],
    )
    group.add_argument(
        "--chunks", type=int, default=-1, help="Pipeline chunk num.",
    )
    group.add_argument(
        "--global_tp_consec", type=int, default=-1, help="Global tensor parallel group consecutive flag."
    )
    group.add_argument(
        "--sdp", type=int, default=0, help="Apply SDP (zero-3)", choices=[0, 1],
    )
    group.add_argument(
        "--galvatron_config_path", type=str, default=None, help="Galvatron strategy config path. If not None, galvatron will run according to json config file.",
    )
    group.add_argument(
        "--global_checkpoint", type=int, default=0, help="Global checkpoint flag."
    )
    group.add_argument(
        "--mixed_precision", type=str, default="bf16", help="Mixed precision option.", choices=["fp32", "fp16", "bf16"],
    )
    group.add_argument(
        "--pipeline_type", type=str, default="gpipe", help="Galvatron pipeline type", choices=["gpipe","pipedream_flush"],
    )
    group.add_argument(
        "--default_dp_type", type=str, default="ddp", help="Default data parallel type", choices=["ddp","zero2","zero3"],
    )
    group.add_argument(
        "--embed_sdp", type=int, default=0, help="Apply SDP (zero-3) for Embeddings and cls", choices=[0, 1],
    )
    group.add_argument(
        "--profile_forward", type=int, default=0, help="Profile forward computation", choices=[0, 1],
    )
    group.add_argument(
        "--allow_tf32", type=int, default=1, help="Whether to allow tf32 on Ampere devices", choices=[0, 1],
    )
    group.add_argument(
        "--exit_after_profiling", type=int, default=1, help="Whether to exit after profiling time and memory.", choices=[0, 1],
    )
    if not use_megatron:
        group.add_argument("--lr", type=float, default=1e-4, help="Learning rate of adam")
        group.add_argument("--gpu_id", type=int, default=0, help="Id of GPU to run.")
        group.add_argument("--local-rank", type=int, default=0, help="Local rank.")
    else:
        group.add_argument("--local-rank", type=int, default=-1, help="Local rank.")
    return parser