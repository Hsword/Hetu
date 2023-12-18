
def model_args(parser):
    group = parser.add_argument_group(title='Model Arguments')

    group.add_argument(
        "--model_size", type=str, default='gpt-1.5b', help="Model size.", choices=['gpt-0.3b', 'gpt-1.5b', 'gpt-2.7b', 'gpt-6.7b']
    )
    group.add_argument(
        "--hidden_size", type=int, default=768, help="Hidden size of transformer model",
    )
    group.add_argument(
        "--num_hidden_layers", type=int, default=12, help="Number of layers"
    )
    group.add_argument(
        "-a", "--num_attention_heads", type=int, default=12, help="Number of attention heads",
    )
    group.add_argument(
        "-s", "--seq_length", type=int, default=128, help="Maximum sequence len"
    )
    group.add_argument(
        "--vocab_size", type=int, default=30522, help="Total number of vocab"
    )
    group.add_argument("--max_predictions_per_seq", type=int, default=20)
    return parser

def layernum_arg_names():
    return ['num_hidden_layers']