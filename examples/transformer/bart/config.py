'''
Bart Config:
--------------------------------------------------------------------------------------------------'''
class BartConfig(object):
    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="relu",
        d_model=1024,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        use_return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding 
  
        self.num_labels=num_labels
        self.pad_token_id=pad_token_id
        self.bos_token_id=bos_token_id
        self.eos_token_id=eos_token_id
        self.is_encoder_decoder=is_encoder_decoder
        self.decoder_start_token_id=decoder_start_token_id
        self.forced_eos_token_id=forced_eos_token_id

        self.use_return_dict = use_return_dict
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states




'''-----------------------------------------------------------------------------------------------'''
