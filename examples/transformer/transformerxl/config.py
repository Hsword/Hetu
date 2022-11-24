'''
Transformer XL configuration:
--------------------------------------------------------------------------------------------------'''

class TransfoXLConfig(object):
    def __init__(
        self,
        vocab_size=267735,
        cutoffs=[20000, 40000, 200000],
        d_model=1024,
        d_embed=1024,
        n_head=16,
        d_head=64,
        d_inner=4096,
        div_val=1,
        pre_lnorm=False,
        n_layer=18,
        mem_len=1600,
        clamp_len=1000,
        same_length=True,
        proj_share_all_but_first=True,
        attn_type=0,
        sample_softmax=-1,
        adaptive=True,
        dropout=0.0,
        dropatt=0.0,
        untie_r=True,
        init="normal",
        init_range=0.01,
        proj_init_std=0.01,
        init_std=0.02,
        layer_norm_epsilon=1e-5,
        eos_token_id=0,
        output_attentions=False,
        output_hidden_states=False,
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.cutoffs = []
        self.cutoffs.extend(cutoffs)
        if proj_share_all_but_first:
            self.tie_projs = [False] + [True] * len(self.cutoffs)
        else:
            self.tie_projs = [False] + [False] * len(self.cutoffs)
        self.d_model = d_model
        self.d_embed = d_embed
        self.d_head = d_head
        self.d_inner = d_inner
        self.div_val = div_val
        self.pre_lnorm = pre_lnorm
        self.n_layer = n_layer
        self.n_head = n_head
        self.mem_len = mem_len
        self.same_length = same_length
        self.attn_type = attn_type
        self.clamp_len = clamp_len
        self.sample_softmax = sample_softmax
        self.adaptive = adaptive
        self.dropout = dropout
        self.dropatt = dropatt
        self.untie_r = untie_r
        self.init = init
        self.init_range = init_range
        self.proj_init_std = proj_init_std
        self.init_std = init_std
        self.layer_norm_epsilon = layer_norm_epsilon

        self.eos_token_id = eos_token_id
        self.output_attentions = output_attentions
        self.output_hidden_states = output_hidden_states


'''-----------------------------------------------------------------------------------------------'''