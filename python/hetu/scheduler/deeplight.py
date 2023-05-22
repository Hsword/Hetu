from .base import EmbeddingTrainer
from .multistage import MultiStageTrainer
from .switchinference import SwitchInferenceTrainer
from ..layers import DeepLightEmbedding
import os.path as osp


class DeepLightOverallTrainer(MultiStageTrainer):
    # in deeplight, parameters and optimizer states are both inherited from warmup

    @property
    def legal_stages(self):
        return (1, 2)

    def fit(self):
        stage = self.stage
        # two stages: warmup, training(pruning)
        if stage == 1:
            self.warmup_trainer = EmbeddingTrainer(
                self.dataset, self.model, self.opt, self.copy_args_with_stage(1), self.data_ops)
        else:
            self.warmup_trainer = None
        self.trainer = DeepLightTrainer(
            self.dataset, self.model, self.opt, self.copy_args_with_stage(2), self.data_ops)
        self.trainer.prepare_path_for_retrain('train')

        if self.warmup_trainer is not None:
            self.warmup_trainer.fit()
            ep, part = self.warmup_trainer.get_best_meta()
            self.trainer.load_ckpt = self.warmup_trainer.join(
                f'ep{ep}_{part}.pkl')
            self.warmup_trainer.executor.return_tensor_values()
            del self.warmup_trainer
        self.trainer.fit()

    def test(self):
        assert self.stage == 2
        trainer = DeepLightTrainer(
            self.dataset, self.model, self.opt, self.args, self.data_ops)
        trainer.test()


class DeepLightTrainer(SwitchInferenceTrainer):
    @property
    def sparse_name(self):
        return 'DeepLightEmb'

    def try_load_ckpt(self):
        meta = super().try_load_ckpt()
        assert meta is not None
        st = meta['state_dict']
        stage = meta['args']['embedding_args']['stage']
        if stage == 1:
            assert self.sparse_name not in st
            st[self.sparse_name] = st.pop('Embedding')
            meta['epoch'] = 0
            meta['part'] = -1
            self.start_ep = 0
            # not use load_only_parameters here, since the optimizer states should be inherited
        return meta

    def get_embed_layer(self):
        return DeepLightEmbedding(
            self.num_embed,
            self.embedding_dim,
            self.prune_rate,
            self.form,
            initializer=self.initializer,
            name=self.sparse_name,
            ctx=self.ectx,
        )

    def get_eval_nodes(self):
        embed_input, dense_input, y_ = self.data_ops
        loss, prediction = self.model(
            self.embed_layer(embed_input), dense_input, y_)
        train_op = self.opt.minimize(loss)
        eval_nodes = {
            self.train_name: [loss, prediction, y_, train_op, self.embed_layer.make_prune_op(y_)],
            self.validate_name: [loss, prediction, y_],
            self.test_name: [loss, prediction, y_],
        }
        return eval_nodes

    def test(self):
        # use filename to distinguish whether test middle ckpt or final ckpt
        test_final = osp.split(self.load_ckpt)[-1].startswith('final')
        if test_final:
            super().test()
        else:
            EmbeddingTrainer.test(self)
