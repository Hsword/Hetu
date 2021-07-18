from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import absolute_import


class PassOp:
    @classmethod
    def version_1(cls, ctx, node, **kwargs):
        pass

    @classmethod
    def version_6(cls, ctx, node, **kwargs):

        cls.version_1(ctx, node, **kwargs)
