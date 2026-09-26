"""通道 A: 浅层结构嵌入与降级阶梯。

Example:
    >>> from heteroforge.embed import ChannelAEmbedder, available_backends
    >>> sorted(available_backends().keys())
    ['node2vec', 'spectral_rw', 'svd']
"""

from __future__ import annotations

from heteroforge.embed.channel_a import (
    BACKEND_LICENSES,
    LADDER,
    ChannelAEmbedder,
    available_backends,
)

__all__ = ["ChannelAEmbedder", "available_backends", "LADDER", "BACKEND_LICENSES"]
