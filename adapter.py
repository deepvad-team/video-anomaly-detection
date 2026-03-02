import torch
import torch.nn as nn

class CopyPlusExtraAdapter(nn.Module):
    def __init__(self, d=1024, use_ln=True):
        super().__init__()
        self.extra = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.extra.weight)   # 처음엔 extra(x)=0

        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(2 * d)

    def forward(self, x):
        """
        x: (..., 1024)
        return: (..., 2048)
        """
        y = torch.cat([x, self.extra(x)], dim=-1)   # (..., 2048)
        if self.use_ln:
            y = self.ln(y)
        return y

#직접 만든 extractor (2048 로 나옴) 로 돌릴 경우 TEA를 위한 layernorm 은 필요하니깐
class ResidualAdapter2048(nn.Module):
    def __init__(self, d=2048, use_ln=True):
        super().__init__()
        self.delta = nn.Linear(d, d, bias=False)
        nn.init.zeros_(self.delta.weight)

        self.use_ln = use_ln
        if use_ln:
            self.ln = nn.LayerNorm(d)

    def forward(self, x):
        y = x + self.delta(x)   # residual
        if self.use_ln:
            y = self.ln(y)
        return y