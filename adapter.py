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