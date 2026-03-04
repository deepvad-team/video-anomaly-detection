# demo/demo_tea.py
# 초기 normal buffer feature 만으로 adapter LN을 적응시키는 코드
import torch
import torch.nn.functional as F


def demo_initial_tea_adapt(
    normal_feat_tensor,   # (N,10,2048)
    adapter,
    model,
    tea_steps=2,
    sgld_steps=10,
    sgld_lr=0.05,
    sgld_noise=0.01,
    tea_lr=1e-3,
):
    """
    초기 normal buffer feature만 사용해서
    adapter.ln.weight / bias만 TEA-style로 업데이트
    """
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    adapter.train()
    for p in adapter.parameters():
        p.requires_grad_(False)

    if not hasattr(adapter, "ln"):
        raise ValueError("Adapter must have LayerNorm for LN-only TEA")

    adapter.ln.weight.requires_grad_(True)
    adapter.ln.bias.requires_grad_(True)

    optimizer = torch.optim.Adam(
        [adapter.ln.weight, adapter.ln.bias],
        lr=tea_lr
    )

    x_real = normal_feat_tensor.detach()

    for _ in range(tea_steps):
        # real energy
        x_real_adapt = adapter(x_real)
        _, logit_real = model(x_real_adapt, return_logits=True)
        logit_real = logit_real.squeeze(-1)
        E_real = F.softplus(logit_real).mean()

        # fake samples in feature space
        x_tilde = (x_real + sgld_noise * torch.randn_like(x_real)).detach()

        for _ in range(sgld_steps):
            x_tilde.requires_grad_(True)

            x_tilde_adapt = adapter(x_tilde)
            _, logit_fake_now = model(x_tilde_adapt, return_logits=True)
            logit_fake_now = logit_fake_now.squeeze(-1)
            E_fake_now = F.softplus(logit_fake_now).mean()

            grad = torch.autograd.grad(E_fake_now, x_tilde, create_graph=False)[0]

            with torch.no_grad():
                x_tilde = x_tilde - (sgld_lr / 2.0) * grad + sgld_noise * torch.randn_like(x_tilde)
            x_tilde = x_tilde.detach()

        x_tilde_adapt = adapter(x_tilde)
        _, logit_fake = model(x_tilde_adapt, return_logits=True)
        logit_fake = logit_fake.squeeze(-1)
        E_fake = F.softplus(logit_fake).mean()

        loss = E_real - E_fake

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    adapter.eval()
    return adapter