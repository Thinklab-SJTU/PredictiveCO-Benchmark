import torch
import torch.nn.functional as F


class SoftTopk(torch.nn.Module):
    """
    Code from:
    """

    def __init__(self, epsilon=0.01, max_iter=200):
        super(SoftTopk, self).__init__()
        self.epsilon = epsilon
        self.anchors = torch.FloatTensor([0, 1])
        self.max_iter = max_iter

    def forward(self, scores, k):
        device = scores.device
        #
        n = scores.shape[-1]
        bs = scores.shape[:-1]
        scores = scores.view([*bs, n, 1])

        anchors = self.anchors.view(*((1,) * (len(bs) + 1)), 2)
        C_raw = (scores - anchors) ** 2
        C = C_raw / C_raw.amax(dim=(-2, -1), keepdim=True)

        mu = torch.ones([*bs, n, 1], requires_grad=False).to(device) / n
        nu = (
            torch.FloatTensor([k / n, (n - k) / n])
            .view((*((1,) * (len(bs) + 1)), 2))
            .to(device)
        )
        Gamma = TopKFunc.apply(C, mu, nu, self.epsilon, self.max_iter, device)
        return Gamma


class TopKFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, C, mu, nu, epsilon, max_iter, device):
        bs, n, k_ = C.shape[:-2], C.shape[-2], C.shape[-1]

        f = torch.zeros([*bs, n, 1]).to(device)
        g = torch.zeros([*bs, 1, k_]).to(device)

        epsilon_log_mu = epsilon * torch.log(mu)
        epsilon_log_nu = epsilon * torch.log(nu)

        def min_epsilon_row(Z, epsilon):
            return -epsilon * torch.logsumexp((-Z) / epsilon, -1, keepdim=True)

        def min_epsilon_col(Z, epsilon):
            return -epsilon * torch.logsumexp((-Z) / epsilon, -2, keepdim=True)

        for _ in range(max_iter):
            f = min_epsilon_row(C - g, epsilon) + epsilon_log_mu
            g = min_epsilon_col(C - f, epsilon) + epsilon_log_nu

        Gamma = torch.exp((-C + f + g) / epsilon)
        ctx.save_for_backward(mu, nu, Gamma)
        ctx.epsilon = epsilon
        return Gamma

    @staticmethod
    def backward(ctx, grad_output_Gamma):
        epsilon = ctx.epsilon
        mu, nu, Gamma = ctx.saved_tensors
        # mu [1, n, 1]
        # nu [1, 1, k+1]
        # Gamma [*bs, n, k+1]

        with torch.no_grad():
            nu_ = nu[..., :-1]
            Gamma_ = Gamma[..., :-1]

            bs, n, k_ = Gamma.shape[:-2], Gamma.shape[-2], Gamma.shape[-1]

            inv_mu = 1.0 / (mu.view([1, -1]))  # [1, n]
            Kappa = torch.diag_embed(nu_.squeeze(-2)) - torch.matmul(
                Gamma_.transpose(-1, -2) * inv_mu.unsqueeze(-2), Gamma_
            )  # [*bs, k, k]
            # print(Kappa, Gamma_)
            padding_value = 1e-10
            ridge = torch.ones([*bs, k_ - 1]).diag_embed()
            inv_Kappa = torch.inverse(Kappa + ridge * padding_value)  # [*bs, k, k]
            # print(Kappa, inv_Kappa)
            mu_Gamma_Kappa = (inv_mu.unsqueeze(-1) * Gamma_).matmul(
                inv_Kappa
            )  # [*bs, n, k]
            H1 = inv_mu.diag_embed() + mu_Gamma_Kappa.matmul(
                Gamma_.transpose(-1, -2)
            ) * inv_mu.unsqueeze(
                -2
            )  # [*bs, n, n]
            H2 = -mu_Gamma_Kappa  # [*bs, n, k]
            H3 = H2.transpose(-1, -2)  # [*bs, k, n]
            H4 = inv_Kappa  # [*bs, k, k]

            H2_pad = F.pad(H2, pad=(0, 1), mode="constant", value=0)
            H4_pad = F.pad(H4, pad=(0, 1), mode="constant", value=0)
            grad_f_C = H1.unsqueeze(-1) * Gamma.unsqueeze(-3) + H2_pad.unsqueeze(
                -2
            ) * Gamma.unsqueeze(
                -3
            )  # [*bs, n, n, k+1]
            grad_g_C = H3.unsqueeze(-1) * Gamma.unsqueeze(-3) + H4_pad.unsqueeze(
                -2
            ) * Gamma.unsqueeze(
                -3
            )  # [*bs, k, n, k+1]

            grad_g_C_pad = F.pad(
                grad_g_C, pad=(0, 0, 0, 0, 0, 1), mode="constant", value=0
            )
            grad_C1 = grad_output_Gamma * Gamma
            grad_C2 = torch.sum(
                grad_C1.view([*bs, n, k_, 1, 1]) * grad_f_C.unsqueeze(-3), dim=(1, 2)
            )
            grad_C3 = torch.sum(
                grad_C1.view([*bs, n, k_, 1, 1]) * grad_g_C_pad.unsqueeze(-4), dim=(1, 2)
            )

            grad_C = (-grad_C1 + grad_C2 + grad_C3) / epsilon

        return grad_C, None, None, None, None, None
