import numpy as np
import torch


class OptimiseSubmodular(torch.autograd.Function):
    """ """

    @staticmethod
    def forward(
        ctx,
        Yhat,  # predicted labels
        get_obj,  # A function that returns the value of the objective we want to minimise
        budget,  # The maximum number of items we can select (in expectation)
        lr,  # learning rate for optimiser
        momentum,  # momentum for optimiser
        num_iters,  # number of optimisation steps
        verbose,  # print intermediate solution statistics
        Z_init,  # value with which to warm start Z
    ):
        """
        Run some variant of SGD for the coverage problem with given
        coverage probabilities Yhat
        """
        # Decision variables
        Z = (
            Z_init.detach().clone()
            if Z_init is not None
            else torch.rand(Yhat.shape[0]).detach().to(Yhat.device)
        )
        # Set up the optimizer
        Zprev = Z.clone().detach()

        # Take projected stochastic gradient steps
        for t in range(num_iters):
            # Find gradient
            with torch.enable_grad():
                Znew = (Z + momentum * (Z - Zprev)).requires_grad_(
                    True
                )  # using accelerated PGD
                loss = -get_obj(Yhat.detach(), Znew)
                Znew.retain_grad()
                loss.backward()

            # Update estimate
            Zprev = Z
            Z = (Znew - lr * Znew.grad).detach()
            Z.data = (
                torch.from_numpy(
                    OptimiseSubmodular._project(Z.data.cpu().numpy(), budget)
                )
                .float()
                .to(Yhat.device)
            )

            # Benchmark performance
            if verbose:
                with torch.no_grad():
                    print(t + 1, get_obj(Yhat, Z).item())

        # Save things that are relevant to the backward pass
        ctx.save_for_backward(Yhat, Z)
        ctx.get_obj = get_obj
        ctx.budget = budget
        # print("Z", Z)
        return Z

    # TODO: Make this operate on torch tensors?
    @staticmethod
    def _project(Z, k, c=1.0):
        """
        Projects Z onto the set {Z': 0 <= Z' <= 1/c, ||Z'||_1 = k}
        Uses the projection algorithm of Karimi et al., 2017
        (Algorithm 2: https://arxiv.org/pdf/1711.01566.pdf)

        (More readable but less efficient version of the implementation by Wilder et. al.)
        """
        # Sanity checks
        assert isinstance(Z, np.ndarray) and Z.ndim == 1
        assert isinstance(c, float)
        k *= c  # Scale budget by maximum value

        # Get all possible values of alpha
        alphas_upper = Z / c
        alphas_lower = (Z * c - 1) / c**2
        alphas = np.append(alphas_lower, alphas_upper)
        alphas = np.unique(alphas)  # also sorts elements

        # Find the right value of \alpha that satisfies h(\alpha) = k
        h = len(Z)
        for i in range(1, len(alphas)):
            hprime = np.clip(Z - alphas[i] * c, 0, 1.0 / c).sum()
            if hprime < k:
                alphastar = alphas[i - 1] + (alphas[i] - alphas[i - 1]) * (h - k) / (
                    h - hprime
                )
                result = np.clip(Z - alphastar * c, 0, 1.0 / c)
                if not np.isclose(result.sum(), k, atol=1e-2):
                    print(
                        f"Warning: Total allocated items {result.sum()} greater than budget {k}"
                    )
                return result
            h = hprime
        raise Exception("Projection did not terminate")

    @staticmethod
    def backward(ctx, grad_output):
        """
        Differentiates the optimal Z returned by the forward pass with respect
        to the ratings matrix that was given as input.
        """
        # Read saved tensor from forward pass
        Yhat, Z = ctx.saved_tensors
        get_obj = ctx.get_obj
        budget = ctx.budget

        # Compute the derivative of decision Z wrt the input predictions Yhat
        dZdYhat = OptimiseSubmodular._get_dZdYhat(Z, Yhat, get_obj, budget)

        # Apply chain rule
        dZdYhat_t = dZdYhat.t()
        out = torch.mm(dZdYhat_t.float(), grad_output.view(len(Z), 1))
        return (
            out.view_as(Yhat),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )  # Only Yhat gets a gradient

    @staticmethod
    def _get_elementwise_derivative(first_derivative, variable):
        second_derivative = []
        for element in first_derivative:
            second_derivative.append(
                torch.autograd.grad(element, variable, retain_graph=True)[0].unsqueeze(0)
            )
        return torch.vstack(second_derivative)

    @staticmethod
    def _get_dZdYhat(
        Z,
        Yhat,
        get_obj,
        budget,
        EPS=1e-6,
        alpha=0.01,  # constant to be added to the diagonal of the A matrix to improve conditioning
    ):
        """
        Returns the derivative of the optimal solution in the region around Z in
        terms of the predicted labels/rating matrix Yhat.

        Z: an optimal solution

        Yhat: the current parameter settings
        """
        # Define useful constants
        num_var = len(Z)
        num_constr = 2 * num_var + 1

        # First, find the first and second order derivatives using autograd
        Yhat_local = Yhat.detach().requires_grad_(True)
        Z_local = Z.detach().requires_grad_(True)
        with torch.enable_grad():
            # Objective
            f = get_obj(Yhat_local, Z_local)
            # Gradient
            dfdZ = torch.autograd.grad(f, Z_local, create_graph=True)[0].to(Z.device)
            # Hessian
            dfdZ_dZ = OptimiseSubmodular._get_elementwise_derivative(dfdZ, Z_local)
            # Cross-Term
            dfdZ_dYhat = OptimiseSubmodular._get_elementwise_derivative(dfdZ, Yhat_local)

        # Second, get the optimal dual variables via the KKT conditions
        #   dual variable for constraint sum(Z) <= k
        if torch.logical_and(Z > EPS, Z < 1 - EPS).any():
            lambda_sum = torch.mean(dfdZ[torch.logical_and(Z > EPS, Z < 1 - EPS)])
        else:
            lambda_sum = torch.tensor(0).float().to(Z.device)
        #   dual variable for constraint Z <= 1
        lambda_upper = torch.where(
            Z > 1 - EPS, dfdZ - lambda_sum, torch.zeros_like(Z, device=Z.device)
        )
        #   dual variable for constraint Z >= 0
        lambda_lower = torch.where(
            Z < EPS, dfdZ - lambda_sum, torch.zeros_like(Z, device=Z.device)
        )
        #   collect value of dual variables
        lam = torch.hstack((lambda_sum, lambda_upper, lambda_lower))
        diag_lambda = torch.diag(lam)

        # Third, collect value of constraints g(z) <= 0
        g_sum = Z_local.sum() - budget
        g_upper = Z_local - 1
        g_lower = -Z_local
        g = torch.hstack((g_sum, g_upper, g_lower))
        diag_g = torch.diag(g)

        # Fourth, get gradient of constraints wrt Z
        #   gradient of constraint sum(Z) <= k
        dgdZ_sum = torch.ones(num_var)
        #   gradient of constraints Z <= 1
        dgdZ_upper = torch.eye(num_var)
        #   gradient of constraints Z >= 0 <--> -Z <= 0
        dgdZ_lower = -torch.eye(num_var)
        dgdZ = torch.vstack((dgdZ_sum, dgdZ_upper, dgdZ_lower)).to(Z.device)

        # Putting it together, coefficient matrix for the linear system
        A = torch.vstack(
            [
                torch.hstack([dfdZ_dZ, dgdZ.t()]),
                torch.hstack([diag_lambda @ dgdZ, diag_g]),
            ]
        )
        # add alpha * I to improve conditioning
        A = A + alpha * torch.eye(num_var + num_constr).to(Z.device)

        # RHS of the linear system, mostly partial derivative of grad f wrt Yhat
        b = torch.vstack(
            [
                torch.hstack([dfdZ_dYhat.view((num_var, Yhat.numel()))]),
                torch.hstack([torch.zeros((num_constr, Yhat.numel())).to(Z.device)]),
            ]
        )

        # solution to the system
        derivatives = torch.linalg.solve(A, b)
        if torch.isnan(derivatives).any():
            print("report")
            print(torch.isnan(A).any())
            print(torch.isnan(b).any())
            print(torch.isnan(dgdZ).any())
            print(torch.isnan(diag_lambda).any())
            print(torch.isnan(diag_g).any())
            print(torch.isnan(dfdZ_dYhat).any())
            print(torch.isnan(dfdZ_dZ).any())

        # first num_var are derivatives of primal variables
        dZdYhat = derivatives[:num_var]
        return dZdYhat
