import pdb
import random

import numpy as np
import torch

from openpto.method.Solvers.neural.softTopk import SoftTopk
from openpto.method.Solvers.utils_solver import (
    gather_incomplete_left,
    solve_lineqn,
    trim_left,
)


class RMABSolver(torch.nn.Module):
    """
    Solves for the Whittle Index policy for RMABs with 2-states and 2-actions.
    """

    def __init__(
        self,
        budget,  # the number of arms that can be selected per round
        isTrain=True,  # variable indicating whether this is training or test
    ):
        super(RMABSolver, self).__init__()
        self.budget = budget
        self.isTrain = isTrain
        if isTrain:
            self.soft_topk = SoftTopk(budget)

    def _get_whittle_indices(self, T, gamma):
        """
        Source: https://github.com/armman-projects/Google-AI/
        Inputs:
            Ts: Transition matrix of dimensions ... X 2 X 2 X 2 where axes are:
                ... (batch), start_state, action, end_state
            gamma: Discount factor
        Returns:
            index: ... X 2 Tensor of Whittle index for states (0,1)
        """
        # Matrix equations for state 0
        row1_s0 = torch.stack(
            [
                torch.ones_like(T[..., 0, 0, 0]).to(T.device),
                gamma * T[..., 0, 0, 0] - 1,
                gamma * T[..., 0, 0, 1],
            ],
            -1,
        )
        row2_s0 = torch.stack(
            [
                torch.zeros_like(T[..., 0, 1, 0]).to(T.device),
                gamma * T[..., 0, 1, 0] - 1,
                gamma * T[..., 0, 1, 1],
            ],
            -1,
        )
        row3a_s0 = torch.stack(
            [
                torch.ones_like(T[..., 1, 0, 0]).to(T.device),
                gamma * T[..., 1, 0, 0],
                gamma * T[..., 1, 0, 1] - 1,
            ],
            -1,
        )
        row3b_s0 = torch.stack(
            [
                torch.zeros_like(T[..., 1, 1, 0]).to(T.device),
                gamma * T[..., 1, 1, 0],
                gamma * T[..., 1, 1, 1] - 1,
            ],
            -1,
        )

        A1_s0 = torch.stack([row1_s0, row2_s0, row3a_s0], -2)
        A2_s0 = torch.stack([row1_s0, row2_s0, row3b_s0], -2)
        b_s0 = torch.tensor([0, 0, -1], dtype=torch.float32).to(T.device)

        # Matrix equations for state 1
        row1_s1 = torch.stack(
            [
                torch.ones_like(T[..., 1, 0, 0]).to(T.device),
                gamma * T[..., 1, 0, 0],
                gamma * T[..., 1, 0, 1] - 1,
            ],
            -1,
        )
        row2_s1 = torch.stack(
            [
                torch.zeros_like(T[..., 1, 1, 0].to(T.device)),
                gamma * T[..., 1, 1, 0],
                gamma * T[..., 1, 1, 1] - 1,
            ],
            -1,
        )
        row3a_s1 = torch.stack(
            [
                torch.ones_like(T[..., 0, 0, 0]).to(T.device),
                gamma * T[..., 0, 0, 0] - 1,
                gamma * T[..., 0, 0, 1],
            ],
            -1,
        )
        row3b_s1 = torch.stack(
            [
                torch.zeros_like(T[..., 0, 1, 0]).to(T.device),
                gamma * T[..., 0, 1, 0] - 1,
                gamma * T[..., 0, 1, 1],
            ],
            -1,
        )

        A1_s1 = torch.stack([row1_s1, row2_s1, row3a_s1], -2)
        A2_s1 = torch.stack([row1_s1, row2_s1, row3b_s1], -2)
        b_s1 = torch.tensor([-1, -1, 0], dtype=torch.float32).to(T.device)

        # Compute candidate whittle indices
        cnd1_s0 = solve_lineqn(A1_s0, b_s0)
        cnd2_s0 = solve_lineqn(A2_s0, b_s0)

        cnd1_s1 = solve_lineqn(A1_s1, b_s1)
        cnd2_s1 = solve_lineqn(A2_s1, b_s1)

        # TODO: Check implementation. Getting WI > 1??
        ## Following line implements condition checking when candidate1 is correct
        ## It results in an array of size N, with value 1 if candidate1 is correct else 0.
        cand1_s0_mask = (cnd1_s0[..., 0] + 1.0) + gamma * (
            T[..., 1, 0, 0] * cnd1_s0[..., 1] + T[..., 1, 0, 1] * cnd1_s0[..., 2]
        ) >= 1.0 + gamma * (
            T[..., 1, 1, 0] * cnd1_s0[..., 1] + T[..., 1, 1, 1] * cnd1_s0[..., 2]
        )
        cand1_s1_mask = (cnd1_s1[..., 0]) + gamma * (
            T[..., 0, 0, 0] * cnd1_s1[..., 1] + T[..., 0, 0, 1] * cnd1_s1[..., 2]
        ) >= gamma * (
            T[..., 0, 1, 0] * cnd1_s1[..., 1] + T[..., 0, 1, 1] * cnd1_s1[..., 2]
        )

        cand2_s0_mask = ~cand1_s0_mask
        cand2_s1_mask = ~cand1_s1_mask

        return torch.stack(
            [
                cnd1_s0[..., 0] * cand1_s0_mask + cnd2_s0[..., 0] * cand2_s0_mask,
                cnd1_s1[..., 0] * cand1_s1_mask + cnd2_s1[..., 0] * cand2_s1_mask,
            ],
            -1,
        )

    def forward(
        self,
        T,  # predicted transition probabilities
        gamma,  # discount factor
    ):
        # Make sure the shape is correct
        assert T.shape[-3:] == (2, 2, 2)
        if T.ndim == 4:
            T = T.unsqueeze(0)
        assert T.ndim == 5

        # Get whittle indices
        W = self._get_whittle_indices(T, gamma)

        # Define policy function
        def pi(
            state,  # a vector denoting the current state
        ):
            # Preprocessing
            state = trim_left(state)
            W_temp = trim_left(W)

            # Find the number of common dimensions between W and state
            common_dims = 0
            while (
                common_dims < min(W_temp.ndim - 1, state.ndim)
                and W_temp.shape[-(common_dims + 2)] == state.shape[-(common_dims + 1)]
            ):
                common_dims += 1
            assert (
                common_dims > 0
            )  # ensures that num_arms is consistent across W and state
            assert state.max() < W_temp.shape[-1] and state.min() >= 0

            # Enable broadcasting
            #   Expand state
            if W_temp.ndim > common_dims + 1 and state.ndim == common_dims:
                for i in range(common_dims + 2, W_temp.ndim + 1):
                    state = state.unsqueeze(0).expand(W_temp.shape[-i], *state.shape)
            #   Expand W
            elif state.ndim > common_dims and W_temp.ndim == common_dims + 1:
                for i in range(common_dims + 1, state.ndim + 1):
                    W_temp = W_temp.unsqueeze(0).expand(state.shape[-i], *W_temp.shape)
            #   Expand both
            elif state.ndim > common_dims and W_temp.ndim > common_dims + 1:
                # Special case for get_obj_exact: We want to calculate the policy for all states and Ws
                if (
                    W_temp.ndim == 3
                    and state.ndim == 2
                    and W_temp.shape[-3] != state.shape[-2]
                ):
                    state = state.unsqueeze(0).expand(W_temp.shape[-3], *state.shape)
                    W_temp = W_temp.unsqueeze(1).expand(
                        W_temp.shape[0], state.shape[1], *W_temp.shape[1:]
                    )
                else:
                    raise AssertionError("Invalid shapes")

            # Get whittle indices for the relevant states
            W_state = gather_incomplete_left(W_temp, state)

            # Choose action based on policy
            if self.isTrain:
                gamma = self.soft_topk(-W_state)
                act = gamma[..., 0] * W_state.shape[-1]
            else:
                _, idxs = torch.topk(W_state, self.budget)
                act = torch.nn.functional.one_hot(idxs.squeeze(), W_state.shape[-1])
            return act

        return pi


# Unit test for submodular optimiser
if __name__ == "__main__":
    # Make it reproducible
    rand_seed = 1
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    np.random.seed(rand_seed)

    # Function to generate random transition probabilities
    def generate_instances(R, num_instances, num_arms, num_states, min_lift):
        T = np.zeros((num_instances, num_arms, num_states, 2, num_states))
        for i in range(num_instances):
            for j in range(num_arms):
                for k in range(num_states):
                    while True:
                        passive_transition = np.random.rand(num_states)
                        passive_transition /= passive_transition.sum()
                        active_transition = np.random.rand(num_states)
                        active_transition /= active_transition.sum()
                        if (
                            active_transition @ R > passive_transition @ R + min_lift
                        ):  # Ensure that calling is significantly better
                            T[i, j, k, 0, :] = passive_transition
                            T[i, j, k, 1, :] = active_transition
                            break
        return torch.from_numpy(T).float().detach()

    R = np.arange(2)
    T = generate_instances(R, 2, 5, 2, 0.2).requires_grad_()
    opt = RMABSolver(budget=1)

    # Perform backward pass
    pdb.set_trace()
    state = torch.bernoulli(0.5 * torch.ones(2, 5))
    pi = opt(T, 0.99)
    act = pi(state)

    # Check gradients
    loss = act.square().sum()
    optimizer = torch.optim.Adam([T], lr=1e-3)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
