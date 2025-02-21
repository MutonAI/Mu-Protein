import warnings
from typing import Any, Dict, Optional, Type, TypeVar, Union

import numpy as np
import torch as th
from gym import spaces
from torch.nn import functional as F

from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import explained_variance, get_schedule_fn

PPOSelf = TypeVar("PPOSelf", bound="PPO")


class PPO(OnPolicyAlgorithm):
    """
    Proximal Policy Optimization algorithm (PPO) (clip version)

    Paper: https://arxiv.org/abs/1707.06347
    Code: This implementation borrows code from OpenAI Spinning Up (https://github.com/openai/spinningup/)
    https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail and
    Stable Baselines (PPO2 from https://github.com/hill-a/stable-baselines)

    Introduction to PPO: https://spinningup.openai.com/en/latest/algorithms/ppo.html

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. rollout buffer size is n_steps * n_envs where n_envs is number of environment copies running in parallel)
        NOTE: n_steps * n_envs must be greater than 1 (because of the advantage normalization)
        See https://github.com/pytorch/pytorch/issues/29372
    :param batch_size: Minibatch size
    :param n_epochs: Number of epoch when optimizing the surrogate loss
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
    :param clip_range: Clipping parameter, it can be a function of the current progress
        remaining (from 1 to 0).
    :param clip_range_vf: Clipping parameter for the value function,
        it can be a function of the current progress remaining (from 1 to 0).
        This is a parameter specific to the OpenAI implementation. If None is passed (default),
        no clipping will be done on the value function.
        IMPORTANT: this clipping depends on the reward scaling.
    :param normalize_advantage: Whether to normalize or not the advantage
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param target_kl: Limit the KL divergence between updates,
        because the clipping is not enough to prevent large update
        see issue #213 (cf https://github.com/hill-a/stable-baselines/issues/213)
        By default, there is no limit on the kl div.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        for_protein: bool = False,
        score_threshold: float = -np.inf,
    ):

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
            for_protein=for_protein,
            score_threshold=score_threshold,
        )

        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizer learning rate
        self._update_learning_rate(self.policy.optimizer)
        if self.for_protein:
            self.policy2.set_training_mode(True)
            self._update_learning_rate(self.policy2.optimizer)
        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)

        self.dual_clip = 3.0  # 3.0
        self.ent_coef = 0.00

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        if self.for_protein:
            entropy_losses_2 = []
            pg_losses_2 = []
            value_losses_2 = []
            clip_fractions_2 = []

        continue_training = True

        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            if self.for_protein:
                log_ratios_2 = []
                ratios_2_minus_1 = []
                approx_kl_divs_2 = []

            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                # Note that rollout buffer will use the function 'swap_and_flatten' to convert shape from [n_steps, n_envs, ...] to [n_steps * n_envs, ...].
                actions = rollout_data.actions  # first subaction, shape: [64, 1]
                # print("rollout_data.actions shape: ", actions.shape)
                actions_2 = rollout_data.actions_2  # second subaction, shape: [64, 1]
                # print("rollout_data.actions_2 shape: ", actions_2.shape)

                # Not used for protein.
                if isinstance(self.action_space, spaces.Discrete):
                    # Convert discrete action from float to long
                    actions = rollout_data.actions.long().flatten()
                    actions_2 = rollout_data.actions_2.long().flatten()

                # Re-sample the noise matrix because the log_std has changed
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)
                    if self.for_protein:
                        self.policy2.reset_noise(self.batch_size)

                # rollout_data.observations shape: [64, 572]
                # print("rollout_data.observations shape: ", rollout_data.observations.shape)
                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                # print("values shape: ", values.shape)  # torch.Size([64]) 
                # print("log_prob shape: ", log_prob.shape)  # torch.Size([64])
                # print("entropy shape: ", entropy.shape)  # torch.Size([64])

                if self.for_protein:
                    intermediate_observations = th.cat((rollout_data.observations, actions), axis=1)
                    # print("intermediate_observations shape: ", intermediate_observations.shape)  # torch.Size([64, 573])
                    # print("rollout_data.observations_2 shape: ", rollout_data.observations_2.shape)  # torch.Size([64, 573])
                    assert (intermediate_observations == rollout_data.observations_2).all()

                    values_2, log_prob_2, entropy_2 = self.policy2.evaluate_actions(intermediate_observations, actions_2)
                    values_2 = values_2.flatten()
                    # print("values_2 shape: ", values_2.shape)  # torch.Size([64])
                    # print("log_prob_2 shape: ", log_prob_2.shape)  # torch.Size([64])
                    # print("entropy_2 shape: ", entropy_2.shape)  # torch.Size([64])

                # Normalize advantage
                advantages = rollout_data.advantages
                # print("advantages shape: ", advantages.shape)  # torch.Size([64])
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # Normalize advantage_2
                advantages_2 = rollout_data.advantages_2
                # print("rollout_data.advantages_2 shape: ", advantages_2.shape)
                if self.normalize_advantage and len(advantages_2) > 1:
                    advantages_2 = (advantages_2 - advantages_2.mean()) / (advantages_2.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                # print("rollout_data.old_log_prob shape: ", rollout_data.old_log_prob.shape)  # torch.Size([64])
                # print("rollout_data.old_log_prob_2 shape: ", rollout_data.old_log_prob_2.shape)  # torch.Size([64])
                ratio = th.exp(log_prob - rollout_data.old_log_prob)  # torch.Size([64])
                if self.for_protein:
                    ratio_2 = th.exp(log_prob_2 - rollout_data.old_log_prob_2)  # sometimes very large

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)

                if self.dual_clip is not None:
                    obj_dual_clip = self.dual_clip * advantages  # self.dual_clip > 1, Ye et al. 2021 use 3
                    policy_loss = -((th.max(th.min(policy_loss_1, policy_loss_2), obj_dual_clip)[advantages < 0]).mean() + \
                                    (th.min(policy_loss_1, policy_loss_2)[advantages > 0]).mean())
                else:
                    policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                if self.for_protein:
                    policy_loss_2_1 = advantages_2 * ratio_2
                    policy_loss_2_2 = advantages_2 * th.clamp(ratio_2, 1 - clip_range, 1 + clip_range)

                    if self.dual_clip is not None:
                        obj_dual_clip_2 = self.dual_clip * advantages_2
                        policy_loss_2 = -((th.max(th.min(policy_loss_2_1, policy_loss_2_2), obj_dual_clip_2)[advantages_2 < 0]).mean() + \
                                        (th.min(policy_loss_2_1, policy_loss_2_2)[advantages_2 > 0]).mean())
                    else:
                        policy_loss_2 = -th.min(policy_loss_2_1, policy_loss_2_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.for_protein:
                    pg_losses_2.append(policy_loss_2.item())
                    clip_fraction_2 = th.mean((th.abs(ratio_2 - 1) > clip_range).float()).item()
                    clip_fractions_2.append(clip_fraction_2)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                # Value loss using the TD(gae_lambda) target
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if self.for_protein:
                    if self.clip_range_vf is None:
                        values_pred_2 = values_2
                    else:
                        values_pred_2 = rollout_data.old_values_2 + th.clamp(
                            values_2 - rollout_data.old_values_2, -clip_range_vf, clip_range_vf
                        )
                    value_loss_2 = F.mse_loss(rollout_data.returns_2, values_pred_2)
                    value_losses_2.append(value_loss_2.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)  # Approximate entropy when no analytical form
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                if self.for_protein:
                    if entropy_2 is None:
                        entropy_loss_2 = -th.mean(-log_prob_2)
                    else:
                        entropy_loss_2 = -th.mean(entropy_2)
                    entropy_losses_2.append(entropy_loss_2.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                if self.for_protein:
                    loss += policy_loss_2 + self.ent_coef * entropy_loss_2 + self.vf_coef * value_loss_2

                # Calculate approximate form of reverse KL Divergence for early stopping
                # see issue #417: https://github.com/DLR-RM/stable-baselines3/issues/417
                # and discussion in PR #419: https://github.com/DLR-RM/stable-baselines3/pull/419
                # and Schulman blog: http://joschu.net/blog/kl-approx.html
                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)
                    if self.for_protein:
                        log_ratio_2 = log_prob_2 - rollout_data.old_log_prob_2
                        log_ratios_2.append(th.mean(log_ratio_2).cpu().numpy())
                        ratios_2_minus_1.append(th.mean(th.exp(log_ratio_2) - 1).cpu().numpy())
                        approx_kl_div_2 = th.mean((th.exp(log_ratio_2) - 1) - log_ratio_2).cpu().numpy()
                        approx_kl_divs_2.append(approx_kl_div_2)

                # Not used for protein.
                if self.target_kl is not None and (approx_kl_div > 1.5 * self.target_kl or self.for_protein and approx_kl_div_2 > 1.5 * self.target_kl):
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                if self.for_protein:
                    self.policy2.optimizer.zero_grad()

                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                if self.for_protein:
                    th.nn.utils.clip_grad_norm_(self.policy2.parameters(), self.max_grad_norm)
                    self.policy2.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

        if self.for_protein:
            self.logger.record("train/entropy_loss_2", np.mean(entropy_losses_2))
            self.logger.record("train/policy_gradient_loss_2", np.mean(pg_losses_2))
            self.logger.record("train/value_loss_2", np.mean(value_losses_2))
            self.logger.record("train/log_ratios_2", np.mean(log_ratios_2))
            self.logger.record("train/ratios_2_minus_1", np.mean(ratios_2_minus_1))
            self.logger.record("train/approx_kl_2", np.mean(approx_kl_divs_2))
            self.logger.record("train/clip_fraction_2", np.mean(clip_fractions_2))
            if hasattr(self.policy2, "log_std"):
                self.logger.record("train/policy2_std", th.exp(self.policy2.log_std).mean().item())

    def learn(
        self: PPOSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> PPOSelf:

        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
