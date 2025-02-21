import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import gym
import numpy as np
import torch as th

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer, RolloutBufferForProtein
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

OnPolicyAlgorithmSelf = TypeVar("OnPolicyAlgorithmSelf", bound="OnPolicyAlgorithm")


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        tensorboard_log: Optional[str] = None,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
        for_protein: bool = False,
        score_threshold: float = -np.inf,
    ):

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            support_multi_env=True,
            seed=seed,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
            for_protein=for_protein,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None
        self.score_threshold = score_threshold

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        if self.for_protein:
            assert self.observation_space == self.sub_step_observation_spaces1

            self.rollout_buffer = RolloutBufferForProtein(
                self.n_steps,
                self.sub_step_observation_spaces1,
                self.sub_step_action_spaces1,
                self.sub_step_observation_spaces2,
                self.sub_step_action_spaces2,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )
        else:
            buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer
            self.rollout_buffer = buffer_cls(
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

        if self.for_protein:
            self.policy = self.policy_class(
                self.sub_step_observation_spaces1,
                self.sub_step_action_spaces1,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs
            )
            self.policy = self.policy.to(self.device)
            self.policy2 = self.policy_class2(
                self.sub_step_observation_spaces2,
                self.sub_step_action_spaces2,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs
            )
            self.policy2 = self.policy2.to(self.device)
        else:
            self.policy = self.policy_class(  # pytype:disable=not-instantiable
                self.observation_space,
                self.action_space,
                self.lr_schedule,
                use_sde=self.use_sde,
                **self.policy_kwargs  # pytype:disable=not-instantiable
            )
            self.policy = self.policy.to(self.device)
            
    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        add_dirichlet_noise: bool,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :param for_protein: Specifications for the protein project
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        if self.for_protein:
            mutant_seqs_all_info = []
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        if self.for_protein:
            self.policy2.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)
            if self.for_protein:
                self.policy2.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)
                if self.for_protein:
                    self.policy2.reset_noise(env.num_envs)
            with th.no_grad():
                # self._last_obs shape: [1, 572]
                # obs_tensor shape: [1, 572]
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                # print("obs_tensor shape", obs_tensor.shape)
                if self.for_protein:
                    # if env.envs[0].env.cur_mutations == 0:
                    #     # I cannot use the initial_prior for the first step, because the initial_prior is designed for protein project
                    #     sub_actions1, values, log_sub_action_1_probs = self.policy(obs_tensor, add_dirichlet_noise=add_dirichlet_noise, initial_prior=True)
                    # else:
                    #     sub_actions1, values, log_sub_action_1_probs = self.policy(obs_tensor, add_dirichlet_noise=add_dirichlet_noise)
                    # TODO: I need to use the initial_prior for the first step
                    sub_actions1, values, log_sub_action_1_probs = self.policy(obs_tensor, add_dirichlet_noise=add_dirichlet_noise)

                    # print("sub_actions1 shape", sub_actions1.shape)  # sub_actions1 shape: [1, 1]
                    # print("values shape", values.shape)  # values shape: [1, 1]
                    # print("log_sub_action_1_probs shape", log_sub_action_1_probs.shape)  # log_sub_action_1_probs shape: [1]
                    intermediate_state = th.cat((obs_tensor, sub_actions1), dim=1)
                    # print("intermediate_state shape", intermediate_state.shape)  # intermediate_state shape: [1, 573]
                    sub_actions2, values_2, log_sub_action_2_probs = self.policy2(intermediate_state, add_dirichlet_noise=add_dirichlet_noise)
                    # print("sub_actions2 shape", sub_actions2.shape)  # sub_actions2 shape: [1, 1]
                    # print("values_2 shape", values_2.shape)  # values_2 shape: [1, 1]
                    # print("log_sub_action_2_probs shape", log_sub_action_2_probs.shape)  # log_sub_action_2_probs shape: [1]
                    actions = th.cat((sub_actions1, sub_actions2), dim=1)  # actions shape: [1, 2]
                    # print("actions shape", actions.shape)
                else:
                    actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # new_obs shape: [1, 572]
            # rewards shape: [1]
            # dones shape: [1]
            # infos: [{}]

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            # Since action space in DARWIN is MultiDiscrete, skip this part.
            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if done and self.for_protein:
                    mutant_seqs_all_info.append((infos[idx]["current_sequence"], rewards[idx], infos[idx]["embedding"], infos[idx]["ensemble_uncertainty"]))
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    assert not self.for_protein # not used for protein
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            if self.for_protein:
                action_1 = sub_actions1.cpu().numpy()  # action_1 shape: [1, 1]
                obs_2 = np.concatenate((self._last_obs, action_1), axis=1)  # obs_2 shape: [1, 573]
                # print("obs_2 shape", obs_2.shape)
                action_2 = sub_actions2.cpu().numpy()  # action_2 shape: [1, 1]
                rollout_buffer.add(self._last_obs, action_1, obs_2, action_2, rewards, self._last_episode_starts, values, values_2, log_sub_action_1_probs, log_sub_action_2_probs)                
            else:
                rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)

            self._last_obs = new_obs
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            last_obs = obs_as_tensor(new_obs, self.device)
            values = self.policy.predict_values(last_obs)  # shape: [1, 1]
            last_obs_2 = th.cat((last_obs, sub_actions1), dim=1)  # shape: [1, 573]
            values_2 = self.policy2.predict_values(last_obs_2)  # shape: [1, 1]

        rollout_buffer.compute_returns_and_advantage(last_values=values, last_values_2=values_2, dones=dones)

        callback.on_rollout_end()

        return mutant_seqs_all_info if self.for_protein else True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self: OnPolicyAlgorithmSelf,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "OnPolicyAlgorithm",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        new_starting_sequence: Optional[str] = None,
    ) -> OnPolicyAlgorithmSelf:
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )
        if self.for_protein:
            candidate_pool = []
            sampling_candidate_pool = []  # for sampling
            from flexs.utils.eval_utils import sequence_to_mutation
            from collections import Counter
            total_seqs_num_during_training = 0
            total_seqs_num_after_filtering = 0
            unique_sequences_during_training = set()
            unique_sequences_after_filtering = set()
            counter = Counter()
            counter_after_filtering = Counter()
            if new_starting_sequence is not None:
                self.env.envs[0].change_starting_seq(new_starting_sequence)

        callback.on_training_start(locals(), globals())
        import time

        while self.num_timesteps < total_timesteps:
            # for demo
            # sampling_returns = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, add_dirichlet_noise=False)
            t0 = time.time()
            returns = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps, add_dirichlet_noise=True)
            t1 = time.time()
            print("Collect rollouts time: ", t1 - t0)

            if self.for_protein:
                candidate_pool += returns
                # sampling_candidate_pool += sampling_returns
                for seq, score, _, _ in returns:
                    seq = self.env.envs[0].array2str(seq) # from numpy array to string

                    if score > self.score_threshold:
                        total_seqs_num_after_filtering += 1
                        if seq not in unique_sequences_after_filtering:
                            unique_sequences_after_filtering.add(seq)
                            _, mutation_sites_num = sequence_to_mutation(seq, self.env.envs[0].starting_seq)
                            counter_after_filtering[mutation_sites_num] += 1

                    if seq not in unique_sequences_during_training:
                        unique_sequences_during_training.add(seq)
                        _, mutation_sites_num = sequence_to_mutation(seq, self.env.envs[0].starting_seq)
                        counter[mutation_sites_num] += 1

                total_seqs_num_during_training += len(returns)
            
            if returns is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
                fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                # logger
                self.logger.record("Darwin/training_unique_seqs_num", len(unique_sequences_during_training))
                self.logger.record("Darwin/training_5_points_num", counter[5])
                self.logger.record("Darwin/training_4_points_num", counter[4])
                self.logger.record("Darwin/training_3_points_num", counter[3])
                self.logger.record("Darwin/training_2_points_num", counter[2])
                self.logger.record("Darwin/training_1_points_num", counter[1])
                self.logger.record("Darwin/training_unique_ratio", len(unique_sequences_during_training) / float(total_seqs_num_during_training))
                self.logger.record("Darwin/training_5_points_ratio", counter[5] / float(len(unique_sequences_during_training)))
                self.logger.record("Darwin/training_4_points_ratio", counter[4] / float(len(unique_sequences_during_training)))
                self.logger.record("Darwin/training_3_points_ratio", counter[3] / float(len(unique_sequences_during_training)))
                self.logger.record("Darwin/training_2_points_ratio", counter[2] / float(len(unique_sequences_during_training)))
                self.logger.record("Darwin/training_1_points_ratio", counter[1] / float(len(unique_sequences_during_training)))

                self.logger.record("Darwin/filtered_unique_seqs_num", len(unique_sequences_after_filtering))
                self.logger.record("Darwin/filtered_5_points_num", counter_after_filtering[5])
                self.logger.record("Darwin/filtered_4_points_num", counter_after_filtering[4])
                self.logger.record("Darwin/filtered_3_points_num", counter_after_filtering[3])
                self.logger.record("Darwin/filtered_2_points_num", counter_after_filtering[2])
                self.logger.record("Darwin/filtered_1_points_num", counter_after_filtering[1])
                if len(unique_sequences_after_filtering) > 0:
                    self.logger.record("Darwin/filtered_unique_ratio", len(unique_sequences_after_filtering) / float(total_seqs_num_after_filtering))
                    self.logger.record("Darwin/filtered_5_points_ratio", counter_after_filtering[5] / float(len(unique_sequences_after_filtering)))
                    self.logger.record("Darwin/filtered_4_points_ratio", counter_after_filtering[4] / float(len(unique_sequences_after_filtering)))
                    self.logger.record("Darwin/filtered_3_points_ratio", counter_after_filtering[3] / float(len(unique_sequences_after_filtering)))
                    self.logger.record("Darwin/filtered_2_points_ratio", counter_after_filtering[2] / float(len(unique_sequences_after_filtering)))
                    self.logger.record("Darwin/filtered_1_points_ratio", counter_after_filtering[1] / float(len(unique_sequences_after_filtering)))
                else:
                    self.logger.record("Darwin/filtered_unique_ratio", 0)
                    self.logger.record("Darwin/filtered_5_points_ratio", 0)
                    self.logger.record("Darwin/filtered_4_points_ratio", 0)
                    self.logger.record("Darwin/filtered_3_points_ratio", 0)
                    self.logger.record("Darwin/filtered_2_points_ratio", 0)
                    self.logger.record("Darwin/filtered_1_points_ratio", 0)
                self.logger.dump(step=self.num_timesteps)

            t2 = time.time()
            print("Logging time: ", t2 - t1)

            self.train()

            t3 = time.time()
            print("Train time: ", t3 - t2)

        callback.on_training_end()

        return [candidate_pool, sampling_candidate_pool] if self.for_protein else self # Inherit SB3

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
