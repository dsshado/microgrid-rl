"""PPO actor-critic policy with GCAPS feature extractor (stable-baselines3 compatible)."""

import torch as th
import gymnasium as gym
from typing import Any, Dict, List, Optional, Tuple, Type, Union
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, FlattenExtractor, MlpExtractor
from stable_baselines3.common.distributions import (
    BernoulliDistribution, CategoricalDistribution, DiagGaussianDistribution,
    Distribution, MultiCategoricalDistribution, StateDependentNoiseDistribution,
    make_proba_distribution,
)
from Policies.bus_123.Feature_Extractor import get_extractor


class ActorCriticGCAPSPolicy(BasePolicy):

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[Dict] = None,
        activation_fn: Type[th.nn.Module] = th.nn.Tanh,
        ortho_init: bool = True,
        use_sde: bool = False,
        log_std_init: float = 0.0,
        full_std: bool = True,
        sde_net_arch: Optional[List[int]] = None,
        use_expln: bool = False,
        squash_output: bool = False,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[th.device, str] = "cpu"
    ):
        super().__init__(
            observation_space, action_space,
            features_extractor_class, features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            squash_output=squash_output
        )

        features_dim = features_extractor_kwargs['features_dim']
        node_dim     = features_extractor_kwargs['node_dim']
        gnn_type     = features_extractor_kwargs.get('gnn_type', 'GCAPS')
        self.node_dim = node_dim

        self.value_net = th.nn.Sequential(
            th.nn.Linear(features_dim, 2 * features_dim),
            th.nn.Linear(2 * features_dim, 2 * features_dim),
            th.nn.Linear(2 * features_dim, 1)
        ).to(device=device)

        ExtractorClass = get_extractor(gnn_type)
        self.features_extractor = ExtractorClass(
            observation_space=observation_space,
            features_dim=features_dim,
            node_dim=node_dim,
        )

        self.optimizer  = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
        self.action_dist = make_proba_distribution(action_space, use_sde=use_sde)

        self.net_arch      = net_arch if net_arch is not None else {"pi": [features_dim], "vf": [features_dim]}
        self.activation_fn = activation_fn
        self.action_net    = self.action_dist.proba_distribution_net(latent_dim=features_dim)
        self.mlp_extractor = MlpExtractor(
            features_dim, net_arch=self.net_arch, activation_fn=self.activation_fn, device="auto"
        )

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features             = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        values               = self.value_net(latent_vf)
        distribution         = self._get_action_dist_from_latent(latent_pi, obs)
        actions              = distribution.get_actions(deterministic=True)
        log_prob             = distribution.log_prob(actions)
        actions              = actions.reshape((-1,) + self.action_space.shape)
        return actions, values, log_prob

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        features             = self.extract_features(obs, self.features_extractor)
        latent_pi, latent_vf = self.mlp_extractor(features)
        distribution         = self._get_action_dist_from_latent(latent_pi, obs)
        log_prob             = distribution.log_prob(actions)
        values               = self.value_net(latent_vf)
        return values, log_prob, distribution.entropy()

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        actions, _, _ = self.forward(observation, deterministic=deterministic)
        return actions

    def predict_values(self, obs: th.Tensor) -> th.Tensor:
        features  = self.extract_features(obs, self.features_extractor)
        latent_vf = self.mlp_extractor.forward_critic(features)
        return self.value_net(latent_vf)

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor, obs) -> Distribution:
        mean_actions = self.action_net(latent_pi)
        mask         = obs["ActionMasking"].to(th.bool)
        mean_actions[mask] += -1000000 * mean_actions[mask].abs()

        if isinstance(self.action_dist, DiagGaussianDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std)
        elif isinstance(self.action_dist, CategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, MultiCategoricalDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, BernoulliDistribution):
            return self.action_dist.proba_distribution(action_logits=mean_actions)
        elif isinstance(self.action_dist, StateDependentNoiseDistribution):
            return self.action_dist.proba_distribution(mean_actions, self.log_std, latent_pi)
        else:
            raise ValueError("Invalid action distribution")
