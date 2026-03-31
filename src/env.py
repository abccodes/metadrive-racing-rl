"""
Racing environment wrapper for single-agent training.

Wraps MetaDrive's MultiAgentRacingEnv into a standard gymnasium.Env interface.
Supports training with configurable opponent policies including self-play.
"""

import gymnasium
import numpy as np
from metadrive.envs.marl_envs.marl_racing_env import MultiAgentRacingEnv

from .opponents import SCRIPTED_OPPONENTS, build_opponent
from .racing_maps import set_racing_map
from .track_guidance import TrackGuidance

OPPONENT_POLICIES = SCRIPTED_OPPONENTS.copy()


class SelfPlayOpponent:
    """Opponent that uses a copy of the training policy (loaded from checkpoint)."""

    def __init__(self, model_path=None):
        self.model = None
        if model_path:
            self.load(model_path)

    def load(self, model_path):
        import torch
        self.model = torch.load(model_path, map_location="cpu", weights_only=False)
        self.model.eval()

    def __call__(self, obs, agent_id):
        if self.model is None:
            return aggressive_opponent(obs, agent_id)
        import torch
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0)
            action = self.model(obs_t).squeeze(0).numpy()
        return np.clip(action, -1.0, 1.0)


class RacingEnv(gymnasium.Env):
    """
    Single-agent wrapper around MultiAgentRacingEnv.

    Supports curriculum learning by adjusting opponent count and policies.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_agents=2,
        opponent_policy="aggressive",
        extra_config=None,
        render_mode=None,
        map_name="circuit",
        progress_reward_weight=0.0,
        speed_reward_weight=0.0,
        early_reward_horizon=0,
        early_progress_weight=0.0,
        early_speed_weight=0.0,
        speed_target_kmh=80.0,
        use_track_guidance=False,
        guidance_lookahead_steps=(10, 25, 45),
        line_progress_reward_weight=0.0,
        line_speed_reward_weight=0.0,
        line_center_penalty_weight=0.0,
        line_speed_lateral_threshold=3.0,
        line_speed_heading_threshold=0.25,
    ):
        super().__init__()
        self._restore_map = set_racing_map(map_name)
        self.map_name = map_name

        config = {
            "num_agents": num_agents,
            "use_render": render_mode == "human",
            "crash_done": False,
            "crash_vehicle_done": False,
            "out_of_road_done": True,
            "allow_respawn": False,
            "horizon": 3000,
            "map_config": {
                "lane_num": max(2, num_agents),
                "exit_length": 20,
                "bottle_lane_num": max(4, num_agents),
                "neck_lane_num": 1,
                "neck_length": 20,
            },
        }
        if extra_config:
            config.update(extra_config)

        self.env = MultiAgentRacingEnv(config)
        self.num_agents = num_agents
        self.ego_id = "agent0"
        self.progress_reward_weight = float(progress_reward_weight)
        self.speed_reward_weight = float(speed_reward_weight)
        self.early_reward_horizon = max(0, int(early_reward_horizon))
        self.early_progress_weight = float(early_progress_weight)
        self.early_speed_weight = float(early_speed_weight)
        self.speed_target_kmh = max(1e-6, float(speed_target_kmh))
        self.use_track_guidance = bool(use_track_guidance)
        self.guidance = (
            TrackGuidance(map_name, lookahead_steps=guidance_lookahead_steps)
            if self.use_track_guidance
            else None
        )
        self.line_progress_reward_weight = float(line_progress_reward_weight)
        self.line_speed_reward_weight = float(line_speed_reward_weight)
        self.line_center_penalty_weight = float(line_center_penalty_weight)
        self.line_speed_lateral_threshold = float(line_speed_lateral_threshold)
        self.line_speed_heading_threshold = float(line_speed_heading_threshold)
        self._episode_step = 0
        self._prev_route_completion = 0.0
        self._guidance_state = self.guidance.initial_state() if self.guidance else None
        self._prev_line_progress = 0.0

        self._opponent_fn = build_opponent(opponent_policy)

        # Get spaces
        temp_obs, _ = self.env.reset()
        sample_id = list(temp_obs.keys())[0]
        self.observation_space = self.env.observation_space[sample_id]
        self.action_space = self.env.action_space[sample_id]
        self._last_ego_obs = temp_obs.get(self.ego_id, np.zeros(self.observation_space.shape))
        self._opponent_obs = {k: v for k, v in temp_obs.items() if k != self.ego_id}
        self.env.close()
        self.env = MultiAgentRacingEnv(config)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.env.config["start_seed"] = seed
        obs_dict, info_dict = self.env.reset()
        if hasattr(self._opponent_fn, "reset"):
            self._opponent_fn.reset()
        self._last_ego_obs = obs_dict.get(
            self.ego_id, np.zeros(self.observation_space.shape)
        )
        self._opponent_obs = {k: v for k, v in obs_dict.items() if k != self.ego_id}
        ego_info = info_dict.get(self.ego_id, {})
        self._episode_step = 0
        self._prev_route_completion = 0.0
        self._guidance_state = self.guidance.initial_state() if self.guidance else None
        self._prev_line_progress = 0.0
        if self.ego_id in self.env.agents:
            try:
                self._prev_route_completion = float(
                    self.env.agents[self.ego_id].navigation.route_completion
                )
            except Exception:
                self._prev_route_completion = 0.0
        if self.use_track_guidance:
            _, _, metrics = self._compute_guidance(obs_dict.get(self.ego_id, self._last_ego_obs))
            self._prev_line_progress = metrics["progress"]
        ego_info["route_completion"] = self._prev_route_completion
        self._last_ego_obs = obs_dict.get(self.ego_id, self._last_ego_obs)
        return self._last_ego_obs.copy(), ego_info

    def step(self, action):
        self._episode_step += 1
        actions = {}
        for agent_id in list(self.env.agents.keys()):
            if agent_id == self.ego_id:
                actions[agent_id] = action
            else:
                opp_obs = self._opponent_obs.get(agent_id, np.zeros(self.observation_space.shape))
                actions[agent_id] = self._opponent_fn(opp_obs, agent_id)

        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self._opponent_obs = {k: v for k, v in obs.items() if k != self.ego_id}

        if self.ego_id in obs:
            ego_obs = obs[self.ego_id]
        else:
            ego_obs = self._last_ego_obs

        ego_reward = rewards.get(self.ego_id, 0.0)
        ego_terminated = terms.get(self.ego_id, terms.get("__all__", False))
        ego_truncated = truncs.get(self.ego_id, truncs.get("__all__", False))
        ego_info = infos.get(self.ego_id, {})

        route_completion = self._prev_route_completion
        if self.ego_id in self.env.agents:
            try:
                route_completion = float(
                    self.env.agents[self.ego_id].navigation.route_completion
                )
            except Exception:
                route_completion = self._prev_route_completion

        route_delta = max(0.0, route_completion - self._prev_route_completion)
        speed_kmh = max(0.0, float(ego_info.get("speed_km_h", 0.0)))
        progress_bonus = self.progress_reward_weight * route_delta
        speed_bonus = self.speed_reward_weight * min(speed_kmh / self.speed_target_kmh, 1.5)
        ego_reward += progress_bonus + speed_bonus
        if self._episode_step <= self.early_reward_horizon:
            early_progress_bonus = self.early_progress_weight * route_delta
            early_speed_bonus = self.early_speed_weight * min(
                speed_kmh / self.speed_target_kmh, 1.5
            )
            ego_reward += early_progress_bonus + early_speed_bonus
            progress_bonus += early_progress_bonus
            speed_bonus += early_speed_bonus

        guidance_bonus = 0.0
        guidance_penalty = 0.0
        guidance_metrics = None
        if self.use_track_guidance:
            _, guidance_metrics = self._guidance_features_and_metrics(ego_obs)
            line_progress_delta = max(0.0, guidance_metrics["progress"] - self._prev_line_progress)
            guidance_bonus += self.line_progress_reward_weight * (line_progress_delta / 5.0)
            if (
                abs(guidance_metrics["lateral_error"]) <= self.line_speed_lateral_threshold
                and abs(guidance_metrics["heading_error"]) <= self.line_speed_heading_threshold
            ):
                guidance_bonus += self.line_speed_reward_weight * min(
                    speed_kmh / self.speed_target_kmh, 1.5
                )
            guidance_penalty = self.line_center_penalty_weight * min(
                abs(guidance_metrics["lateral_error"]) / 10.0, 2.0
            )
            ego_reward += guidance_bonus - guidance_penalty
            self._prev_line_progress = guidance_metrics["progress"]

        self._prev_route_completion = route_completion
        ego_info["route_completion"] = route_completion
        ego_info["base_reward"] = float(rewards.get(self.ego_id, 0.0))
        ego_info["early_progress_bonus"] = float(progress_bonus)
        ego_info["early_speed_bonus"] = float(speed_bonus)
        ego_info["line_guidance_bonus"] = float(guidance_bonus)
        ego_info["line_guidance_penalty"] = float(guidance_penalty)
        if guidance_metrics is not None:
            ego_info["line_progress"] = float(guidance_metrics["progress"])
            ego_info["line_lateral_error"] = float(guidance_metrics["lateral_error"])
            ego_info["line_heading_error"] = float(guidance_metrics["heading_error"])
        ego_info["shaped_reward"] = float(ego_reward)

        self._last_ego_obs = ego_obs
        return ego_obs.copy(), float(ego_reward), bool(ego_terminated), bool(ego_truncated), ego_info

    def _get_ego_pose(self):
        if self.ego_id in self.env.agents:
            agent = self.env.agents[self.ego_id]
            try:
                return np.asarray(agent.position, dtype=np.float32), float(agent.heading_theta)
            except Exception:
                pass
        return np.zeros(2, dtype=np.float32), 0.0

    def _compute_guidance(self, base_obs):
        if not self.use_track_guidance:
            return np.zeros(0, dtype=np.float32), self._guidance_state, {
                "progress": 0.0,
                "lateral_error": 0.0,
                "heading_error": 0.0,
            }
        position, heading = self._get_ego_pose()
        features, self._guidance_state, metrics = self.guidance.compute(
            position, heading, self._guidance_state
        )
        return features, self._guidance_state, metrics

    def _guidance_features_and_metrics(self, base_obs):
        features, _, metrics = self._compute_guidance(base_obs)
        return features, metrics

    def set_opponent_policy(self, policy):
        """Update the opponent policy (e.g., for self-play curriculum).

        Args:
            policy: Either a string name ("random", "aggressive", "still")
                    or a callable with signature (obs, agent_id) -> action.
        """
        self._opponent_fn = build_opponent(policy)

    def render(self):
        pass

    def close(self):
        self.env.close()
        self._restore_map()


def make_racing_env(
    rank=0,
    num_agents=2,
    opponent_policy="aggressive",
    extra_config=None,
    map_names=None,
    map_name=None,
    progress_reward_weight=0.0,
    speed_reward_weight=0.0,
    early_reward_horizon=0,
    early_progress_weight=0.0,
    early_speed_weight=0.0,
    speed_target_kmh=80.0,
    use_track_guidance=False,
    guidance_lookahead_steps=(10, 25, 45),
    line_progress_reward_weight=0.0,
    line_speed_reward_weight=0.0,
    line_center_penalty_weight=0.0,
    line_speed_lateral_threshold=3.0,
    line_speed_heading_threshold=0.25,
):
    def _init():
        chosen_map = map_name
        if chosen_map is None:
            if map_names:
                chosen_map = map_names[rank % len(map_names)]
            else:
                chosen_map = "circuit"
        env = RacingEnv(
            num_agents=num_agents,
            opponent_policy=opponent_policy,
            extra_config=extra_config,
            map_name=chosen_map,
            progress_reward_weight=progress_reward_weight,
            speed_reward_weight=speed_reward_weight,
            early_reward_horizon=early_reward_horizon,
            early_progress_weight=early_progress_weight,
            early_speed_weight=early_speed_weight,
            speed_target_kmh=speed_target_kmh,
            use_track_guidance=use_track_guidance,
            guidance_lookahead_steps=guidance_lookahead_steps,
            line_progress_reward_weight=line_progress_reward_weight,
            line_speed_reward_weight=line_speed_reward_weight,
            line_center_penalty_weight=line_center_penalty_weight,
            line_speed_lateral_threshold=line_speed_lateral_threshold,
            line_speed_heading_threshold=line_speed_heading_threshold,
        )
        return env
    return _init
