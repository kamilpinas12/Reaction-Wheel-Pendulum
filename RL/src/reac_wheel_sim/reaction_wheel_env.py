import gymnasium as gym
from gymnasium import spaces
import numpy as np
from utils.config_manager import cfg_get
from os import path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pygame

RenderStateType = tuple["pygame.Surface", "pygame.time.Clock", float | None]

class ReactionWheelEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 60}

    def __init__(self, config_name, render_mode=None):
        super(ReactionWheelEnv, self).__init__()

        self.dt = cfg_get("env.dt", config_name, default=0.01)
        self.max_episode_steps = cfg_get(
            "env.max_episode_steps", config_name, default=1000
        )
        self.seed = cfg_get("env.seed", config_name, default=None)
        self.step_count = 0
        self.prev_u = 0.0

        self.action_space = spaces.Box(
            low=np.array([-1.0], dtype=np.float32), 
            high=np.array([1.0], dtype=np.float32), 
            shape=(1,), 
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.inf] * 4, dtype=np.float32), 
            high=np.array([np.inf] * 4, dtype=np.float32), 
            shape=(4,), 
            dtype=np.float32
        )
        self.initial_state_range = cfg_get("env.initial_state_range", config_name, None)
        self.terminate_pend_vel = cfg_get("env.terminate_pend_vel", config_name, default=30.0)
        self.terminate_wheel_vel = cfg_get("env.terminate_wheel_vel", config_name, default=800.0)

        # A
        self.K_pend_vel = cfg_get("env.K_pend_vel", config_name, default=0.085634)
        # B
        self.K_sin = cfg_get("env.K_sin", config_name, default=-9.101332)
        # C
        self.K_reac_wheel = cfg_get("env.K_reac_wheel", config_name, default=-0.009168)

        self.K_motor = 484.73  # K
        self.K_wheel_vel = 0.00229  # D

        self.state = None
        
        self.render_mode = render_mode
        self.render_state = None
        self.render_params = SimpleNamespace(screen_dim=500)
        if self.render_mode is not None and self.render_mode not in self.metadata["render_modes"]:
            raise ValueError(
                f"Unsupported render_mode '{self.render_mode}'. "
                f"Choose one of {self.metadata['render_modes']}"
            )
        self.metadata["render_fps"] = max(1, int(round(1.0 / self.dt)))

    def get_model_params(self):
        return {
            "K_sin": self.K_sin,
            "K_reac_wheel": self.K_reac_wheel,
            "K_pend_vel": self.K_pend_vel,
            "K_motor": self.K_motor,
            "K_wheel_vel": self.K_wheel_vel,
        }
    
    def get_physical_params(self):
        Ip, f, ml = self._convert_model_params_to_phys()
        return {"Ip": Ip, "f":f, "ml":ml}

    @staticmethod
    def _angle_normalize(angle):
        return (angle + np.pi) % (2.0 * np.pi) - np.pi

    def reset(self, seed=None, options=None):
        if seed is None and self.seed is not None:
            seed = self.seed
        super().reset(seed=seed, options=options)

        info = {}
        self.step_count = 0
        self.prev_u = 0.0

        # Enables overrites of rest functinality by passing options
        if options is not None and "initial_state" in options:
            self.state = np.array(options["initial_state"], dtype=np.float32)
        elif options is not None and "initial_range" in options:
            self.state = self._random_initial_state(options["initial_range"])
        elif self.initial_state_range is not None:
            self.state = self._random_initial_state(self.initial_state_range)
        else:
            self.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        if options is not None and "model_params" in options:
            self.K_sin = options["model_params"].get("K_sin", self.K_sin)
            self.K_pend_vel = options["model_params"].get("K_pend_vel", self.K_pend_vel)
            self.K_reac_wheel = options["model_params"].get("K_reac_wheel", self.K_reac_wheel)
            # motor params
            self.K_motor = options["model_params"].get("K_motor", self.K_motor)
            self.K_wheel_vel = options["model_params"].get("K_wheel_vel", self.K_wheel_vel)

        info["model_params"] = self.get_model_params()
        info["phys_params"] = self.get_physical_params()
        return self._get_observation(), info
    
    def step(self, action):
        self.step_count += 1
        u = float(np.clip(action[0], -1.0, 1.0))

        self.state = self._rk4_step(u).astype(np.float32)
        self.state[0] = self._angle_normalize(self.state[0])

        reward = 0.0

        # Safety termination only for clearly divergent trajectories.
        terminated = bool(
            abs(self.state[1]) > self.terminate_pend_vel
            or abs(self.state[2]) > self.terminate_wheel_vel
        )
        truncated = self.step_count >= self.max_episode_steps

        self.prev_u = u
        info = {
            "u_cmd": u,
            "theta": self.state[0],
            "theta_dot": self.state[1],
            "phi": self.state[2],
        }
        return self._get_observation(), reward, terminated, truncated, info

    def _get_observation(self):
        pend_pos, pend_vel, wheel_vel = self.state
        return np.array(
            [pend_pos, pend_vel, wheel_vel, self.prev_u],
            dtype=np.float32,
        )

    def _dynamics(self, state, u):
        pend_pos, pend_vel, wheel_vel = state

        wheel_acc = self.K_motor * (u - self.K_wheel_vel * wheel_vel)
        pend_acc = (
            -self.K_pend_vel * pend_vel
            + self.K_sin * np.sin(pend_pos)
            + self.K_reac_wheel * wheel_acc
        )
        return pend_vel, pend_acc, wheel_acc

    def _rk4_step(self, u):
        s = self.state
        # K1
        k1_0, k1_1, k1_2 = self._dynamics(s, u)
        
        # K2
        s2 = (s[0] + 0.5*self.dt*k1_0, s[1] + 0.5*self.dt*k1_1, s[2] + 0.5*self.dt*k1_2)
        k2_0, k2_1, k2_2 = self._dynamics(s2, u)
        
        # K3
        s3 = (s[0] + 0.5*self.dt*k2_0, s[1] + 0.5*self.dt*k2_1, s[2] + 0.5*self.dt*k2_2)
        k3_0, k3_1, k3_2 = self._dynamics(s3, u)
        
        # K4
        s4 = (s[0] + self.dt*k3_0, s[1] + self.dt*k3_1, s[2] + self.dt*k3_2)
        k4_0, k4_1, k4_2 = self._dynamics(s4, u)

        new_state = np.array([
            s[0] + (self.dt / 6.0) * (k1_0 + 2.0*k2_0 + 2.0*k3_0 + k4_0),
            s[1] + (self.dt / 6.0) * (k1_1 + 2.0*k2_1 + 2.0*k3_1 + k4_1),
            s[2] + (self.dt / 6.0) * (k1_2 + 2.0*k2_2 + 2.0*k3_2 + k4_2)
        ], dtype=np.float32)
        
        return new_state
    
    def _random_initial_state(self, initial_range):
        generator = self.unwrapped.np_random
        pend_pos = generator.uniform(-initial_range[0], initial_range[0])
        pend_vel = generator.uniform(-initial_range[1], initial_range[1])
        wheel_vel = generator.uniform(-initial_range[2], initial_range[2])
        new_state = np.array([pend_pos, pend_vel, wheel_vel], dtype=np.float32)
        return new_state
    
    def _convert_model_params_to_phys(self):
        Iw = 0.00023
        Km = 484.73
        d = 0.00229
        g = 9.81
        Ip = -Iw/self.K_reac_wheel
        f = self.K_sin*Ip
        ml = -self.K_pend_vel*Ip/g
        return Ip, f, ml
    
    def render_image(
        self,
        state,
        render_state: RenderStateType,
        params,
    ):
        """Renders an RGB image."""
        try:
            import pygame
            from pygame import gfxdraw
        except ImportError as e:
            raise AssertionError(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        screen, clock, last_u = render_state

        surf = pygame.Surface((params.screen_dim, params.screen_dim))
        surf.fill((255, 255, 255))

        bound = 2.2
        scale = params.screen_dim / (bound * 2)
        offset = params.screen_dim // 2

        rod_length = 1 * scale
        rod_width = 0.12 * scale
        rod_color = (240, 210, 60)
        l, r, t, b = 0, rod_length, rod_width / 2, -rod_width / 2
        coords = [(l, b), (l, t), (r, t), (r, b)]
        transformed_coords = []
        for c in coords:
            c = pygame.math.Vector2(c).rotate_rad(state[0] - np.pi / 2)
            c = (c[0] + offset, c[1] + offset)
            transformed_coords.append(c)
        gfxdraw.aapolygon(surf, transformed_coords, rod_color)
        gfxdraw.filled_polygon(surf, transformed_coords, rod_color)

        gfxdraw.aacircle(surf, offset, offset, int(rod_width / 2), rod_color)
        gfxdraw.filled_circle(surf, offset, offset, int(rod_width / 2), rod_color)

        rod_end = (rod_length, 0)
        rod_end = pygame.math.Vector2(rod_end).rotate_rad(state[0] - np.pi / 2)
        rod_end = (int(rod_end[0] + offset), int(rod_end[1] + offset))
        gfxdraw.aacircle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), rod_color
        )
        gfxdraw.filled_circle(
            surf, rod_end[0], rod_end[1], int(rod_width / 2), rod_color
        )

        wheel_radius = 0.15 * scale
        gfxdraw.aacircle(surf, rod_end[0], rod_end[1], int(wheel_radius), (60, 60, 60))
        gfxdraw.filled_circle(surf, rod_end[0], rod_end[1], int(wheel_radius), (120, 120, 120))
        wheel_angle = state[2] * getattr(params, "time", 0.0)
        spoke = pygame.math.Vector2(wheel_radius * 0.9, 0).rotate_rad(wheel_angle)
        pygame.draw.line(
            surf,
            (30, 30, 30),
            (rod_end[0], rod_end[1]),
            (rod_end[0] + int(spoke.x), rod_end[1] + int(spoke.y)),
            max(1, int(0.03 * scale)),
        )

        fname = path.join(path.dirname(__file__), "assets/clockwise.png")
        if path.exists(fname) and last_u is not None:
            img = pygame.image.load(fname)
            scale_img = pygame.transform.smoothscale(
                img,
                (scale * np.abs(last_u) / 2, scale * np.abs(last_u) / 2),
            )
            is_flip = bool(last_u > 0)
            scale_img = pygame.transform.flip(scale_img, is_flip, True)
            scale_rect = scale_img.get_rect()
            surf.blit(
                scale_img,
                (
                    rod_end[0] - scale_rect.centerx,
                    rod_end[1] - scale_rect.centery,
                ),
            )

        gfxdraw.aacircle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))
        gfxdraw.filled_circle(surf, offset, offset, int(0.05 * scale), (0, 0, 0))

        surf = pygame.transform.flip(surf, False, True)
        screen.blit(surf, (0, 0))

        return (screen, clock, last_u), np.transpose(
            np.array(pygame.surfarray.pixels3d(screen)), axes=(1, 0, 2)
        )

    def render(self):
        if self.render_mode is None:
            return None
        if self.state is None:
            return None

        if self.render_state is None:
            self.render_state = self.render_init(
                screen_width=self.render_params.screen_dim,
                screen_height=self.render_params.screen_dim,
            )

        self.render_params.time = self.step_count * self.dt
        screen, clock, _ = self.render_state
        self.render_state, frame = self.render_image(
            self.state,
            (screen, clock, self.prev_u),
            self.render_params,
        )

        return frame

    def render_init(
        self,
        screen_width: int = 600,
        screen_height: int = 400,
    ):
        """Initialises the render state."""
        try:
            import pygame
        except ImportError as e:
            raise AssertionError(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e

        pygame.init()
        screen = pygame.Surface((screen_width, screen_height))
        clock = pygame.time.Clock()

        return screen, clock, None

    def render_close(
        self,
        render_state: RenderStateType,
        params,
    ):
        """Closes the render state."""
        try:
            import pygame
        except ImportError as e:
            raise AssertionError(
                'pygame is not installed, run `pip install "gymnasium[classic_control]"`'
            ) from e
        pygame.display.quit()
        pygame.quit()

    def close(self):
        if self.render_state is not None:
            self.render_close(self.render_state, self.render_params)
            self.render_state = None
