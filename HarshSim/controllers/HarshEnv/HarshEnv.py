import os, random, math, cv2, torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gymnasium as gym
from gymnasium import spaces
from gymnasium.spaces import Dict as SpaceDict
from gymnasium.spaces import Box, Dict

from controller import Supervisor, Motor, GPS, Compass, Receiver, Emitter, Lidar

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from sb3_contrib  import RecurrentPPO
from gymnasium import Env

# ----------------------------------------------------------------------
# Global constants (robot-independent)
# ----------------------------------------------------------------------
TIME_STEP     = 32                 # [ms] Webots basic timestep
MAX_LINEAR    = 6                  # [m/s] after scaling
MAX_ANGULAR   = 6                  # [rad/s] after scaling
HORIZON       = 3                  # 1 current + 9 future actions
SEED          = 42
_goal_reached = 1
LIDAR_ANGLES  = 9
MAX_DIST      = 5
max_steps     = 1000


CM_SIZE       = 20
map_dim       = (200, 200)
map_res       = .1
map_num_rays = 180

lamda_progress   = 100
lamda_lidar      = -1/50
lamda_heading    = -1/40
lamda_terrain    = -1/1000
lamda_ang        = -1/1000
lamda_goal       = 1
lamda_obs        = -1
lamda_future     = 1/HORIZON
# ======================================================================
# Environment definition
# ======================================================================
class HarshEnv(Env):
    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    def __init__(self, seed: int = SEED, train: bool = True):
        super().__init__()
        self.horizon = HORIZON
        self.train   = train

        # === Webots devices ==========================================
        self.robot     = Supervisor()
        self.robot_node= self.robot.getSelf()

        self.gps       = self.robot.getDevice("gps");     self.gps.enable(TIME_STEP)
        self.compass   = self.robot.getDevice("compass"); self.compass.enable(TIME_STEP)
        self.receiver  = self.robot.getDevice("receiver");self.receiver.enable(TIME_STEP); self.receiver.setChannel(1)
        self.emitter   = self.robot.getDevice("emitter"); self.emitter.setChannel(2)
        self.lidar     = self.robot.getDevice("lidar");   self.lidar.enable(TIME_STEP)

        # ----  LIDAR parameters read automatically -------------------
        self.num_rays2   = int(self.lidar.getHorizontalResolution())
        self.num_rays    = LIDAR_ANGLES
        self.fov         = float(self.lidar.getFov())            # [rad]
        self.max_range   = float(self.lidar.getMaxRange())       # [m]
        self.ang_per_ray = self.fov / (self.num_rays - 1)
            
        # Wheels ------------------------------------------------------
        wheel_names = ['front left wheel', 'front right wheel',
                       'back left wheel',  'back right wheel']
        self.wheels = [self.robot.getDevice(n) for n in wheel_names]
        for w in self.wheels:
            w.setPosition(float('inf')); w.setVelocity(0.0)

        # Cameras for terrain colour cues ----------------------------
        for cam_name in ["camera_right", "camera_front", "camera_left", "camera"]:
            cam = self.robot.getDevice(cam_name); cam.enable(TIME_STEP)
            setattr(self, cam_name, cam)

        # Global mao
        self.map_res = map_res  # [m/cell] resolution of map
        self.global_map = -1 * np.ones(map_dim, dtype=np.float32)
        self.map_ang_per_ray = self.fov / (map_num_rays - 1)

        # === Gym spaces =============================================
        act_dim    = 2 * self.horizon
        future_dim = 2 * (self.horizon - 1)

        self.action_space = spaces.Box(low=-1.0, high=1.0,
                                       shape=(act_dim,), dtype=np.float32)

        flat_dim = LIDAR_ANGLES + 1 + 1 + 3 + 2 * (HORIZON - 1)
        self.observation_space = Dict({
            "flat": Box(low=-1, high=1, shape=(flat_dim,), dtype=np.float32),
            "costmap": Box(low=-1, high=1, shape=(CM_SIZE, CM_SIZE), dtype=np.float32)
        })

        # Episode bookkeeping ---------------------------------------
        self.human_position   = [0.0, 0.0]
        self.max_steps        = max_steps
        self.steps            = 0
        self.visitHarsh       = 0
        self.prev_dist_to_goal= None
        self.reached_goal     = False
        
        self.lidar_ranges = np.zeros(LIDAR_ANGLES, dtype=np.float32)
        self.dist_goal = 0
        self.ang_diff = 0
        self.harsh = [0, 0, 0]
        
        self.seed(seed)

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.visitHarsh = 0
        self.prev_dist_to_goal = None
        self.reached_goal = False
        self.global_map = -1 * np.ones(map_dim, dtype=np.float32)
        self.global_map = -1 * np.ones(map_dim, dtype=np.float32)
        self.log_odds_map = np.zeros(map_dim, dtype=np.float32)  # this line is missing

        # Ask supervisor world-builder to reset environment
        self.emitter.send("reset".encode())
        for _ in range(20): self.robot.step(TIME_STEP)  # let everything settle

        self._read_human_position()
        obs = self._get_obs(np.zeros(2*(self.horizon-1)))
        return obs, {}

    def step(self, action_horizon):
        self.steps += 1
        actions         = [action_horizon[i:i+2] for i in range(0, len(action_horizon), 2)]
        curr_action     = actions[0]
        self.future_actions = actions[1:]

        # --- Apply current action -----------------------------------
        self._apply_velocity(curr_action[0], curr_action[1])
        self.robot.step(TIME_STEP)
        self._read_human_position()

        # --- Observation -------------------------------------------
        flat_future = np.array([v for pair in self.future_actions for v in pair])
        obs         = self._get_obs(flat_future)

        # --- Future-reward look-ahead ------------------------------
        future_reward = 0.0
        if self.train:
            state = self._save_robot_state()
            future_reward, terminated = self._evaluate_future_actions()
            self._restore_robot_state(state)

        base_reward, terminated = self._compute_reward(curr_action)
        reward    = base_reward + future_reward* lamda_future
        truncated = self.steps >= self.max_steps
        
        self._build_costmap()
        #if self.steps % 1 == 0 and self.train:
            #self.show_map()
        return obs, float(reward), bool(terminated), bool(truncated), {}

    def close(self): self._apply_velocity(0, 0)
    def seed(self, seed=None):
        np.random.seed(seed); random.seed(seed)

    # ------------------------------------------------------------------
    # Observation / reward helpers
    # ------------------------------------------------------------------
    def _get_obs(self, future_actions):
        lidar_ranges = np.clip(self._get_lidar(), 0, MAX_DIST)               
        pos_robot  = np.array(self.gps.getValues()[:2])
        yaw_robot  = math.atan2(*self.compass.getValues()[0:2])

        pos_goal   = np.array(self.human_position)
        vec_goal   = pos_goal - pos_robot
        dist_goal  = np.linalg.norm(vec_goal)
        ang_goal   = math.atan2(vec_goal[1], vec_goal[0])
        ang_diff   = (ang_goal - yaw_robot + math.pi) % (2*math.pi) - math.pi
        right_c  = self.get_dominant_color_value(self.camera_right)
        front_c  = self.get_dominant_color_value(self.camera_front)
        left_c   = self.get_dominant_color_value(self.camera_left)
        if self.get_dominant_color_value(self.camera) != 0:  # current cell harsh?
            self.visitHarsh += 1
        self.lidar_ranges = lidar_ranges
        self.dist_goal = dist_goal
        self.ang_diff = ang_diff
        self.harsh = [right_c, front_c, left_c]
                            
        flat_obs = np.concatenate((lidar_ranges / self.max_range,
                                   [dist_goal / 100, ang_diff / math.pi],
                                   [right_c, front_c, left_c],
                                   future_actions)).astype(np.float32)
    
        local_patch = self._get_local_patch()  # shape (CM_SIZE, CM_SIZE)
        return {
            "flat": flat_obs,
            "costmap": local_patch
        }
        
    def _compute_reward(self, action):
        lidar, dist_goal, ang_diff = self.lidar_ranges, self.dist_goal, self.ang_diff
        right_c , front_c , left_c = self.harsh
        
        progress = (self.prev_dist_to_goal - dist_goal)*lamda_progress if self.prev_dist_to_goal is not None else 0.0
        self.prev_dist_to_goal = dist_goal

        near_obstacle = np.any(lidar[:]< 0.5)
        lidar_penalty = np.sum((self.max_range - lidar)**2) * lamda_lidar
        ang_penalty   = abs(action[1]) * lamda_ang
        head_penalty  = abs(ang_diff) * lamda_heading
        goal_bonus    = lamda_goal if dist_goal < 1.0 else 0.0
        terrain_r     = (right_c + front_c + left_c) * lamda_terrain

        reward = progress + lidar_penalty + ang_penalty + head_penalty + goal_bonus + terrain_r
        if near_obstacle: reward += lamda_obs

        self.reached_goal = dist_goal < _goal_reached
        terminated = self.reached_goal or near_obstacle
        return reward, terminated

    # ------------------------------------------------------------------
    # Future-reward evaluation
    # ------------------------------------------------------------------
    def _evaluate_future_actions(self):
        total_reward = 0.0
        prev_dist = self.prev_dist_to_goal
        scan0 = self._get_lidar()
        prev_pos = np.array(self.gps.getValues()[:2])
        prev_yaw = math.atan2(*self.compass.getValues()[0:2])
        
        for i, (lin, ang) in enumerate(self.future_actions):
            # Apply action
            self._apply_velocity(lin, ang)
            self.robot.step(TIME_STEP)
    
            # Get actual new position and orientation

            new_pos = np.array(self.gps.getValues()[:2])
            new_yaw = math.atan2(*self.compass.getValues()[0:2])
            
            # Predict lidar at new pose
            debug = False
            if i == len(self.future_actions)-1: debug = False
            pred_lidar = self.estimate_lidar(prev_pos, prev_yaw, new_pos, new_yaw, scan0, debug)
            pred_lidar = np.clip(pred_lidar, 0, MAX_DIST)               

            # Goal-related calculations
            vec_goal = np.array(self.human_position) - new_pos
            dist_goal = np.linalg.norm(vec_goal)
            ang_goal = math.atan2(vec_goal[1], vec_goal[0])
            ang_diff = (ang_goal - new_yaw + math.pi) % (2 * math.pi) - math.pi
    
            # Rewards
            progress_r = (prev_dist - dist_goal) * lamda_progress if prev_dist is not None else 0.0
            prev_dist = dist_goal
    
            near_obs = np.any(pred_lidar < 0.5)
            lidar_r = np.sum((self.max_range - pred_lidar)**2) * lamda_lidar 
            ang_r = abs(ang) * lamda_ang
            head_r = abs(ang_diff) * lamda_heading
            goal_r = lamda_goal if dist_goal < _goal_reached else 0.0
            obs_r = lamda_obs if near_obs else 0.0
    
            step_reward = progress_r + lidar_r/ ((i+1)*2) + ang_r + head_r + goal_r + obs_r
            total_reward += step_reward / ((i+1)*2)
    
            if dist_goal < _goal_reached or near_obs:
                break
    
        self._apply_velocity(0, 0)
        return total_reward, dist_goal < _goal_reached or near_obs


    # ------------------------------------------------------------------
    # LIDAR Exposure helper
    # ------------------------------------------------------------------
    def estimate_lidar(self, pos0, yaw0, pos1, yaw1, scan0, debug=False):
        ray_angles = np.linspace(-self.fov / 2, self.fov / 2, self.num_rays)
        d_pos = pos1 - pos0
        d_yaw = (yaw1 - yaw0 + np.pi) % (2 * np.pi) - np.pi
    
        est = []
        keep_indices = []
    
        for i, rel_a in enumerate(ray_angles):
            new_rel = rel_a - d_yaw
            if -self.fov / 2 <= new_rel <= self.fov / 2:
                global_a = yaw0 + rel_a
                ray_dir = np.array([np.cos(global_a), np.sin(global_a)])
                along = np.dot(d_pos, ray_dir)
                projected_dist = scan0[i] - along
                est.append(projected_dist)
                keep_indices.append(i)
    
        est = np.array(est, dtype=np.float32)
        if debug:
            plt.clf()
            plt.plot(scan0, label="Original Scan (scan0)",linewidth=3, linestyle="--", color='red')
            plt.plot(est, label="Estimated Scan (est)", color='green', linewidth=1.5)
            plt.title("LiDAR Projection Debug")
            plt.xlabel("Ray Index")
            plt.ylabel("Distance (m)")
            plt.ylim(0, self.max_range)
            plt.xlim(0, 180)

            plt.legend()
            plt.grid(True)
            plt.pause(0.01)
    
        return est
    


    
    # ------------------------------------------------------------------
    # Robot state save / restore (for look-ahead)
    # ------------------------------------------------------------------
    def _save_robot_state(self):
        return {
            'pos':  list(self.robot_node.getField("translation").getSFVec3f()),
            'rot':  list(self.robot_node.getField("rotation").getSFRotation()),
            'step': self.steps,
            'visit':self.visitHarsh,
            'prev': self.prev_dist_to_goal,
            'goal': self.reached_goal
        }

    def _restore_robot_state(self, s):
        self.robot_node.getField("translation").setSFVec3f(s['pos'])
        self.robot_node.getField("rotation").setSFRotation(s['rot'])
        self.robot_node.resetPhysics()

        self.steps            = s['step']
        self.visitHarsh       = s['visit']
        self.prev_dist_to_goal= s['prev']
        self.reached_goal     = s['goal']
        for _ in range(5): self.robot.step(TIME_STEP)   # let sensors settle

    # ------------------------------------------------------------------
    # low-level controller
    # ------------------------------------------------------------------
    def _apply_velocity(self, lin_cmd, ang_cmd):
        lin = (lin_cmd + 1.0) / 2.0 * MAX_LINEAR
        ang = ang_cmd * MAX_ANGULAR
        left_speed  = np.clip(lin - ang, -6, 6)
        right_speed = np.clip(lin + ang, -6, 6)
        for i in [0, 2]: self.wheels[i].setVelocity(left_speed)
        for i in [1, 3]: self.wheels[i].setVelocity(right_speed)
    
    
    # ------------------------------------------------------------------
    # Misc. helpers
    # ------------------------------------------------------------------
    def _get_lidar(self):
        scan = np.array(self.lidar.getRangeImage())
        lidar_ranges = np.interp(np.linspace(0, len(scan) - 1, LIDAR_ANGLES),
                         np.arange(len(scan)), scan)
        return lidar_ranges

    def _read_human_position(self):
        while self.receiver.getQueueLength() > 0:
            msg = self.receiver.getString()
            try:
                self.human_position = [float(x) for x in msg.split(',')[:2]]
            except ValueError:
                self.human_position = [0.0, 0.0]
            self.receiver.nextPacket()

    def get_dominant_color_value(self, cam):
        image  = cam.getImage()
        w, h   = cam.getWidth(), cam.getHeight()
        img_np = np.zeros((h, w, 3), dtype=np.uint8)
        for y in range(h):
            for x in range(w):
                r = cam.imageGetRed  (image, w, x, y)
                g = cam.imageGetGreen(image, w, x, y)
                b = cam.imageGetBlue (image, w, x, y)
                img_np[y, x] = [b, g, r]      # BGR → OpenCV

        hsv     = cv2.cvtColor(img_np, cv2.COLOR_BGR2HSV)
        ranges  = {"red":((0,150,150),(10,255,255)),
                   "green":((45,100,100),(75,255,255)),
                   "blue":((100,150,150),(130,255,255))}
        scores  = {"red":0.05,"blue":0,"green":0,"other":0}
        counts  = {c:np.sum(cv2.inRange(hsv, np.array(lo), np.array(hi))>0)
                   for c,(lo,hi) in ranges.items()}
        dom     = max(counts, key=counts.get) if counts else "other"
        return scores.get(dom,0)
       
    # ――――― cost‑map ―――――――――――――――――――――――――――――――――――――――――――――――――
    def _build_costmap(self):
        scan = np.clip(np.array(self.lidar.getRangeImage()), 0.01, self.max_range)
        robot_pos = np.array(self.gps.getValues()[:2])
        yaw = math.atan2(*self.compass.getValues()[0:2])
    
        # Log-odds parameters
        l_occ = 0.85    # log odds of occupancy
        l_free = -0.4   # log odds of free space
        l_min = -2.0
        l_max = 3.5
        unknown = -1
    
        if not hasattr(self, "log_odds_map"):
            self.log_odds_map = np.zeros(map_dim, dtype=np.float32)
    
        for i, dist in enumerate(scan):
            angle_rel = self.fov / 2 - i * self.map_ang_per_ray  # flip direction
            angle_global = yaw + angle_rel
    
            # Ray casting
            num_steps = max(int(dist / self.map_res), 1)
            for step_frac in np.linspace(0, 1, num_steps, endpoint=False):
                x = robot_pos[0] + step_frac * dist * math.cos(angle_global)
                y = robot_pos[1] + step_frac * dist * math.sin(angle_global)
                mx = int(x / self.map_res)
                my = int(y / self.map_res)
    
                if 0 <= mx < map_dim[1] and 0 <= my < map_dim[0]:
                    self.log_odds_map[my, mx] = np.clip(self.log_odds_map[my, mx] + l_free, l_min, l_max)
    
            # End of ray — potential obstacle
            if dist < self.max_range - 1e-2:
                x = robot_pos[0] + dist * math.cos(angle_global)
                y = robot_pos[1] + dist * math.sin(angle_global)
                mx = int(x / self.map_res)
                my = int(y / self.map_res)
    
                if 0 <= mx < map_dim[1] and 0 <= my < map_dim[0]:
                    self.log_odds_map[my, mx] = np.clip(self.log_odds_map[my, mx] + l_occ, l_min, l_max)
    
        # Convert log-odds to probability for visualization / usage
        self.global_map = 1 - 1 / (1 + np.exp(self.log_odds_map))
        self.global_map[self.log_odds_map == 0] = unknown  # unknown

    def show_map(self):    
        plt.figure("Global Map with Local Patch",figsize=(8, 8))
        plt.clf()
        plt.imshow(self.global_map, cmap='gray', origin='lower')
        plt.title(f"Global Map (step {self.steps})")
        
        # Get robot position on global map grid
        robot_pos = np.array(self.gps.getValues()[:2])  # (x, y)
        mx = int(robot_pos[0] / self.map_res)
        my = int(robot_pos[1] / self.map_res)
    
        # Get local patch
        patch = self._get_local_patch()  # shape: (CM_SIZE, CM_SIZE)
        half = CM_SIZE // 2
        #print(patch)

        # Overlay the local patch using color-coded rectangles
        for dy in range(CM_SIZE):
            for dx in range(CM_SIZE):
                gx = mx + dx - half
                gy = my + dy - half
                # Make sure it's within global map bounds
                if 0 <= gx < self.global_map.shape[1] and 0 <= gy < self.global_map.shape[0]:
                    val = patch[dy, dx]
                    if val == -1:
                        color = 'blue'  # unknown
                    elif val > 0.7:
                        color = 'darkgreen'  # likely obstacle
                    elif val < 0.3:
                        color = 'lightgreen'  # likely free
                    else:
                        color = 'yellow'  # uncertain
                    # Draw colored square
                    plt.gca().add_patch(plt.Rectangle((gx - 0.5, gy - 0.5), 1, 1, color=color, alpha=0.6))
    
        # Optionally draw robot center
        plt.plot(mx, my, 'ro', markersize=4, label='Robot')
        # Draw robot orientation arrow
        yaw = math.atan2(self.compass.getValues()[0], self.compass.getValues()[1])
        arrow_length = 2.0
        dx = arrow_length * math.cos(yaw)
        dy = arrow_length * math.sin(yaw)
    
        plt.arrow(mx, my, dx, dy,
                  head_width=1, head_length=1,
                  fc='red', ec='red',
                  linewidth=1.5, zorder=15)
        plt.axis("equal")
        plt.grid(False)
        plt.pause(0.01)

        
    def _get_local_patch(self, CM_SIZE=CM_SIZE):
        robot_pos = np.array(self.gps.getValues()[:2])
        mx = int(robot_pos[0] / self.map_res)
        my = int(robot_pos[1] / self.map_res)
    
        half = CM_SIZE // 2
        padded = np.pad(self.global_map, pad_width=half, mode='constant', constant_values=1)
        patch = padded[my:my+CM_SIZE, mx:mx+CM_SIZE]
    
        # Keep original values including -1 for unknown, clip others to [0, 1]
        patch = np.where(patch == -1, -1.0, np.clip(patch, 0.0, 1.0))
        return patch.astype(np.float32)
   
# ======================================================================
# Training / evaluation entry-point
# ======================================================================
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
import torch.nn.functional as F

class CostmapEncoder(BaseFeaturesExtractor):
    def __init__(self, observation_space: spaces.Dict, features_dim=64):
        super().__init__(observation_space, features_dim)

        cm_size = observation_space["costmap"].shape[-1]
        flat_dim = observation_space["flat"].shape[0]

        self.cnn_output_dim = 32

        # Costmap CNN
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(16 * cm_size * cm_size, self.cnn_output_dim),
            nn.ReLU()
        )

        self.attn_dim = flat_dim + self.cnn_output_dim
        self.query = nn.Linear(self.attn_dim, self.attn_dim)
        self.key = nn.Linear(self.attn_dim, self.attn_dim)
        self.value = nn.Linear(self.attn_dim, self.attn_dim)

        # Output MLP
        self.linear = nn.Sequential(
            nn.Linear(self.attn_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        flat = observations["flat"]                              # (B, F)
        costmap = observations["costmap"].unsqueeze(1)           # (B, 1, H, W)

        cost_feat = self.cnn(costmap)                            # (B, C)
        merged = th.cat([flat, cost_feat], dim=1)                # (B, F+C)

        # Gated Attention: Q, K, V from merged
        Q = self.query(merged)                                   # (B, D)
        K = self.key(merged)                                     # (B, D)
        V = self.value(merged)                                   # (B, D)

        attn_scores = th.bmm(Q.unsqueeze(1), K.unsqueeze(2)) / (self.attn_dim ** 0.5)  # (B,1,1)
        attn_weights = th.sigmoid(attn_scores)                   # (B,1,1), gating not softmax
        attn_out = attn_weights.squeeze(-1) * V                  # (B,D), elementwise

        return self.linear(attn_out)                             # (B, features_dim)

# ───────── Training Parameters ─────────────────────────────
TRAIN = True
TIMESTEPS = 2_000_000
NUM_EPISODES = 5
MODEL_PATH = f"ppo_harsh_{HORIZON}.zip"
LOG_DIR = "./logs_ppo"

# ───────── Environment Setup ─────────────────────────────
env = Monitor(HarshEnv(seed=SEED, train=True))
device = "cuda" if th.cuda.is_available() else "cpu"
device = "cpu"
# ───────── Policy Configuration ─────────────────────────────
policy_kwargs = dict(
    features_extractor_class=CostmapEncoder,
    features_extractor_kwargs=dict(features_dim=64)
)

# ───────── Training / Evaluation Logic ─────────────────────────────
if TRAIN:
    cb = CheckpointCallback(
        save_freq=100_000,
        save_path="./checkpoints",
        name_prefix="ppo_harsh"
    )

    if os.path.exists(MODEL_PATH):
        print(f"✅ Loading existing model from: {MODEL_PATH}")
        model = PPO.load(MODEL_PATH, env=env, device=device)
    else:
        print("🚀 No saved model found. Starting training from scratch.")
        model = PPO(
            policy=MlpPolicy,
            env=env,
            policy_kwargs=policy_kwargs,
            tensorboard_log=LOG_DIR,
            verbose=1,
            device=device
        )

    model.learn(total_timesteps=TIMESTEPS, callback=cb)
    model.save(MODEL_PATH)

else:
    model = PPO.load(MODEL_PATH, env=env, device=device)
    success = total_r = harsh = steps = 0

    for _ in range(NUM_EPISODES):
        obs, _ = env.reset()
        done = truncated = False
        ep_r = 0
        while not (done or truncated):
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, truncated, _ = env.step(action)
            ep_r += r
        total_r += ep_r
        steps += env.steps
        harsh += env.visitHarsh
        success += env.reached_goal

    print(f"✅ Success Rate: {success / NUM_EPISODES * 100:.1f}%")
    print(f"📈 Avg Reward: {total_r / NUM_EPISODES:.1f}")
    print(f"⏱️ Avg Steps: {steps / NUM_EPISODES:.1f}")
    print(f"🔥 Avg Harsh Visits: {harsh / NUM_EPISODES:.1f}")
