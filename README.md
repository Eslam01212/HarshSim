
# HarshSim — Simulator for AI‑Based Mobile Robot Operation in Harsh, Unpredictable Environments

> **Webots‑based simulator + Gymnasium environment** for training and evaluating DRL navigation policies (PPO / Recurrent PPO) in adverse terrains (debris, water, fire, walls, unknown). Includes a **Tkinter GUI** to build maps and configure experiments, a **Supervisor** that spawns the scene (robot + optional moving human target), and a **robot controller** that exposes a `Gymnasium` API with a costmap encoder.

![Goal pin](HarshSim/controllers/supervisor_controller/SIMUI.png)

---

## ✨ Highlights

- **End‑to‑end DRL**: `HarshEnv` implements a Gymnasium `Env` that runs *inside Webots* using the Supervisor API.
- **SB3‑ready**: Works with Stable‑Baselines3 (`PPO`, `RecurrentPPO` via `sb3_contrib`), including checkpointing and Monitor logging.
- **Terrain & hazards**: Parametric generation of harsh regions (Debris / Water / Fire / Unknown / Walls) with adjustable percentages.
- **Costmap observation**: Local `CM_SIZE × CM_SIZE` costmap patch + LiDAR beams + goal/heading features.
- **GUI workflow**: Design maps, set parameters, toggle *Training* vs *Testing*, then run with one click.
- **Human target (optional)**: A separate controller (`humanGPS`) can broadcast a moving target via Webots `Emitter/Receiver`.

---

## 🗃️ Repository Layout

```
HarshSim/
  controllers/
    HarshEnv/                 # Gymnasium env + SB3 training/eval loop (Python controller)
      HarshEnv.py
    supervisor_controller/    # Webots Supervisor + Tkinter GUI integration
      supervisor_controller.py
      GUI.py                  # Map/params UI (writes Dim.txt, parms.txt, mode.txt, etc.)
      map.txt                 # Example terrain map (ASCII)
      terrain_map*.txt        # Saved terrain maps
      goal_pin2.png           # UI icon
    humanGPS/
      humanGPS.py             # Emits human GPS pose (channel 1) for goal following
  protos/
    Pioneer3at_custom.proto   # Custom Pioneer 3‑AT robot with sensors
    Pioneer3DistanceSensor.proto
  worlds/
    map.wbt                   # Webots world (R2025a)
LICENSE
README.md
```

Key text files written/consumed at runtime by the GUI/Supervisor:
- `Dim.txt` — rows, cols, cell size for the grid (written by GUI).
- `parms.txt` — experiment parameters (written by GUI).
- `mode.txt` — "mode = train" or "mode = test" (written by GUI).
- `map.txt` / `terrain_map.txt` — terrain cost grid (load/save).

---

## 🔧 Prerequisites

- **Webots** R2025a or newer (as indicated by `#VRML_SIM R2025a` in the world file).
- **Python 3.10+** (match Webots Python command to your environment).
- **Packages**: `numpy`, `matplotlib`, `gymnasium`, `stable-baselines3`, `sb3-contrib`, `torch`, `opencv-python`, `tkinter` (standard library), and Webots' `controller` module.

> **Tip — Webots Python:** in Webots go to **Tools → Preferences → Python command** and point it to the Python inside your virtualenv where you install the requirements below.

### Example (Linux/macOS/WSL)

```bash
# 1) Clone
git clone https://github.com/Eslam01212/HarshSim.git
cd HarshSim

# 2) Create env & install deps
python3 -m venv .venv
source .venv/bin/activate  # (on Windows: .venv\Scripts\activate)
pip install --upgrade pip

# Install PyTorch suited to your OS/CUDA (see https://pytorch.org/get-started/locally/)
pip install torch

# Core dependencies
pip install gymnasium stable-baselines3 sb3-contrib numpy matplotlib opencv-python

# (Webots provides the `controller` module; no pip install needed)
```

> If you prefer pinning, a minimal `requirements.txt` could be:
>
> ```text
> gymnasium>=0.29
> stable-baselines3>=2.3
> sb3-contrib>=2.3
> numpy>=1.24
> matplotlib>=3.7
> opencv-python>=4.8
> # Install torch separately for your platform
> ```

---

## ▶️ Quick Start (GUI Workflow)

1. **Open the world** in Webots: `HarshSim/worlds/map.wbt`.
2. Ensure the **Simulation** is set to *Real‑time* and press **Play**.
3. A **GUI** window appears:
   - Set **Rows**, **Cols**, **Cell size**.
   - Optionally **Load** an existing `map.txt` or draw/configure terrain percentages.
   - Click **Save Parms** to write `parms.txt` (optional).
   - Choose **Training Mode** or **Testing Mode** (writes `mode.txt`).
   - Click **Continue** — this writes `continue.txt` and `Dim.txt` which the Supervisor waits for.
4. The Supervisor (`supervisor_controller.py`) spawns the robot (`Pioneer3at_custom`) and, if enabled, a **human** (target) with `humanGPS` (emits GPS via Webots Emitter on channel 1).
5. The robot runs the **`HarshEnv`** Python controller:
   - By default, `HarshEnv.py` is configured to **train** a PPO agent (`TRAIN = True`).
   - Checkpoints are saved under `./checkpoints` and the latest model at `MODEL_PATH` (see code).

> **Switch to evaluation:** Open `HarshSim/controllers/HarshEnv/HarshEnv.py` and set `TRAIN = False`. With a trained `MODEL_PATH` present, the script runs evaluation over `NUM_EPISODES` and prints Success Rate, Avg Reward, etc.

---

## 🧠 Environment Interface (HarshEnv)

- **Action space**: `Box(low=-1, high=1, shape=(2 * HORIZON,))` — linear & angular velocity pairs over a receding horizon.
- **Observation space** (Gymnasium `Dict`):
  - `"flat"`: concatenated features (downsampled LiDAR beams `LIDAR_ANGLES`, distance/heading, previous action history, etc.).
  - `"costmap"`: local `CM_SIZE × CM_SIZE` patch centered on the robot.
- **Sensors**: Webots LiDAR, GPS, Compass; optional human goal via `Receiver`.
- **Reward** (high‑level): progress to goal, obstacle/terrain penalties, smoothness/turning, and episode termination on success/collision/timeout.
- **Logging & checkpoints**: SB3 `Monitor` + `CheckpointCallback` (path/interval configurable in code).

---

## 📊 Training / Evaluation from Code

Although the typical entry is through Webots + GUI, `HarshEnv.py` contains a self‑contained SB3 loop:

```python
# Inside HarshEnv.py
TRAIN = True            # set False for evaluation
TIMESTEPS = 2_000_000
MODEL_PATH = f"ppo_harsh_{HORIZON}.zip"

env = Monitor(HarshEnv(seed=SEED, train=TRAIN))
model = PPO("MultiInputPolicy", env, policy_kwargs=dict(
    features_extractor_class=CostmapEncoder,
    features_extractor_kwargs=dict(features_dim=64)
))
model.learn(total_timesteps=TIMESTEPS, callback=CheckpointCallback(...))
model.save(MODEL_PATH)
```

For **Recurrent PPO**, see the `sb3_contrib.RecurrentPPO` import and adapt `policy` and `policy_kwargs` accordingly.

---

## 🧩 Tips & Troubleshooting

- **Webots Python path**: If `ModuleNotFoundError: controller` appears, set Webots **Python command** to your venv interpreter (Tools → Preferences).
- **GUI on headless/containers**: Tkinter requires a display (X server). On WSL/containers, enable X11 forwarding or use a desktop session.
- **Torch install**: Use the official selector to match your CUDA/OS. CPU‑only works too.
- **Real‑time vs. Fast**: For stable RL training, keep Webots in real‑time; too fast can desynchronize GUI/IO.
- **Files not found**: The Supervisor waits for `continue.txt` and reads `Dim.txt`/`parms.txt`/`map.txt`. Use the GUI buttons to generate them.

---

## 📄 License

This project is licensed under the **Apache 2.0** License. See [LICENSE](LICENSE).

---

## 📚 Citation

If you use HarshSim in academic work, please cite:

```bibtex
@misc{harshsim2025,
  title  = {HarshSim: Simulator for AI-Based Mobile Robot Operation in Harsh, Unpredictable Environments},
  author = {Eslam M. Mohamed and collaborators},
  year   = {2025},
  howpublished = {GitHub repository},
  url    = {https://github.com/Eslam01212/HarshSim}
}
```

---

## 🤝 Acknowledgments

Built on top of:
- Webots — https://cyberbotics.com/
- Gymnasium — https://gymnasium.farama.org/
- Stable‑Baselines3 — https://stable-baselines3.readthedocs.io/
- SB3 Contrib — https://sb3-contrib.readthedocs.io/
- PyTorch — https://pytorch.org/
