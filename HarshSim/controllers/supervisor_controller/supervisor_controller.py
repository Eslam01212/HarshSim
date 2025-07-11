from controller import Supervisor, Keyboard, Receiver
import numpy as np
from threading import Thread
import random
import time,os
import runpy  # For running GUI.py
from controller import Robot, Motor, DistanceSensor, GPS, Compass, Receiver, Emitter

def run_gui():
    import GUI  # Runs GUI.py (but use import, not run_path)

# Start GUI in background thread
gui_thread = Thread(target=run_gui)
gui_thread.start()

# Wait until the continue.txt file is created
while not os.path.exists("continue.txt"):
    print("Waiting for user to press continue in GUI...")
    time.sleep(1)

# Once continue.txt exists, you can proceed
print("User pressed continue! Proceeding with rest of the code...")
os.remove("continue.txt")

# === Run GUI to get user-defined parameters ===
#runpy.run_path("GUI.py")  # Make sure this generates 'parms.txt' and optionally 'map.txt'
def reset_robot_pose(children_field, robot_nodes, terrain_map, terrain_size):
    robot_node = children_field.getMFNode(robot_nodes[0])
    translation_field = robot_node.getField("translation")
    rotation_field = robot_node.getField("rotation")

    i, j = get_random_zero_cell(terrain_map)
    x, y = i * terrain_size, j * terrain_size

    translation_field.setSFVec3f([x, y, 0.0])
    rotation_field.setSFRotation([0, 1, 0, 0])  # Facing forward
    robot_node.resetPhysics()


def is_testing_mode():
    try:
        with open("mode.txt", "r") as f:
            for line in f:
                if line.strip().startswith("mode"):
                    key, value = map(str.strip, line.strip().split("="))
                    return value.lower() == "test"
    except FileNotFoundError:
        print("mode.txt not found. Assuming training mode is OFF.")
    return False


def check_for_reset_signal():
    if reset_receiver.getQueueLength() > 0:
        try:
            message = reset_receiver.getString()
            reset_receiver.nextPacket()
            if message.strip().lower() == "reset":
                #print(">>> Received reset signal.")
                clearEnv(children_field, terrain_nodes, robot_nodes, human_nodes)
                reset_receiver.nextPacket()
                return True
        except Exception as e:
            print("Error processing message:", e)
            reset_receiver.nextPacket()
    return False

# === Terrain label-value map ===
value_map = {
    "Debris": 0.04,
    "Water": 0.1,
    "Fire": 0.2,
    "Unknown": 0.8,
    "Wall": 1.0
}

# === Helper functions ===
def load_config(path):
    config = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and ":" in line:
                k, v = map(str.strip, line.split(":", 1))
                try:
                    config[k] = float(v)
                except ValueError:
                    config[k] = v
    return config

def generate_random_map(rows, cols, config, value_map):
    print("Generate Random Map")
    total_cells = rows * cols
    terrain_flat = []
    for label, percent in config.items():
        if isinstance(percent, float):
            value = value_map.get(label, 0.0)
            count = int(percent * total_cells + 0.5)
            terrain_flat.extend([value] * count)
    while len(terrain_flat) < total_cells:
        terrain_flat.append(0.0)
    random.shuffle(terrain_flat)
    return [terrain_flat[i*cols:(i+1)*cols] for i in range(rows)]

def get_random_zero_cell(terrain_map):
    zero_cells = [(i, j) for i in range(len(terrain_map)) for j in range(len(terrain_map[0])) if terrain_map[i][j] == 0.0]
    if not zero_cells:
        raise ValueError("No zero-value cell available for spawning.")
    return random.choice(zero_cells)

def height_to_color(height):
    terrain_rgb = {
        0.0: (1.0, 1.0, 1.0),
        0.04: (0.5, 0.5, 0.5),
        0.1: (0.0, 0.0, 1.0),
        0.2: (1.0, 1.0, 0.0),
        0.8: (1.0, 0.0, 0.0),
        1.0: (0.0, 0.0, 0.0)
    }
    return f"{terrain_rgb.get(round(height, 2), (0.0, 1.0, 0.0))[0]:.2f} {terrain_rgb.get(round(height, 2), (0.0, 1.0, 0.0))[1]:.2f} {terrain_rgb.get(round(height, 2), (0.0, 1.0, 0.0))[2]:.2f}"

def clearEnv(children_field, terrain_nodes, robot_nodes, human_nodes):
    for index in sorted(terrain_nodes + human_nodes, reverse=True):  # DO NOT remove robot_nodes
        children_field.removeMF(index)
    terrain_nodes.clear()
    human_nodes.clear()
    
# === Webots supervisor init ===
supervisor = Supervisor()
test = False
reset_receiver = supervisor.getDevice("receiver")
reset_receiver.enable(32)
reset_receiver.setChannel(2)

terrain_nodes = []
robot_nodes = []
human_nodes = []

root = supervisor.getRoot()
children_field = root.getField("children")

# === Config ===
config = load_config("parms.txt")
test = config.get("test", 0) == 1  # Add test flag if used
terrain_size = 1.0
cell_size = 1

with open("Dim.txt", "r") as f:
    dim_line = f.read().strip()
    rows, cols, cell_size = dim_line.split(",")
    rows = int(rows)
    cols = int(cols)
    cell_size = float(cell_size)
# Spawn robot
terrain_map = np.loadtxt("map.txt").tolist()
i, j = get_random_zero_cell(terrain_map)
x, y = i * terrain_size, j * terrain_size

robot_string = f"""
Pioneer3at_custom {{
  supervisor TRUE
  translation {x} {y} 0
  rotation 0 1 0 0
  controller "HarshEnv"
}}
"""
index = children_field.getCount()
children_field.importMFNodeFromString(-1, robot_string)
robot_nodes.append(index)

# === Main reset loop ===
test_flag = True
while supervisor.step(32) != -1 :
  if check_for_reset_signal() == True:
    check_for_reset_signal() == False
    if is_testing_mode():
        terrain_map = np.loadtxt("map.txt").tolist()
        test_flag = False
        print("Testing mood.......")
        rows = len(terrain_map)
        cols = len(terrain_map[i])
    else:
        print("Training mood.......")
        terrain_map = generate_random_map(rows, cols,  config, value_map)
    
    
    for i in range(rows):
        for j in range(cols):
            height = terrain_map[i][j]
            if height == 1:
                box_height = height/2 # Convert terrain value to height
                color = height_to_color(height)
                obstacle_string = f"""
                Solid {{
                  translation {i * terrain_size} {j * terrain_size} {box_height / 2}
                  rotation 0 1 0 0
                  name "Obstacle_{i}_{j}"
                  children [
                    Shape {{
                      appearance Appearance {{
                        material Material {{
                          diffuseColor 1,1,1
                        }}
                      }}
                      geometry Box {{
                        size 1 1 {box_height}
                      }}
                    }}
                  ]
                  boundingObject Box {{
                    size 1 1 {box_height}
                  }}
                }}
                """
                index = children_field.getCount()
                children_field.importMFNodeFromString(-1, obstacle_string)
                terrain_nodes.append(index)
                
    for i in range(rows):
        for j in range(cols):
            height = terrain_map[i][j]
            if height>0 and height != 1:
                color = height_to_color(height)
                terrain_string = f"""
                UnevenTerrain {{
                  translation {i * terrain_size} {j * terrain_size} {-height/2}
                  size {cell_size} {cell_size} {height}
                  appearance Appearance {{
                    material Material {{
                      diffuseColor {color}
                    }}
                  }}
                }}
                """
                index = children_field.getCount()
                children_field.importMFNodeFromString(-1, terrain_string)
                terrain_nodes.append(index)
                
    # Spawn human (goal marker)
    i, j = get_random_zero_cell(terrain_map)
    x, y = i * terrain_size, j * terrain_size
    human_string = f"""
    Robot {{
      translation {x} {y} .1
      rotation 0 0 0 1.57  # Optional: rotate if needed
      name "human"
      controller "humanGPS"
      boundingObject NULL
      children [
        GPS {{
          name "h_gps"
        }}
        Emitter {{
          channel 1
        }}
        Shape {{
          appearance Appearance {{
            texture ImageTexture {{
              url ["goal_pin.png"]
            }}
          }}
          geometry Plane {{
            size 2 2
          }}
        }}
      ]
    }}
    """

    index = children_field.getCount()
    children_field.importMFNodeFromString(-1, human_string)
    human_nodes.append(index)
    
    reset_robot_pose(children_field, robot_nodes, terrain_map, terrain_size)
    terrain_map = np.asarray(terrain_map, dtype=float)
    np.savetxt("terrain_map.txt", terrain_map, fmt='%.2f')  # Save with 2 decimal places
