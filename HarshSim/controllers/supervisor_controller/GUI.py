import tkinter as tk
from tkinter import messagebox, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np
import tkinter.font as tkFont
from matplotlib.figure import Figure

plt.rcParams['toolbar'] = 'none'

# ========== Global Setup ==========
root = tk.Tk()
root.title("AI-Based Simulator for Mobile Robot Operation on Unknown Harsh Environment")
root.configure(bg="#2B2B2B")

default_font = tkFont.nametofont("TkDefaultFont")
default_font.configure(family="Helvetica", size=14)
bold_font = ("Helvetica", 14, "bold")

terrain_cost = {0: 0, 0.04: 4, 0.1: 1, 0.2: 2, 0.8: 8, 1: 9}
color_dict = {0: "white", 1: "gray", 2: "blue", 3: "yellow", 4: "red", 5: "black"}
terrain_percent_entries = {}
param_entries = {}

terrain_map = None
percent_frame = None


# ========== Core Functions ==========
def on_continue():
    try:
        rows = int(entry_rows.get())
        cols = int(entry_cols.get())
        Cell_size = entry_Cell_size.get().strip()
    except ValueError:
        messagebox.showerror("Input Error", "Rows and Cols must be integers.")
        return

    with open("continue.txt", "w") as f:
        f.write("GO")

    with open("Dim.txt", "w") as f:
        f.write(f"{rows},{cols},{Cell_size}")

    print(f"User pressed continue. Dimensions saved: {rows}x{cols}, Cell_size: {Cell_size}")


def showEnv():
    try:
        rows = int(entry_rows.get())
        cols = int(entry_cols.get())
        global terrain_map
        terrain_map = draw_terrain_map_gui(rows, cols)
        ax_movement.clear()

        if hasattr(showEnv, "cbar") and showEnv.cbar:
            showEnv.cbar.remove()
            showEnv.cbar = None

        img = ax_movement.imshow(np.array(terrain_map), cmap='viridis', interpolation='nearest')
        showEnv.cbar = fig_movement.colorbar(img, ax=ax_movement, orientation='vertical')
        showEnv.cbar.set_label('Terrain Cost')

        ax_movement.set_title("Robot's Path on Unknown Harsh Environment")
        ax_movement.set_xlabel("X Axis")
        ax_movement.set_ylabel("Y Axis")
        canvas_movement.draw()
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid rows and cols numerical values.")


def train_model():
    messagebox.showinfo("Mode", "Training Started")
    with open("mode.txt", "w") as f:
        f.write("mode = train\n")


def test_model():
    with open("mode.txt", "w") as f:
        f.write("mode = test\n")
    messagebox.showinfo("Mode", "Testing Started")


def save_parms():
    config = {param: entry.get() for param, entry in param_entries.items()}
    config['terrain_percentages'] = {label: entry.get() for label, entry in terrain_percent_entries.items()}
    filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if filepath:
        try:
            with open(filepath, 'w') as f:
                f.write("# RL Configuration File\n")
                f.write(f"rl_method: {config.get('rl_method', 'N/A')}\n\n")
                f.write("# RL Parameters\n")
                for key in param_entries:
                    f.write(f"{key}: {config[key]}\n")
                f.write("\n# Terrain Percentages\n")
                for label, value in config['terrain_percentages'].items():
                    f.write(f"{label}: {value}\n")
            messagebox.showinfo("Saved", f"Configuration saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save config:\n{str(e)}")


def save_terrain():
    if not terrain_map:
        messagebox.showwarning("Warning", "No terrain to save!")
        return
    filepath = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if filepath:
        try:
            np.savetxt(filepath, np.array(terrain_map), fmt="%.2f")
            messagebox.showinfo("Saved", f"Terrain saved to:\n{filepath}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save map:\n{str(e)}")


def load_terrain():
    global terrain_map
    filepath = filedialog.askopenfilename(filetypes=[("Text files", "*.txt")])
    if not filepath:
        return
    try:
        loaded_map = np.loadtxt(filepath)
        rows, cols = loaded_map.shape
        terrain_map = draw_terrain_map_gui(rows, cols)
        cell_size = int(500 / max(rows, cols))

        for row in range(rows):
            for col in range(cols):
                val = loaded_map[row, col]
                terrain_map[row][col] = val
                for key, value in terrain_cost.items():
                    if np.isclose(val, key):
                        color = color_dict[list(terrain_cost.keys()).index(key)]
                        terrain_map_canvas.create_rectangle(
                            col * cell_size, row * cell_size,
                            (col + 1) * cell_size, (row + 1) * cell_size,
                            fill=color, outline="black"
                        )
        messagebox.showinfo("Loaded", "Terrain map loaded successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load map:\n{str(e)}")


def draw_terrain_map_gui(rows, cols):
    global terrain_map_canvas
    cell_size = int(500 / max(rows, cols))
    terrain_map = [[0 for _ in range(cols)] for _ in range(rows)]

    terrain_keys = [0, 1, 2, 3, 4, 5]
    terrain_values = [0, 0.04, 0.1, 0.2, 0.8, 1]
    key_to_value = dict(zip(terrain_keys, terrain_values))
    selected_key = tk.IntVar(value=0)

    for widget in frame_env_plot.winfo_children():
        widget.destroy()

    terrain_map_canvas = tk.Canvas(frame_env_plot, width=cols * cell_size, height=rows * cell_size, bg="white")
    terrain_map_canvas.grid(row=0, column=1, rowspan=10, padx=10, pady=5)

    def draw_cell(event):
        col = event.x // cell_size
        row = event.y // cell_size
        if 0 <= row < rows and 0 <= col < cols:
            terrain_id = selected_key.get()
            value = key_to_value[terrain_id]
            terrain_map[row][col] = value
            terrain_map_canvas.create_rectangle(
                col * cell_size, row * cell_size,
                (col + 1) * cell_size, (row + 1) * cell_size,
                fill=color_dict[terrain_id],
                outline="black"
            )

    terrain_map_canvas.bind("<Button-1>", draw_cell)
    terrain_map_canvas.bind("<B1-Motion>", draw_cell)

    radio_frame = tk.Frame(frame_env_plot)
    radio_frame.grid(row=0, column=0, padx=10, pady=5, sticky="n")

    name_dict = {
        0: "Flat Ground", 1: "Debris", 2: "Water",
        3: "Fire", 4: "Unknown", 5: "Wall"
    }

    for terrain_id in terrain_keys:
        tk.Radiobutton(
            radio_frame,
            text=f"{name_dict[terrain_id]} ({key_to_value[terrain_id]})",
            variable=selected_key,
            value=terrain_id,
            font=default_font
        ).pack(anchor="w")

    tk.Button(radio_frame, text="Done", font=default_font,
              command=lambda: [print(np.array(terrain_map)), calculate_percentages()]).pack(pady=5)

    return terrain_map


def calculate_percentages():
    global percent_frame
    if not terrain_map:
        return

    total_cells = len(terrain_map) * len(terrain_map[0])
    terrain_labels = [(0.0, "Flat Ground"), (0.04, "Debris"), (0.1, "Water"), (0.2, "Fire"), (1.0, "Wall"), (0.8, "Unknown")]
    percentages = {}

    for value, label in terrain_labels:
        count = sum(1 for row in terrain_map for val in row if np.isclose(val, value))
        percentages[label] = (count / total_cells) * 100

    predefined_percentage = 100.0 - percentages["Unknown"]

    if percent_frame:
        percent_frame.destroy()

    percent_frame = tk.LabelFrame(frame_env_plot, text="Terrain Composition (%)", font=bold_font, padx=10, pady=5)
    percent_frame.grid(row=2, column=0, padx=10, pady=5, sticky="nw")

    display_labels = ["Flat Ground", "Debris", "Water", "Fire", "Wall"]
    for idx, label in enumerate(display_labels):
        tk.Label(percent_frame, text=f"{label}: {percentages[label]:.2f}%", font=default_font).grid(row=idx, column=0, sticky="w", pady=2)

    tk.Label(percent_frame, text=f"Unknown: {percentages['Unknown']:.2f}%", font=default_font).grid(row=len(display_labels), column=0, sticky="w", pady=2)
    tk.Label(percent_frame, text=f"Predefined (non-unknown): {predefined_percentage:.2f}%", font=default_font).grid(row=len(display_labels) + 1, column=0, sticky="w", pady=2)


# ========== GUI Layout ==========
frame_EnvSpecs = tk.LabelFrame(root, text="Unknown Harsh Specs", font=bold_font, padx=10, pady=5)
frame_EnvSpecs.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

frame_env_plot = tk.Frame(root)
frame_env_plot.grid(row=1, column=0,  columnspan=3, padx=10, pady=5, sticky="nsew")

frame_train = tk.LabelFrame(root, text="Environment Parameters for Training", font=bold_font, padx=10, pady=5)
frame_train.grid(row=0, column=1, columnspan=2, padx=10, pady=5, sticky="nsew")

entry_rows = tk.Entry(frame_EnvSpecs, font=default_font); entry_rows.insert(0, "10"); entry_rows.grid(row=0, column=1)
entry_cols = tk.Entry(frame_EnvSpecs, font=default_font); entry_cols.insert(0, "10"); entry_cols.grid(row=1, column=1)
entry_Cell_size = tk.Entry(frame_EnvSpecs, font=default_font); entry_Cell_size.insert(0, ".5"); entry_Cell_size.grid(row=2, column=1)

tk.Label(frame_EnvSpecs, text="Rows", font=default_font).grid(row=0, column=0, sticky="w" , pady=5)
tk.Label(frame_EnvSpecs, text="Cols", font=default_font).grid(row=1, column=0, sticky="w" , pady=5)
tk.Label(frame_EnvSpecs, text="Cell size", font=default_font).grid(row=2, column=0, sticky="w", pady=5)

tk.Button(frame_EnvSpecs, text="Design Harsh Environment", font=default_font, bg="#607D8B", fg="white", command=showEnv).grid(row=12, column=0, padx=5, pady=(5,5))
tk.Button(frame_EnvSpecs, text="Save Terrain", font=default_font, bg="#3F51B5", fg="white", command=save_terrain).grid(row=13, column=0, sticky='w',padx=5, pady=5)
tk.Button(frame_EnvSpecs, text="Load Terrain", font=default_font, bg="#009688", fg="white", command=load_terrain).grid(row=14, column=0, sticky='w', padx=5, pady=5)
tk.Button(frame_EnvSpecs, text="Testing Mode", font=default_font, bg="#f44336", fg="white", command=test_model).grid(row=15, column=0, sticky='w', padx=5, pady=(5,5))

tk.Button(frame_train, text="Save Parms", font=default_font, bg="#3F51B5", fg="white", command=save_parms).grid(row=9, column=0, padx=5, pady=(5,5))
tk.Button(frame_train, text="Training Mode", font=default_font, bg="#f44336", fg="white", command=train_model).grid(row=10, column=0, padx=5, pady=5)
tk.Button(frame_train, text="Continue", font=default_font, bg="#FF9800", fg="white", command=on_continue).grid(row=11, column=1, sticky="e", padx=5, pady=(5,1))

tk.Label(frame_train, text="Terrain %", font=bold_font).grid(row=0, column=0, columnspan=2, sticky="w", pady=2)
terrain_types = {"Debris": 0.0, "Water": 0.0, "Fire": 0.0, "Unknown": 0.0, "Wall": 0.0}
for idx, (label, default_val) in enumerate(terrain_types.items()):
    tk.Label(frame_train, text=label, font=default_font).grid(row=idx+1, column=0, sticky="e", pady=5)
    entry = tk.Entry(frame_train, font=default_font); entry.insert(0, str(default_val))
    entry.grid(row=idx+1, column=1, sticky="ew")
    terrain_percent_entries[label] = entry

fig_movement = Figure(figsize=(5, 5))
ax_movement = fig_movement.add_subplot(111)
canvas_movement = FigureCanvasTkAgg(fig_movement, master=frame_env_plot)
canvas_movement.get_tk_widget().grid(row=1, column=0, pady=5)

plt.close('all')
root.mainloop()
