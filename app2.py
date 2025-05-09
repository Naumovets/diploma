import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.spatial.distance import cdist

data = None
current_frame = 0
fig = None
ax = None
canvas = None
time_label = None
info_label = None
prev_data = []

# TODO
# вывод ускорения и скорости

def select_file():
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        load_data(file_path)
        show_main_window()

def load_data(file_path):
    global data, current_frame
    data = pd.read_csv(file_path)
    current_frame = data['frame'].min()

def show_main_window():
    global fig, ax, canvas, time_label, info_label

    main_window = tk.Toplevel(root)
    main_window.title("3D Визуализация плоскостей")
    main_window.geometry("1000x800")

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    canvas = FigureCanvasTkAgg(fig, master=main_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    info_frame = tk.Frame(main_window)
    info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)

    info_label = tk.Label(info_frame, text="Информация о плоскостях:", justify=tk.LEFT)
    info_label.pack()

    control_frame = tk.Frame(main_window)
    control_frame.pack(side=tk.BOTTOM, fill=tk.X)

    time_slider = ttk.Scale(
        control_frame,
        from_=data['frame'].min(),
        to=data['frame'].max(),
        orient=tk.HORIZONTAL,
        command=on_frame_change
    )
    time_slider.pack(side=tk.LEFT, fill=tk.X, expand=1, padx=5, pady=5)

    time_label = tk.Label(control_frame, text=f"Кадр: {current_frame}")
    time_label.pack(side=tk.LEFT, padx=5, pady=5)

    rotate_frame = tk.Frame(control_frame)
    rotate_frame.pack(side=tk.LEFT, padx=5, pady=5)

    buttons = [
        ("←", 'left'), ("→", 'right'),
        ("↑", 'up'), ("↓", 'down')
    ]

    for text, direction in buttons:
        btn = tk.Button(rotate_frame, text=text, command=lambda d=direction: rotate(d))
        btn.pack(side=tk.LEFT, padx=2)

    update_plot()

def update_plot():
    global info_label    
    filtered_data = data[data['frame'] == current_frame]
    
    planes = []
    normals = []
    centers = []
    rotations = []
    
    for _, row in filtered_data.iterrows():
        plane, normal = plot_rotated_plane(
            ax, 
            row['x'], row['y'], row['z'],
            row['xy'], row['yz'], row['xz']
        )
        rotations.append([row['xy'], row['yz'], row['xz']])
        planes.append(plane)
        normals.append(normal)
        centers.append([row['x'], row['y'], row['z']])

    ax.set(xlim=(-0.5,0.5), ylim=(-0.5,0.5), zlim=(0,1))
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    info_text = calculate_plane_info(centers, normals, rotations)
    info_label.config(text=info_text)
    
    canvas.draw()


def plot_rotated_plane(ax, x, y, z, xy, yz, xz, size=0.05, color='b', alpha=1):
    u = np.linspace(-size, size, 10)
    v = np.linspace(-size, size, 10)
    u, v = np.meshgrid(u, v)
    points = np.stack([u, v, np.zeros_like(u)], axis=-1)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(xy), -np.sin(xy)],
                   [0, np.sin(xy), np.cos(xy)]])
    
    Ry = np.array([[np.cos(yz), 0, np.sin(yz)],
                   [0, 1, 0],
                   [-np.sin(yz), 0, np.cos(yz)]])
    
    Rz = np.array([[np.cos(xz), -np.sin(xz), 0],
                   [np.sin(xz), np.cos(xz), 0],
                   [0, 0, 1]])
    
    R = Rz @ Ry @ Rx
    
    points = points @ R.T
    points += [x, y, z]
    
    xx, yy, zz = points[...,0], points[...,1], points[...,2]
    
    plane = ax.plot_surface(xx, yy, zz, color=color, alpha=alpha, edgecolor='none')
    normal = R @ [0, 0, 1]
    return plane, normal

def calculate_plane_info(centers, normals, rotations):
    if len(centers) < 2:
        return "Недостаточно плоскостей для расчетов"
    
    info_text = "Информация о плоскостях:\n"
    
    for i, (center, normal) in enumerate(zip(centers, normals)):
        distance = abs(np.dot(center, normal)) / np.linalg.norm(normal)
        info_text += f"Плоскость {i+1}:\n"
        info_text += f"  Расстояние до начала: {distance:.3f}\n"
    
    for i in range(len(centers)):
        for j in range(i+1, len(centers)):
            dist = np.linalg.norm(np.array(centers[i]) - np.array(centers[j]))
            info_text += f"Расстояние между {i+1} и {j+1}: {dist:.3f}\n"
        
    # for i in range(len(rotations)):
    #     info_text += f"плоскость {i+1} xy: {np.rad2deg(rotations[i][0]):.2f} yz: {np.rad2deg(rotations[i][1]):.2f}; zx: {np.rad2deg(rotations[i][2]):.2f}\n"
    
    return info_text

def on_frame_change(val):
    global current_frame
    current_frame = int(float(val))
    time_label.config(text=f"Кадр: {current_frame}")
    update_plot()

def rotate(direction):
    elev = ax.elev
    azim = ax.azim
    
    if direction == 'left': azim -= 10
    elif direction == 'right': azim += 10
    elif direction == 'up': elev += 10
    elif direction == 'down': elev -= 10
    
    ax.view_init(elev=elev, azim=azim)
    canvas.draw()

root = tk.Tk()
root.title("Выбор файла")
root.geometry("300x100")

select_btn = tk.Button(root, text="Выбрать файл CSV", command=select_file)
select_btn.pack(expand=True, padx=20, pady=20)

root.mainloop()