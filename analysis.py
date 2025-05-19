import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import filedialog, ttk
from scipy.spatial.distance import cdist
from scipy.interpolate import CubicSpline, UnivariateSpline

data = None
current_frame = 0
fig = None
ax = None
canvas = None
time_label = None
info_label = None
prev_data = []

fps = 6
dt = 5/30

def calculate_velocities_and_accelerations(data):
    # Проверяем, что в данных есть колонка 'id'
    if 'id' not in data.columns:
        raise ValueError("Data must contain 'id' column for marker identification")
    
    # Сортируем данные по кадрам и id маркера
    data = data.sort_values(['id', 'frame'])
    
    # Получаем уникальные id маркеров
    marker_ids = data['id'].unique()
    
    # Словари для хранения результатов
    velocities = {marker_id: [] for marker_id in marker_ids}
    accelerations = {marker_id: [] for marker_id in marker_ids}
    valid_frames = {marker_id: [] for marker_id in marker_ids}
    
    # Функции для расчёта скорости и ускорения
    def calc_velocity(prev_coords, next_coords):
        return np.linalg.norm(next_coords - prev_coords)/(2*dt)
    
    def calc_acceleration(prev_coords, curr_coords, next_coords):
        return np.linalg.norm(next_coords - 2 * curr_coords + prev_coords)/(dt**2)
    
    # Обрабатываем каждый маркер отдельно
    for marker_id in marker_ids:
        marker_data = data[data['id'] == marker_id].sort_values('frame')
        frames = marker_data['frame'].values[:]
        
        # Проходим по кадрам с проверкой на последовательность
        for i in range(5, len(frames) - 5, 5):
            prev_frame = marker_data[marker_data['frame'] == frames[i-5]].iloc[0]
            curr_frame = marker_data[marker_data['frame'] == frames[i]].iloc[0]
            next_frame = marker_data[marker_data['frame'] == frames[i+5]].iloc[0]
            
            # Координаты для текущего, предыдущего и следующего кадров
            coords_curr = np.array([curr_frame['x'], curr_frame['y'], curr_frame['z']])
            coords_prev = np.array([prev_frame['x'], prev_frame['y'], prev_frame['z']])
            coords_next = np.array([next_frame['x'], next_frame['y'], next_frame['z']])
            
            # Вычисляем скорость и ускорение
            velocity = calc_velocity(coords_prev, coords_next)
            acceleration = calc_acceleration(coords_prev, coords_curr, coords_next)
            
            # Сохраняем модули величин
            velocities[marker_id].append(velocity)
            accelerations[marker_id].append(acceleration)
            valid_frames[marker_id].append(frames[i])
    
    return velocities, accelerations, valid_frames

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
    global fig_velocity, ax_velocity, canvas_velocity
    global fig_acceleration, ax_acceleration, canvas_acceleration

    main_window = tk.Toplevel(root)
    main_window.title("Анализ движения")
    main_window.geometry("1400x700")  # Оптимизированный размер окна

    # Создаем вкладки
    notebook = ttk.Notebook(main_window)
    notebook.pack(fill=tk.BOTH, expand=True)

    # Вкладка 1: Скорость с анализом
    tab_velocity = ttk.Frame(notebook)
    notebook.add(tab_velocity, text="Анализ скорости")

    # Разделяем вкладку скорости на график и аналитику
    paned_velocity = ttk.PanedWindow(tab_velocity, orient=tk.HORIZONTAL)
    paned_velocity.pack(fill=tk.BOTH, expand=True)

    # График скорости
    frame_velocity_graph = ttk.Frame(paned_velocity)
    fig_velocity = plt.figure(figsize=(9, 5))
    ax_velocity = fig_velocity.add_subplot(111)
    canvas_velocity = FigureCanvasTkAgg(fig_velocity, master=frame_velocity_graph)
    canvas_velocity.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Аналитика скорости
    frame_velocity_analysis = ttk.Frame(paned_velocity)
    text_velocity_analysis = tk.Text(frame_velocity_analysis, wrap=tk.WORD, font=('Courier New', 10))
    scroll_velocity = ttk.Scrollbar(frame_velocity_analysis, command=text_velocity_analysis.yview)
    text_velocity_analysis.configure(yscrollcommand=scroll_velocity.set)
    scroll_velocity.pack(side=tk.RIGHT, fill=tk.Y)
    text_velocity_analysis.pack(fill=tk.BOTH, expand=True)
    
    paned_velocity.add(frame_velocity_graph, weight=2)
    paned_velocity.add(frame_velocity_analysis, weight=1)

    # Вкладка 2: Ускорение с анализом
    tab_acceleration = ttk.Frame(notebook)
    notebook.add(tab_acceleration, text="Анализ ускорения")

    # Разделяем вкладку ускорения аналогично
    paned_acceleration = ttk.PanedWindow(tab_acceleration, orient=tk.HORIZONTAL)
    paned_acceleration.pack(fill=tk.BOTH, expand=True)

    # График ускорения
    frame_acceleration_graph = ttk.Frame(paned_acceleration)
    fig_acceleration = plt.figure(figsize=(9, 5))
    ax_acceleration = fig_acceleration.add_subplot(111)
    canvas_acceleration = FigureCanvasTkAgg(fig_acceleration, master=frame_acceleration_graph)
    canvas_acceleration.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    # Аналитика ускорения
    frame_acceleration_analysis = ttk.Frame(paned_acceleration)
    text_acceleration_analysis = tk.Text(frame_acceleration_analysis, wrap=tk.WORD, font=('Courier New', 10))
    scroll_acceleration = ttk.Scrollbar(frame_acceleration_analysis, command=text_acceleration_analysis.yview)
    text_acceleration_analysis.configure(yscrollcommand=scroll_acceleration.set)
    scroll_acceleration.pack(side=tk.RIGHT, fill=tk.Y)
    text_acceleration_analysis.pack(fill=tk.BOTH, expand=True)
    
    paned_acceleration.add(frame_acceleration_graph, weight=2)
    paned_acceleration.add(frame_acceleration_analysis, weight=1)

    # Рассчитываем скорости и ускорения
    velocities_dict, accelerations_dict, frames_dict = calculate_velocities_and_accelerations(data)


    for marker_id in velocities_dict.keys():
        frames = np.array(frames_dict[marker_id])
        velocities = np.array(velocities_dict[marker_id])
        accelerations = np.array(accelerations_dict[marker_id])

        if len(frames) == 0:
            continue

        # Создаем сглаживающие сплайны
        spl_velocity = UnivariateSpline(frames, velocities, k=5)
        # spl_acceleration = UnivariateSpline(frames, accelerations, k=3)
        
        # Гладкие значения для отрисовки
        smooth_frames = np.linspace(frames.min(), frames.max(), 300)
        smooth_velocities = spl_velocity(smooth_frames)
        # smooth_accelerations = spl_acceleration(smooth_frames)
        
        # Отрисовка скорости
        ax_velocity.plot(frames, velocities, 'o', markersize=5, 
                        label=f'Маркер {marker_id} - Данные')
        ax_velocity.plot(smooth_frames, smooth_velocities, '-',
                        label=f'Маркер {marker_id} - Аппроксимация')
        
        # Отрисовка ускорения
        ax_acceleration.plot(frames, accelerations, 'o', markersize=5,
                           label=f'Маркер {marker_id} - Данные')
        # ax_acceleration.plot(smooth_frames, smooth_accelerations, '-',
        #                    label=f'Маркер {marker_id} - Аппроксимация')
        
        
        # Аналитика для скорости
        velocity_stats = f"""МАРКЕР {marker_id} - СКОРОСТЬ
        {'-'*40}
        Максимальная: {velocities.max():.4f} м/с
        Минимальная:  {velocities.min():.4f} м/с
        Средняя:      {velocities.mean():.4f} м/с
        Среднее апроксимации: {smooth_velocities.mean():.4f} м/с
        Стандартное отклонение: {velocities.std():.4f}
        {'-'*40}\n\n"""
        text_velocity_analysis.insert(tk.END, velocity_stats)
        
        # Аналитика для ускорения
        acceleration_stats = f"""МАРКЕР {marker_id} - УСКОРЕНИЕ
        {'-'*40}
        Максимальное: {accelerations.max():.4f} м/с²
        Минимальное:  {accelerations.min():.4f} м/с²
        Среднее:      {accelerations.mean():.4f} м/с²
        Среднее апроксимации: {smooth_accelerations[:].mean():.4f} м/с²
        Стандартное отклонение: {accelerations.std():.4f}
        {'-'*40}\n\n"""
        text_acceleration_analysis.insert(tk.END, acceleration_stats)

    # Настройка графиков скорости
    ax_velocity.set_title('Анализ скорости маркеров', pad=20)
    ax_velocity.set_xlabel('Номер кадра', labelpad=10)
    ax_velocity.set_ylabel('Скорость (м/с)', labelpad=10)
    ax_velocity.legend(loc='upper right')
    ax_velocity.grid(True, linestyle=':', alpha=0.7)
    fig_velocity.tight_layout()

    # Настройка графиков ускорения
    ax_acceleration.set_title('Анализ ускорения маркеров', pad=20)
    ax_acceleration.set_xlabel('Номер кадра', labelpad=10)
    ax_acceleration.set_ylabel('Ускорение (м/с²)', labelpad=10)
    ax_acceleration.legend(loc='upper right')
    ax_acceleration.grid(True, linestyle=':', alpha=0.7)
    fig_acceleration.tight_layout()

    # Обновление canvas
    canvas_velocity.draw()
    canvas_acceleration.draw()

    # Делаем текст только для чтения
    text_velocity_analysis.config(state=tk.DISABLED)
    text_acceleration_analysis.config(state=tk.DISABLED)
# Главное окно
root = tk.Tk()
root.title("Выбор файла")
root.geometry("300x100")

select_btn = tk.Button(root, text="Выбрать файл CSV", command=select_file)
select_btn.pack(expand=True, padx=20, pady=20)

root.mainloop()