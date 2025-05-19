import cv2
import cv2.aruco as aruco
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import csv
import pandas as pd


dist_coeffs = np.zeros((4, 1))
marker_size = 0.05  # Размер маркера в метрах
object_points = np.array([
    [-marker_size / 2, -marker_size / 2, 0],  # Нижний левый угол
    [marker_size / 2, -marker_size / 2, 0],   # Нижний правый угол
    [marker_size / 2, marker_size / 2, 0],    # Верхний правый угол
    [-marker_size / 2, marker_size / 2, 0]    # Верхний левый угол
], dtype=np.float32)

# Основной класс GUI
class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("ArUco Video Player with Camera Settings")
        
        # Инициализация переменных
        self.video_path = ""
        self.cap = None
        self.current_frame = None
        self.is_playing = False
        self.frame_rate = 30
        self.video_writer = None
        
        # Параметры камеры (по умолчанию)
        self.camera_matrix = np.array([[1000, 0, 0],
                                     [0, 1000, 0],
                                     [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros((5, 1), dtype=np.float32)
        
        # Настройка интерфейса
        self.setup_ui()
        
    def setup_ui(self):
        # Основной фрейм
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Холст для видео
        self.canvas = tk.Canvas(self.main_frame, width=800, height=600)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Панель управления и настроек
        control_frame = tk.Frame(self.main_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)
        
        # Кнопки управления видео
        video_control_frame = tk.Frame(control_frame)
        video_control_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(video_control_frame, text="Открыть видео", command=self.open_video).pack(fill=tk.X, pady=2)
        tk.Button(video_control_frame, text="Play/Pause", command=self.toggle_playback).pack(fill=tk.X, pady=2)
        tk.Button(video_control_frame, text="Стоп", command=self.stop_video).pack(fill=tk.X, pady=2)
        tk.Button(video_control_frame, text="Сохранить видео", command=self.save_video).pack(fill=tk.X, pady=2)
        tk.Button(video_control_frame, text="Сохранить данные", command=self.save_data).pack(fill=tk.X, pady=2)
        
        # Панель настроек камеры
        settings_frame = tk.LabelFrame(control_frame, text="Настройки камеры", padx=5, pady=5)
        settings_frame.pack(fill=tk.X, pady=5)
        
        # Ползунки и текстовые поля для параметров камеры
        self.fx_var = tk.DoubleVar(value=self.camera_matrix[0,0])
        self.fy_var = tk.DoubleVar(value=self.camera_matrix[1,1])
        self.cx_var = tk.DoubleVar(value=self.camera_matrix[0,2])
        self.cy_var = tk.DoubleVar(value=self.camera_matrix[1,2])
        
        # Создаем функцию для обновления значений
        def create_scale_with_entry(frame, label_text, var, from_, to):
            frame_row = tk.Frame(frame)
            frame_row.pack(fill=tk.X, pady=2)
            
            tk.Label(frame_row, text=label_text, width=4).pack(side=tk.LEFT)
            
            # Текстовое поле для точного ввода
            entry = tk.Entry(frame_row, width=7, textvariable=var)
            entry.pack(side=tk.LEFT, padx=5)
            entry.bind('<Return>', lambda e: self.update_from_entry(var, entry, from_, to))
            
            # Ползунок
            scale = tk.Scale(frame_row, from_=from_, to=to, orient=tk.HORIZONTAL,
                            variable=var, command=self.update_camera_params, showvalue=0)
            scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            return entry, scale
        
        # FX (фокусное расстояние по X)
        self.fx_entry, self.fx_scale = create_scale_with_entry(
            settings_frame, "fx:", self.fx_var, 10, 5000)
        
        # FY (фокусное расстояние по Y)
        self.fy_entry, self.fy_scale = create_scale_with_entry(
            settings_frame, "fy:", self.fy_var, 10, 5000)
        
        # CX (координата центра по X)
        self.cx_entry, self.cx_scale = create_scale_with_entry(
            settings_frame, "cx:", self.cx_var, 10, 5000)
        
        # CY (координата центра по Y)
        self.cy_entry, self.cy_scale = create_scale_with_entry(
            settings_frame, "cy:", self.cy_var, 10, 5000)
        
        # Кнопки загрузки/сохранения конфига
        config_btn_frame = tk.Frame(control_frame)
        config_btn_frame.pack(fill=tk.X, pady=5)
        
        tk.Button(config_btn_frame, text="Загрузить конфиг", 
                 command=self.load_camera_config_dialog).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        tk.Button(config_btn_frame, text="Сохранить конфиг", 
                 command=self.save_camera_config).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        # Информация о маркерах
        self.marker_info = tk.Text(control_frame, height=15, width=30)
        self.marker_info.pack(fill=tk.BOTH, expand=True)
        
        self.root.bind("<Configure>", self.on_resize)
    def save_camera_config(self):
        """Сохранение текущих параметров камеры в файл"""
        filepath = filedialog.asksaveasfilename(
            title="Сохранить конфигурацию камеры",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        if not filepath:
            return
        
        try:
            with open(filepath, mode='w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['Parameter', 'Value'])
                writer.writerow(['fx', self.camera_matrix[0,0]])
                writer.writerow(['fy', self.camera_matrix[1,1]])
                writer.writerow(['cx', self.camera_matrix[0,2]])
                writer.writerow(['cy', self.camera_matrix[1,2]])
                writer.writerow(['k1', self.dist_coeffs[0,0]])
                writer.writerow(['k2', self.dist_coeffs[1,0]])
                writer.writerow(['p1', self.dist_coeffs[2,0]])
                writer.writerow(['p2', self.dist_coeffs[3,0]])
                writer.writerow(['k3', self.dist_coeffs[4,0]])
                
            print(f"Параметры камеры сохранены в {filepath}")
        except Exception as e:
            print(f"Ошибка при сохранении конфигурации: {e}")

    def update_camera_params(self, event=None):
        """Обновление параметров камеры"""
        self.camera_matrix[0,0] = self.fx_var.get()  # fx
        self.camera_matrix[1,1] = self.fy_var.get()  # fy
        self.camera_matrix[0,2] = self.cx_var.get()  # cx
        self.camera_matrix[1,2] = self.cy_var.get()  # cy
        
        # Обновление текстовых полей (на случай изменения через ползунок)
        self.fx_entry.delete(0, tk.END)
        self.fx_entry.insert(0, str(int(self.fx_var.get())))
        self.fy_entry.delete(0, tk.END)
        self.fy_entry.insert(0, str(int(self.fy_var.get())))
        self.cx_entry.delete(0, tk.END)
        self.cx_entry.insert(0, str(int(self.cx_var.get())))
        self.cy_entry.delete(0, tk.END)
        self.cy_entry.insert(0, str(int(self.cy_var.get())))
        
        # Если видео загружено, обновляем отображение
        if self.cap and self.current_frame is not None:
            self.show_frame()
   
    def update_from_entry(self, var, entry, min_val, max_val):
        """Обновление значения из текстового поля"""
        try:
            value = float(entry.get())
            if min_val <= value <= max_val:
                var.set(value)
                self.update_camera_params()
            else:
                entry.delete(0, tk.END)
                entry.insert(0, str(var.get()))
        except ValueError:
            entry.delete(0, tk.END)
            entry.insert(0, str(var.get()))
   
    def load_camera_config_dialog(self):
        """Открытие диалога для выбора файла конфигурации"""
        filepath = filedialog.askopenfilename(
            title="Выберите файл конфигурации",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if filepath:
            self.load_camera_config(filepath)
    
    def load_camera_config(self, filename):
        """Загрузка параметров камеры из CSV файла"""
        try:
            with open(filename, mode='r') as csvfile:
                reader = csv.DictReader(csvfile)
                if not reader.fieldnames or 'Parameter' not in reader.fieldnames or 'Value' not in reader.fieldnames:
                    print("Неправильный формат файла конфигурации")
                    return
                
                params = {}
                for row in reader:
                    try:
                        params[row['Parameter']] = float(row['Value'])
                    except ValueError:
                        continue
                
                # Обновляем параметры камеры
                fx = params.get('fx', 1000)
                fy = params.get('fy', 1000)
                cx = params.get('cx', 0)
                cy = params.get('cy', 0)
                
                self.camera_matrix = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
                
                # Обновляем коэффициенты дисторсии
                self.dist_coeffs = np.array([
                    [params.get('k1', 0)],
                    [params.get('k2', 0)],
                    [params.get('p1', 0)],
                    [params.get('p2', 0)],
                    [params.get('k3', 0)]
                ], dtype=np.float32)
                
                # Обновляем интерфейс
                self.fx_var.set(fx)
                self.fy_var.set(fy)
                self.cx_var.set(cx)
                self.cy_var.set(cy)
                
                print(f"Параметры камеры загружены из {filename}")
                
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации: {e}")
        
    def open_video(self):
        """Открыть видеофайл"""
        self.video_path = filedialog.askopenfilename()
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            self.is_playing = False
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)  # Получаем частоту кадров видео
            self.play_video()

    def save_video(self):
        """Сохранение видео с результатами"""
        if self.cap:
            # Получаем название исходного видеофайла
            filename = os.path.basename(self.video_path)
            name, ext = os.path.splitext(filename)

            # Генерация нового имени файла
            result_filename = f"result_{name}.mp4"

            # Получаем ширину и высоту кадра
            frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Устанавливаем кодек для записи в формат MP4
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Кодек для .mp4 файлов
            self.video_writer = cv2.VideoWriter(result_filename, fourcc, self.frame_rate, (frame_width, frame_height))

            if not self.video_writer.isOpened():
                print("Ошибка: Не удалось открыть VideoWriter для записи.")
                return

            print(f"Видео будет сохранено как {result_filename}")

            # Перематываем видео на начало
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Обрабатываем и записываем каждый кадр
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break  # Если кадры закончились, выходим из цикла

                # Преобразуем кадр в оттенки серого
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Детектируем маркеры ArUco
                aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
                parameters = aruco.DetectorParameters_create()
                corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                if ids is not None:
                    frame = aruco.drawDetectedMarkers(frame, corners, ids)

                    # Для каждого маркера решаем задачу PnP
                    for i, corner in zip(ids.flatten(), corners):
                        marker_2d_points = corner[0].reshape(-1, 2)

                        # Решаем PnP для нахождения положения маркера
                        success, rvec, tvec = cv2.solvePnP(object_points, marker_2d_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP, iterationsCount=200, reprojectionError=0.1)

                        if success:
                            # Рисуем оси для маркера на изображении
                            cv2.drawFrameAxes(frame, self.camera_matrix, dist_coeffs, rvec, tvec, 1)

                            # Вычисляем расстояние до маркера
                            distance = np.linalg.norm(tvec)
                            cv2.putText(frame, f"ID {i}: {distance:.2f} m", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Записываем обработанный кадр в выходной файл
                self.video_writer.write(frame)

            # Закрываем VideoWriter после завершения записи
            self.video_writer.release()
            print(f"Видео успешно сохранено как {result_filename}")

    def save_data(self):
        """Сохранение данных с результатами"""
        if self.cap:
            # Получаем название исходного видеофайла
            filename = os.path.basename(self.video_path)
            name, ext = os.path.splitext(filename)
            filename = f"data/{name}.csv"

            with open(filename, mode="w", encoding='utf-8') as w_file:
                # Перематываем видео на начало
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                frame_number = 0
                
                file_writer = csv.writer(w_file, delimiter = ",", lineterminator="\r")
                file_writer.writerow(["id", "frame", "x", "y", "z", "xy", "yz", "xz"])

                # Обрабатываем и записываем каждый кадр
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break  # Если кадры закончились, выходим из цикла

                    frame_number += 1

                    # Преобразуем кадр в оттенки серого
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    # Детектируем маркеры ArUco
                    aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
                    parameters = aruco.DetectorParameters_create()
                    corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

                    if ids is not None:
                        frame = aruco.drawDetectedMarkers(frame, corners, ids)

                        # Для каждого маркера решаем задачу PnP
                        for i, corner in zip(ids.flatten(), corners):
                            marker_2d_points = corner[0].reshape(-1, 2)

                            # Решаем PnP для нахождения положения маркера
                            success, rvec, tvec = cv2.solvePnP(object_points, marker_2d_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

                            if success:
                                file_writer.writerow([i, frame_number, tvec[0][0], tvec[1][0], tvec[2][0], rvec[0][0], rvec[1][0], rvec[2][0]])

    def toggle_playback(self):
        """Переключить режим воспроизведения"""
        if self.is_playing:
            self.is_playing = False
        else:
            self.is_playing = True
            self.play_video()

    def stop_video(self):
        """Остановить видео"""
        self.is_playing = False
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.show_frame()

    def play_video(self):
        """Воспроизведение видео"""
        if self.cap and self.is_playing:
            ret, frame = self.cap.read()
            if not ret:
                return  # Если нет кадра, то остановить воспроизведение

            # Преобразуем кадр в оттенки серого
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Детектируем маркеры ArUco
            aruco_dict = aruco.Dictionary_get(aruco.DICT_4X4_50)
            parameters = aruco.DetectorParameters_create()
            corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            if ids is not None:
                frame = aruco.drawDetectedMarkers(frame, corners, ids)

                # Очищаем текстовое поле с информацией о маркерах
                self.marker_info.delete(1.0, tk.END)

                # Для каждого маркера решаем задачу PnP
                for i, corner in zip(ids.flatten(), corners):
                    marker_2d_points = corner[0].reshape(-1, 2)

                    # Решаем PnP для нахождения положения маркера
                    success, rvec, tvec = cv2.solvePnP(object_points, marker_2d_points, self.camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_SQPNP)

                    if success:
                        # Вычисляем расстояние до маркера
                        distance = np.linalg.norm(tvec)
                        cv2.putText(frame, f"ID {i}: {distance:.2f} m", (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                        # Добавляем информацию о маркере в текстовое поле
                        self.marker_info.insert(tk.END, f"Маркер ID {i}:\n")
                        self.marker_info.insert(tk.END, f"Расстояние: {distance:.2f} м\n\n")

            self.current_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.show_frame()

            # Отображаем кадры через определенный интервал
            if self.is_playing:
                self.root.after(int(1000 / self.frame_rate), self.play_video)

    def show_frame(self):
        """Отобразить кадр в Tkinter canvas"""
        if self.current_frame is not None:
            # Получаем размер холста
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            # Изменяем размер кадра под размер холста
            resized_frame = cv2.resize(self.current_frame, (canvas_width, canvas_height))

            image = Image.fromarray(resized_frame)
            image_tk = ImageTk.PhotoImage(image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=image_tk)
            self.canvas.image = image_tk

    def on_resize(self, event):
        """Обработка изменения размера окна"""
        if self.current_frame is not None:
            self.show_frame()

# Инициализация Tkinter
root = tk.Tk()
app = VideoPlayerApp(root)
root.mainloop()