import pyrealsense2 as rs
import numpy as np
from sklearn.metrics import mean_squared_error
import time
from sklearn.linear_model import LinearRegression
import open3d as o3d
import math

def calculate_angles(accel_data, gyro_data, dt):
    # Desde acelerómetro y giroscopio
    pitch = math.atan2(accel_data.y, math.sqrt(accel_data.x**2 + accel_data.z**2))
    roll = math.atan2(-accel_data.x, math.sqrt(accel_data.y**2 + accel_data.z**2))

    # Yaw usando dt
    yaw = gyro_data.z * dt

    # A grados sexagesimales
    pitch_deg = math.degrees(pitch)
    roll_deg = math.degrees(roll)
    yaw_deg = math.degrees(yaw)
    
    return pitch_deg, roll_deg, yaw_deg
    
def encontrar_bordes(total_ridges, num_sections=12):
    range_x_min, range_x_max = -0.3, 0.3  # Rango inicial en el eje X que queremos considerar

    y_min, y_max = np.min(total_ridges[:, 1]), np.max(total_ridges[:, 1])
    y_sections = np.linspace(y_min, y_max, num_sections + 1)
    

    combined_borders = []
    ridge_points_within_borders = []
    

    previous_x_min, previous_x_max = range_x_min, range_x_max  # Inicializar límites con el rango inicial
    
    for i in range(num_sections - 1, -1, -1):  # Iterar de abajo hacia arriba
        y_lower, y_upper = y_sections[i], y_sections[i + 1]
    

        section_points = total_ridges[(total_ridges[:, 1] >= y_lower) & (total_ridges[:, 1] < y_upper)]
    
     
        current_x_min, current_x_max = previous_x_min, previous_x_max
    
   
        while True:
     
            section_points_in_range = section_points[(section_points[:, 0] >= current_x_min) & (section_points[:, 0] <= current_x_max)]
    
  
            points_left_beyond = section_points[(section_points[:, 0] < current_x_min) & (section_points[:, 0] >= current_x_min - 0.1)]
            points_right_beyond = section_points[(section_points[:, 0] > current_x_max) & (section_points[:, 0] <= current_x_max + 0.1)]
    
            if len(points_left_beyond) > 0:
                current_x_min -= 0.1  # Expandir el rango hacia la izquierda
            elif len(points_right_beyond) > 0:
                current_x_max += 0.1  # Expandir el rango hacia la derecha
            else:
                # Salir del bucle si no hay más puntos fuera del rango
                break
    
        if len(section_points_in_range) > 0:
            left_point = section_points_in_range[np.argmin(section_points_in_range[:, 0])]
            right_point = section_points_in_range[np.argmax(section_points_in_range[:, 0])]
    
            if abs(right_point[0] - left_point[0]) > 0.05:
                combined_borders.append(left_point)
                combined_borders.append(right_point)
    
                points_within = section_points[(section_points[:, 0] >= left_point[0]) & (section_points[:, 0] <= right_point[0])]
                ridge_points_within_borders.extend(points_within)
            previous_x_min, previous_x_max = current_x_min, current_x_max

    combined_borders = np.array(combined_borders)
    ridge_points_within_borders = np.array(ridge_points_within_borders)

    return combined_borders, ridge_points_within_borders

def recta_estimada(verts):
    MIN_DISTANCE = 0.001  
    MAX_DISTANCE = 3.5 
  
    valid_vertices = verts[
    (np.abs(verts[:, 0]) >= MIN_DISTANCE) &
    (np.abs(verts[:, 1]) >= MIN_DISTANCE) &
    (np.abs(verts[:, 2]) >= MIN_DISTANCE) &
    (np.abs(verts[:, 2]) <= MAX_DISTANCE)
    ]

    valid_vertices[:, 1] *= -1
    valid_vertices[:, 2] *= -1
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_vertices)
    voxel_size = 0.01  
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_vertices = np.asarray(downsampled_pcd.points)
    total_ridges= [];
    total_furrows= [];
    #Por cada frame:
    downsampled_vertices = np.array(downsampled_vertices)
    cut_distance = 0.25
    z_max = -1.1
    z_min = z_max-0.04
    for i in range (8):
        sliced_frame = downsampled_vertices[(downsampled_vertices[:, 2] >= (z_min-cut_distance*i)) & (downsampled_vertices[:, 2] <= (z_max-cut_distance*i))&(downsampled_vertices[:, 1] < 2)]
        rotated_frame_cut = sliced_frame[(sliced_frame[:, 0] >= (-2)) & (sliced_frame[:, 0] <= (2))]
        X = rotated_frame_cut[:, 0].reshape(-1, 1)  
        Y = rotated_frame_cut[:, 1]
        try:
            model = LinearRegression()
            model.fit(X, Y)
            Y_pred = model.predict(X)
    
            # Clasificar puntos como 'ridge' o 'furrow'
            ridge_points = rotated_frame_cut[rotated_frame_cut[:, 1] > Y_pred]
            total_ridges.extend(ridge_points)
            furrow_points = rotated_frame_cut[rotated_frame_cut[:, 1] <= Y_pred]
            total_furrows.extend(furrow_points)
    
        except Exception as e:
            print(f"Error en la iteración {i}: {e}")
            # Continuar con la siguiente iteración si hay un error
            continue
        
    total_ridges = np.array(total_ridges)
    total_furrows = np.array(total_furrows)   
    bordes_total, central_ridge = encontrar_bordes(total_ridges)# Función para encontrar los bordes
    X = bordes_total[:, 1].reshape(-1, 1)  # Variable independiente (eje X)
    Y = bordes_total[:, 0]
    modelX = LinearRegression()
    modelX.fit(X, Y)
    X_pred = modelX.predict(X)
    
    two_points = []
    search_radius = 0.01
    n = 0
    while len(two_points) < 2 and search_radius <= 0.03:
        for idx, x_pred in enumerate(X_pred):  # idx índice y x_pred valor
            if idx >= len(X_pred)/2:
                n = int((idx-len(X_pred)/2)*2+1)
            else:
                n = idx*2
            for ridge in central_ridge:
                if abs(X_pred[n] - ridge[0]) < search_radius and abs(bordes_total[n, 1] - ridge[1]) < search_radius:
                    if not any( abs(bordes_total[n, 1] - point[1]) <= 0.1 for point in two_points):
                        two_points.append((X_pred[n], bordes_total[n, 1], ridge[2])) 
                        break
            if len(two_points) >= 2:
                break
        search_radius += 0.01
         
    two_points = np.asarray(two_points)

    return two_points, downsampled_vertices

def rotate_point_cloud_y(points, pitch_angle):
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(pitch_angle), -np.sin(pitch_angle)],
        [0, np.sin(pitch_angle), np.cos(pitch_angle)]
    ])

    rotated_points = points @ R_x.T  # Usamos la transposición para multiplicación de matrices
    return rotated_points
########

Points_to_eval = []
pipeline = rs.pipeline()
config = rs.config()
config.enable_device_from_file('Videos/Ultima_hilera.bag')
config.enable_stream(rs.stream.depth)
config.enable_stream(rs.stream.accel)
config.enable_stream(rs.stream.gyro)
pipeline.start(config)
profile = pipeline.get_active_profile()
depth_stream = profile.get_stream(rs.stream.depth)
intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()
pc = rs.pointcloud()



try: 
    frames = pipeline.wait_for_frames()
    start_time = time.time()
    last_time = start_time
    depth_frame = frames.get_depth_frame()
    accel_frame = frames.first_or_default(rs.stream.accel)
    gyro_frame = frames.first_or_default(rs.stream.gyro)
    if not depth_frame:
        print("No se pudo obtener el frame de profundidad")
        exit()

    depth_image = np.asanyarray(depth_frame.get_data())
    accel_data = accel_frame.as_motion_frame().get_motion_data()
    gyro_data = gyro_frame.as_motion_frame().get_motion_data()
    points = pc.calculate(depth_frame)
    pc.map_to(depth_frame)

    v,t = points.get_vertices(), points.get_texture_coordinates()
    verts = np. asanyarray(v).view(np.float32).reshape(-1,3) 

    # Tiempo
    current_time = time.time()
    dt = current_time - last_time
    last_time = current_time

    # Pitch, roll, yaw
    pitch, roll, yaw = calculate_angles(accel_data, gyro_data, dt)
    
    two_points, down_verts = recta_estimada(verts)
    pitch_angle = np.radians(-90-pitch)  
    rotated_two_points = rotate_point_cloud_y(two_points, pitch_angle)

finally:
    pipeline.stop()
