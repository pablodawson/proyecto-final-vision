########################################################################
#
# Copyright (c) 2022, STEREOLABS.
#
# All rights reserved.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
########################################################################

"""
    This sample shows how to track the position of the ZED camera 
    and displays it in a OpenGL window.
"""

import ogl_viewer.tracking_viewer as gl
import pyzed.sl as sl
import argparse
import time 
import json
import os
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn.functional as F
import sys

device = 'cuda'

# Argumentos
def parse_args(init):
    if len(opt.input_svo_file) > 0 and opt.input_svo_file.endswith(".svo2"):
        init.set_from_svo_file(opt.input_svo_file)
        print("[Sample] Using SVO File input: {0}".format(opt.input_svo_file))
    elif len(opt.ip_address) > 0 :
        ip_str = opt.ip_address
        if ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4 and len(ip_str.split(':'))==2:
            init.set_from_stream(ip_str.split(':')[0],int(ip_str.split(':')[1]))
            print("[Sample] Using Stream input, IP : ",ip_str)
        elif ip_str.replace(':','').replace('.','').isdigit() and len(ip_str.split('.'))==4:
            init.set_from_stream(ip_str)
            print("[Sample] Using Stream input, IP : ",ip_str)
        else :
            print("Unvalid IP format. Using live stream")
    if ("resolution" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD2K
        print("[Sample] Using Camera in resolution HD2K")
    elif ("HD1200" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1200
        print("[Sample] Using Camera in resolution HD1200")
    elif ("HD1080" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD1080
        print("[Sample] Using Camera in resolution HD1080")
    elif ("HD720" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.HD720
        print("[Sample] Using Camera in resolution HD720")
    elif ("SVGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.SVGA
        print("[Sample] Using Camera in resolution SVGA")
    elif ("VGA" in opt.resolution):
        init.camera_resolution = sl.RESOLUTION.VGA
        print("[Sample] Using Camera in resolution VGA")
    elif len(opt.resolution)>0: 
        print("[Sample] No valid resolution entered. Using default")
    else : 
        print("[Sample] Using default resolution")
        
def main():
    init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720,
                                 coordinate_units=sl.UNIT.METER,
                                 coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP)
    parse_args(init_params)

    # Crear directorios para el output
    os.makedirs('dataset_leapvo', exist_ok=True)
    os.makedirs('dataset_leapvo/rgb', exist_ok=True)
    os.makedirs('dataset_leapvo/depth', exist_ok=True)

    # Crear los archivos de texto para guardar las poses y las imagenes (imitando TUM-RGBD)
    # Headers
    poses_txt = open('dataset_leapvo/groundtruth.txt', 'w')
    poses_txt.write("# ground truth trajectory\n")
    poses_txt.write("# file: 'rgbd_dataset_piscina.bag'\n")
    poses_txt.write("# timestamp tx ty tz qx qy qz qw\n")

    rgb_txt = open('dataset_leapvo/rgb.txt', 'w')
    rgb_txt.write("# color images\n")
    rgb_txt.write("# file: 'rgbd_dataset_piscina.bag'\n")
    rgb_txt.write("# timestamp filename\n")

    depth_txt = open('dataset_leapvo/depth.txt', 'w')
    depth_txt.write("# depth maps\n")
    depth_txt.write("# file: 'rgbd_dataset_piscina.bag'\n")
    depth_txt.write("# timestamp filename\n")
    
    # Iniciar camara
    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print("Camera Open", status, "Exit program.")
        exit(1)

    if len(opt.roi_mask_file) > 0:
        mask_roi = sl.Mat()
        err = mask_roi.read(opt.roi_mask_file)
        if err == sl.ERROR_CODE.SUCCESS:
            zed.set_region_of_interest(mask_roi, [sl.MODULE.ALL])
        else:
            print(f"Error loading Region of Interest file {opt.roi_mask_file}. Please check the path.")

    init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    # Configuracion extra de camara
    tracking_params = sl.PositionalTrackingParameters() #set parameters for Positional Tracking
    tracking_params.enable_imu_fusion = True
    tracking_params.mode = sl.POSITIONAL_TRACKING_MODE.GEN_1
    status = zed.enable_positional_tracking(tracking_params) #enable Positional Tracking
    if status != sl.ERROR_CODE.SUCCESS:
        print("[Sample] Enable Positional Tracking : "+repr(status)+". Exit program.")
        zed.close()
        exit()

    runtime = sl.RuntimeParameters()
    camera_pose = sl.Pose()

    roi = sl.Mat()
    roi_name = "roi_mask.png"
    roi_param = sl.RegionOfInterestParameters()

    if opt.roi_mask_file == "":
        roi_param.auto_apply_module = {sl.MODULE.DEPTH, sl.MODULE.POSITIONAL_TRACKING}
        zed.start_region_of_interest_auto_detection(roi_param)
        print("[Sample]  Region Of Interest auto detection is running.")

    camera_info = zed.get_camera_information()

    # OpenGL viewer para renderizar el recorrido
    viewer = gl.GLViewer()
    viewer.init(camera_info.camera_model)
    py_translation = sl.Translation()
    pose_data = sl.Transform()
    
    # Resolucion a guardar
    eval_w, eval_h = (768, 384)

    # Guardar los parametros de calibracion de la camara, ajustar segun nueva resolucion
    calibration_params = camera_info.camera_configuration.calibration_parameters
    focal_left_x = calibration_params.left_cam.fx
    focal_left_y = calibration_params.left_cam.fy
    principal_left_x = calibration_params.left_cam.cx
    principal_left_y = calibration_params.left_cam.cy
    baseline = calibration_params.stereo_transform.m[0][3]

    width = calibration_params.left_cam.image_size.width
    height = calibration_params.left_cam.image_size.height

    ratio_x = eval_w/ width
    ratio_y = eval_h/ height

    with open('dataset_leapvo/calib.txt', 'w') as f:
        f.write(f"{focal_left_x*ratio_x} {focal_left_y*ratio_y} {principal_left_x*ratio_x} {principal_left_y*ratio_y}")

    # Extraer imagen izquierda, profundidad como imagen, profundidad como medida
    mat_img = sl.Mat()
    mat_depth_view = sl.Mat()
    mat_depth_measure = sl.Mat()


    text_translation = ""
    text_rotation = ""

    roi_state = sl.REGION_OF_INTEREST_AUTO_DETECTION_STATE.NOT_ENABLED

    # Recorrer la secuencia de frames hasta la 300
    num_frames = 300
    i=0

    while viewer.is_available():
        i+=1
        if i>num_frames:
            break

        if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
            tracking_state = zed.get_position(camera_pose,sl.REFERENCE_FRAME.WORLD) #Get the position of the camera in a fixed reference frame (the World Frame)
            tracking_status = zed.get_positional_tracking_status()

            timestamp = str(zed.get_timestamp(sl.TIME_REFERENCE.CURRENT).get_microseconds())

            #Get rotation and translation and displays it
            if tracking_state == sl.POSITIONAL_TRACKING_STATE.OK:
                rotation = camera_pose.get_rotation_vector()
                translation = camera_pose.get_translation(py_translation)
                text_rotation = str((round(rotation[0], 2), round(rotation[1], 2), round(rotation[2], 2)))
                text_translation = str((round(translation.get()[0], 2), round(translation.get()[1], 2), round(translation.get()[2], 2)))


            pose_data = camera_pose.pose_data(sl.Transform())
            pose_matrix = pose_data.m
            
            # Guardar pose en texto, X Y Z QX QY QZ QW (coordenadas y cuaterniones)
            tx, ty, tz = pose_matrix[0][3], pose_matrix[1][3], pose_matrix[2][3]

            rotation_matrix = pose_matrix[:3, :3]
            r = R.from_matrix(rotation_matrix)
            qx, qy, qz, qw = r.as_quat()

            poses_txt.write(f"{timestamp} {tx} {ty} {tz} {qx} {qy} {qz} {qw}\n")

            # Update rotation, translation and tracking state values in the OpenGL window
            viewer.updateData(pose_data, text_translation, text_rotation, tracking_status)

            # Obtener imagen y depth
            zed.retrieve_image(mat_img, sl.VIEW.LEFT)
            zed.retrieve_image(mat_depth_view, sl.VIEW.DEPTH)
            zed.retrieve_measure(mat_depth_measure, sl.MEASURE.DEPTH)

            img_np = mat_img.get_data()
            
            # Cambiar tamaño
            depth_np_view = mat_depth_view.get_data()
            depth_np_view = cv2.resize(depth_np_view, (eval_w, eval_h))

            depth_np = mat_depth_measure.get_data()
            depth_np = np.nan_to_num(depth_np)
            depth_np = cv2.resize(depth_np, (eval_w, eval_h))

            img_np_in = cv2.resize(img_np, (eval_w, eval_h))
            img_np_in = cv2.cvtColor(img_np_in, cv2.COLOR_RGBA2RGB)

            # Guardar todo en la carpeta
            img_np_in = cv2.cvtColor(img_np_in, cv2.COLOR_RGBA2RGB)

            cv2.imwrite(f'dataset_leapvo/rgb_l/frame_{str(i).zfill(4)}.png', img_np_in)
            np.save(f'dataset_leapvo/depth_raw/frame_{str(i).zfill(4)}.npy', depth_np)
            
            # Agregar ubicacion del archivo a los txt
            rgb_txt.write(f"{timestamp} rgb/frame_{str(i).zfill(4)}.png\n")
            depth_txt.write(f"{timestamp} depth/frame_{str(i).zfill(4)}.png\n")

            # If the region of interest auto detection is running, the resulting mask can be saved and reloaded for later use
            if opt.roi_mask_file == "" and roi_state == sl.REGION_OF_INTEREST_AUTO_DETECTION_STATE.RUNNING and zed.get_region_of_interest_auto_detection_status() == sl.REGION_OF_INTEREST_AUTO_DETECTION_STATE.READY:
                print("Region Of Interest detection done! Saving into {}".format(roi_name))
                zed.get_region_of_interest(roi, sl.Resolution(0,0), sl.MODULE.POSITIONAL_TRACKING)
                roi.write(roi_name)

            roi_state = zed.get_region_of_interest_auto_detection_status()
        else: 
            time.sleep(0.001)
    
    viewer.exit()
    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_svo_file', type=str, help='Path to an .svo file, if you want to replay it',default = '')
    parser.add_argument('--ip_address', type=str, help='IP Adress, in format a.b.c.d:port or a.b.c.d, if you have a streaming setup', default = '')
    parser.add_argument('--resolution', type=str, help='Resolution, can be either HD2K, HD1200, HD1080, HD720, SVGA or VGA', default = 'HD720')
    parser.add_argument('--roi_mask_file', type=str, help='Path to a Region of Interest mask file', default = '')
    opt = parser.parse_args()
    if (len(opt.input_svo_file)>0 and len(opt.ip_address)>0):
        print("Specify only input_svo_file or ip_address, or none to use wired camera, not both. Exit program")
        exit()
    main() 
    