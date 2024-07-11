# Proyecto final visión por computador

Pablo Dawson, Vicente Toledo

## Estructura

*dataset*: Scripts que convierten el archivo **svo2** de la captura de Zed en un dataset en el formato de el dataset **TUM RGBD**.

*leapvo*: El sistema de SLAM.

## Instrucciones para ejecutar

1. Seguir las instrucciones en el *Readme* de leapvo para crear el entorno.

2. Instalar ZED Python API, junto con ZED SDK: https://www.stereolabs.com/docs/app-development/python/install

3. Crear dataset:
```
cd dataset
python dataset_creator_leapvo.py --input_svo_file (archivo a procesar)
```

4. Editar el archivo de configuración en **leapvo/configs/pool.yaml** con la dirección del dataset y el groundtruth generado:

```
data:
  imagedir: '/home/pablo/proyecto-vision/dataset_leapvo/rgb'
  calib: '/home/pablo/proyecto-vision/dataset_leapvo/calib.txt'
  stride: 3
  skip: 0
  max_length: 900
  gt_traj: '/home/pablo/proyecto-vision/dataset_leapvo/groundtruth.txt' ##
  name: 'pool' 
  savedir: 'logs/pool'
  traj_format: 'tum' ##
```

5. Finalmente ejecutar leapvo:

```
cd leapvo
python main/eval.py --config-path=../configs --config-name=pool
```