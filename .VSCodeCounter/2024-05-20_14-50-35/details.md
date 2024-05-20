# Details

Date : 2024-05-20 14:50:35

Directory /home/wenzihao/Desktop/WandCalibration

Total : 65 files,  3997 codes, 457 comments, 420 blanks, all 4874 lines

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)

## Files
| filename | language | code | comment | blank | total |
| :--- | :--- | ---: | ---: | ---: | ---: |
| [README.md](/README.md) | Markdown | 39 | 0 | 3 | 42 |
| [blender/test.py](/blender/test.py) | Python | 12 | 1 | 10 | 23 |
| [config/board/board_sample.yaml](/config/board/board_sample.yaml) | YAML | 3 | 14 | 4 | 21 |
| [config/board/charucoboard_4_5.yaml](/config/board/charucoboard_4_5.yaml) | YAML | 5 | 10 | 3 | 18 |
| [config/board/checkerboard_11_8.yaml](/config/board/checkerboard_11_8.yaml) | YAML | 4 | 12 | 2 | 18 |
| [config/cfg_archery.yaml](/config/cfg_archery.yaml) | YAML | 25 | 11 | 13 | 49 |
| [config/cfg_archery_outdoor.yaml](/config/cfg_archery_outdoor.yaml) | YAML | 25 | 11 | 13 | 49 |
| [config/cfg_uni.yaml](/config/cfg_uni.yaml) | YAML | 41 | 32 | 9 | 82 |
| [config/cfg_wtt.yaml](/config/cfg_wtt.yaml) | YAML | 26 | 11 | 8 | 45 |
| [config/cfg_wtt_red.yaml](/config/cfg_wtt_red.yaml) | YAML | 25 | 11 | 7 | 43 |
| [config/config.yaml](/config/config.yaml) | YAML | 17 | 10 | 13 | 40 |
| [config/labelme/labelme_sample.yaml](/config/labelme/labelme_sample.yaml) | YAML | 6 | 2 | 0 | 8 |
| [config/point/point_sample.yaml](/config/point/point_sample.yaml) | YAML | 6 | 2 | 0 | 8 |
| [config/point/point_wtt.yaml](/config/point/point_wtt.yaml) | YAML | 6 | 1 | 0 | 7 |
| [config/pole/pole500.yaml](/config/pole/pole500.yaml) | YAML | 4 | 0 | 0 | 4 |
| [config/pole/pole760.yaml](/config/pole/pole760.yaml) | YAML | 4 | 0 | 0 | 4 |
| [config/pole/pole760_red.yaml](/config/pole/pole760_red.yaml) | YAML | 4 | 0 | 0 | 4 |
| [config/pole/pole_sample.yaml](/config/pole/pole_sample.yaml) | YAML | 3 | 7 | 2 | 12 |
| [config/wand/wand750.yaml](/config/wand/wand750.yaml) | YAML | 10 | 8 | 1 | 19 |
| [config/wand/wand_archery.yaml](/config/wand/wand_archery.yaml) | YAML | 10 | 4 | 1 | 15 |
| [config/wand/wand_sample.yaml](/config/wand/wand_sample.yaml) | YAML | 9 | 7 | 1 | 17 |
| [config/wand/wand_wtt.yaml](/config/wand/wand_wtt.yaml) | YAML | 10 | 4 | 1 | 15 |
| [config/wand/wand_wtt_red.yaml](/config/wand/wand_wtt_red.yaml) | YAML | 10 | 8 | 4 | 22 |
| [extrinsicParameter/__init__.py](/extrinsicParameter/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [extrinsicParameter/initPose/initPose.py](/extrinsicParameter/initPose/initPose.py) | Python | 65 | 3 | 7 | 75 |
| [extrinsicParameter/initPose/integratePose.py](/extrinsicParameter/initPose/integratePose.py) | Python | 50 | 3 | 4 | 57 |
| [extrinsicParameter/initPose/mst.py](/extrinsicParameter/initPose/mst.py) | Python | 55 | 1 | 8 | 64 |
| [extrinsicParameter/initPose/recoverPose.py](/extrinsicParameter/initPose/recoverPose.py) | Python | 78 | 5 | 8 | 91 |
| [extrinsicParameter/poleDetection/__init__.py](/extrinsicParameter/poleDetection/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [extrinsicParameter/poleDetection/blobDetection.py](/extrinsicParameter/poleDetection/blobDetection.py) | Python | 166 | 21 | 11 | 198 |
| [extrinsicParameter/poleDetection/maskGeneration.py](/extrinsicParameter/poleDetection/maskGeneration.py) | Python | 122 | 15 | 10 | 147 |
| [extrinsicParameter/poleDetection/poleDetection.py](/extrinsicParameter/poleDetection/poleDetection.py) | Python | 193 | 10 | 11 | 214 |
| [extrinsicParameter/poleDetection/wandDetection.py](/extrinsicParameter/poleDetection/wandDetection.py) | Python | 194 | 9 | 11 | 214 |
| [extrinsicParameter/refinePose/boundleAdjustment.py](/extrinsicParameter/refinePose/boundleAdjustment.py) | Python | 258 | 26 | 19 | 303 |
| [extrinsicParameter/refinePose/multiViewTriangulate.py](/extrinsicParameter/refinePose/multiViewTriangulate.py) | Python | 194 | 16 | 11 | 221 |
| [extrinsicParameter/refinePose/normalizedImagePlane.py](/extrinsicParameter/refinePose/normalizedImagePlane.py) | Python | 50 | 6 | 6 | 62 |
| [extrinsicParameter/refinePose/refinePose.py](/extrinsicParameter/refinePose/refinePose.py) | Python | 251 | 17 | 15 | 283 |
| [extrinsicParameter/worldCoord/adjustCamParam.py](/extrinsicParameter/worldCoord/adjustCamParam.py) | Python | 65 | 0 | 3 | 68 |
| [extrinsicParameter/worldCoord/clickPoint.py](/extrinsicParameter/worldCoord/clickPoint.py) | Python | 98 | 27 | 10 | 135 |
| [extrinsicParameter/worldCoord/cluster.py](/extrinsicParameter/worldCoord/cluster.py) | Python | 40 | 4 | 5 | 49 |
| [extrinsicParameter/worldCoord/get_cam0_extrinsic.py](/extrinsicParameter/worldCoord/get_cam0_extrinsic.py) | Python | 215 | 16 | 9 | 240 |
| [extrinsicParameter/worldCoord/get_id_with_distance.py](/extrinsicParameter/worldCoord/get_id_with_distance.py) | Python | 62 | 0 | 8 | 70 |
| [extrinsicParameter/worldCoord/handle_board.py](/extrinsicParameter/worldCoord/handle_board.py) | Python | 106 | 3 | 5 | 114 |
| [extrinsicParameter/worldCoord/handle_labelme.py](/extrinsicParameter/worldCoord/handle_labelme.py) | Python | 133 | 1 | 9 | 143 |
| [extrinsicParameter/worldCoord/icp.py](/extrinsicParameter/worldCoord/icp.py) | Python | 78 | 16 | 28 | 122 |
| [extrinsicParameter/worldCoord/noIdReconstruction.py](/extrinsicParameter/worldCoord/noIdReconstruction.py) | Python | 182 | 25 | 15 | 222 |
| [extrinsicParameter/worldCoord/solve_icp.py](/extrinsicParameter/worldCoord/solve_icp.py) | Python | 21 | 2 | 3 | 26 |
| [intrinsicParameter/__init__.py](/intrinsicParameter/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [intrinsicParameter/charucoboardCalibration/get_cam_calibration.py](/intrinsicParameter/charucoboardCalibration/get_cam_calibration.py) | Python | 115 | 12 | 6 | 133 |
| [intrinsicParameter/charucoboardCalibration/test_cam_calibration.py](/intrinsicParameter/charucoboardCalibration/test_cam_calibration.py) | Python | 44 | 1 | 8 | 53 |
| [intrinsicParameter/checkerboardCalibration/get_cam_calibration.py](/intrinsicParameter/checkerboardCalibration/get_cam_calibration.py) | Python | 91 | 9 | 5 | 105 |
| [intrinsicParameter/checkerboardCalibration/test_cam_calibration.py](/intrinsicParameter/checkerboardCalibration/test_cam_calibration.py) | Python | 49 | 14 | 18 | 81 |
| [intrinsicParameter/intrinsicCalibration/get_intrinsic.py](/intrinsicParameter/intrinsicCalibration/get_intrinsic.py) | Python | 66 | 10 | 6 | 82 |
| [main.py](/main.py) | Python | 185 | 2 | 7 | 194 |
| [pyinstaller.md](/pyinstaller.md) | Markdown | 6 | 0 | 1 | 7 |
| [requirements.txt](/requirements.txt) | pip requirements | 12 | 0 | 1 | 13 |
| [utils/__init__.py](/utils/__init__.py) | Python | 0 | 0 | 1 | 1 |
| [utils/imageConcat.py](/utils/imageConcat.py) | Python | 36 | 3 | 2 | 41 |
| [utils/verifyAccuracy.py](/utils/verifyAccuracy.py) | Python | 52 | 0 | 6 | 58 |
| [utils/yamlLoader.py](/utils/yamlLoader.py) | Python | 58 | 8 | 14 | 80 |
| [visualize/get_init_camera_params.py](/visualize/get_init_camera_params.py) | Python | 17 | 0 | 3 | 20 |
| [visualize/visCameraParams.py](/visualize/visCameraParams.py) | Python | 48 | 3 | 3 | 54 |
| [visualize/vis_pole_detection.py](/visualize/vis_pole_detection.py) | Python | 67 | 1 | 8 | 76 |
| [visualize/vis_pole_spread.py](/visualize/vis_pole_spread.py) | Python | 64 | 0 | 6 | 70 |
| [visualize/vis_reproj_error.py](/visualize/vis_reproj_error.py) | Python | 107 | 2 | 11 | 120 |

[Summary](results.md) / Details / [Diff Summary](diff.md) / [Diff Details](diff-details.md)