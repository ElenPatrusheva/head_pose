- install req.txt
- download shape_predictor_68_face_landmarks.dat
- to process video offline run (it takes a few minutes):<br>
python offline_video_head_pose.py --shape-predictor shape_predictor_68_face_landmarks.dat --in [path to input video] --out [path to output video]
- to process from web camera run (slow -> low quality): <br>
python real_time_video_head_pose.py --shape-predictor shape_predictor_68_face_landmarks.dat
