ADULT CONTENT DETECTOR
-----------------------

!!AT LEAST 2GB SPACE SHOULD BE AVAILABLE
!!CONDA MUST BE INSTALLED
!!INSTALL CONDA FROM https://docs.anaconda.com/free/miniconda/
!!SET PATH IN SYSTEM VARIABLES
!!OPEN ANACONDA PROMPT AND TYPE WHERE CONDA
!!COPY ALL THREE PATHS SEPERATELY AND PASTE IT IN PATHS IN SYSTEM VARIABLES
!!RUN INSTALL_DEPENDENCIES.BAT

2ND METHOD
INSTALL ALL DEPENDENCIES FROM REQUIREMENTS.TXT MANUALLY

SOME PROBLEMS:

problem: some problem with descriptors.
ERRORMSG: "TypeError: Descriptors cannot not be created directly.
If this call came from a _pb2.py file, your generated code is out of date and must be regenerated with protoc >= 3.19.0.
If you cannot immediately regenerate your protos, some other possible workarounds are:
 1. Downgrade the protobuf package to 3.20.x or lower.
 2. Set PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python (but this will use pure-Python parsing and will be much slower)."

to fix: pip install protobuf==3.20.*

problem: pytube cannot download age restricted youtube videos.
to fix: go to C:\Users\{your user directory}\AppData\Roaming\Python\Python311\site-packages\pytube
and find the innertube.py file.
go to line 223 and change:
> def __init__(self, client='ANDROID_MUSIC', use_oauth=False, allow_cache=True):
to
> def __init__(self, client='ANDROID', use_oauth=False, allow_cache=True):

How to use:
WITH  FRONTEND
    1. start the server by running the server.py file
    2. go to http://localhost:8080/


WITH FUNCTION(!!!UNTESTED)
    1. go to main.py and find the def main() class
    2. use the ytvideo_predict_per_frame(url) function(!!!UNTESTED)
    3. input your yt video url


manual use:
    1. download a yt video with youtube_downloader.download_video(url, './inputs/video')
    or insert your own video into the './inputs.video' folder
    2. extract frames with extract_frames(video_path, output_dir, frame_interval),
    take video path as './inputs/video' and output path as './inputs/frames'
    3. use prepare_inputs function to normalise the input frames, use
    prepare_inputs('./inputs/frames', (224, 224)). note: model is trained on images of size 224x224
    4. import the model with model = tensorflow.keras.models.load_model('model.keras')
    5. test_model(model, input_generator) gives the prediction list.

    prediction list is in form [0.9, 0.1] and adds up to 1.0, prediction[0] is safe, prediction[1] is unsafe

    6. match_filename_with_prediction('./inputs/frames', pred) matches the input frames with the prediction for the frame
    (!!!UNTESTED)




