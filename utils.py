import os
from glob import glob
import cv2
from PIL import Image
import io
import base64
from IPython.display import HTML

def show_video(path: str):
    """
    show video in jupyter notebook, agent interaction in environment.
    Takes - path to dir with videos.
    Returns - html video player in jupyter notebook.
    """  
    video_path = sorted(glob(path + "/*.mp4"))[-1]
    video = io.open(video_path, 'r+b').read()
    encoded = base64.b64encode(video)

    return HTML(data='''<video alt="test" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4" /> </video>'''
    .format(encoded.decode('ascii')))

def convert_gif(path: str,
                gif_name: str = 'gif_name.gif',
                frame_limit: int = 100):
    """
    convert video into GIF file.
    path - path to dir with videos.
    gif_name - name to save the GIF file.
    frame_limit - the maximum number of frames in a GIF.
    """
    video_path = glob(path + "/*.mp4")[-1]
    vidcap = cv2.VideoCapture(video_path)
    success, frame = vidcap.read()
    count = 0

    # extracting and saving video frames.
    while success:
        cv2.imwrite(f"{path}/frame{count}.png", frame)     
        success, frame = vidcap.read()
        count += 1
        if count > frame_limit:
            break
    print("total frames:", count)

    # generate animated GIF.
    img, *imgs = [Image.open(f) for f in sorted(glob(path+"/*.png"))]
    img.save(fp=f"{path}/{gif_name}", format='GIF', append_images=imgs,
         save_all=True, duration=200, loop=0)
    
    # remove frames
    [os.remove(os.path.join(path, f)) for f in glob(path+"/*.png")]

