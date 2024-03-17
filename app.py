import numpy as np
import torch.nn.functional as F
from torchvision.transforms.functional import normalize
from skimage import io
import torch, os
from PIL import Image
from briarmbg import BriaRMBG
import gradio as gr
import cv2
import numpy as np
import time
import random
from PIL import Image
import moviepy.editor as moviepy

bgrm = BriaRMBG.from_pretrained("briaai/RMBG-1.4")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"
bgrm.to(device)


def resize_image(image):
    image = image.convert('RGB')
    model_input_size = (1024, 1024)
    image = image.resize(model_input_size, Image.BILINEAR)
    return image


def process(image):

    # prepare input
    orig_image = Image.fromarray(image)
    w,h = orig_im_size = orig_image.size
    image = resize_image(orig_image)
    im_np = np.array(image)
    im_tensor = torch.tensor(im_np, dtype=torch.float32).permute(2,0,1)
    im_tensor = torch.unsqueeze(im_tensor,0)
    im_tensor = torch.divide(im_tensor,255.0)
    im_tensor = normalize(im_tensor,[0.5,0.5,0.5],[1.0,1.0,1.0])
    if torch.cuda.is_available():
        im_tensor=im_tensor.cuda()

    #inference
    result=bgrm(im_tensor)
    # post process
    result = torch.squeeze(F.interpolate(result[0][0], size=(h,w), mode='bilinear') ,0)
    ma = torch.max(result)
    mi = torch.min(result)
    result = (result-mi)/(ma-mi)    
    # image to pil
    im_array = (result*255).cpu().data.numpy().astype(np.uint8)
    pil_im = Image.fromarray(np.squeeze(im_array))
    # paste the mask on the original image
    new_im = Image.new("RGBA", pil_im.size, (0,255,0,255))
    new_im.paste(orig_image, mask=pil_im)
    # new_orig_image = orig_image.convert('RGBA')
    return new_im



def to_mp4(txt):
    clip = moviepy.VideoFileClip(txt)
    out = 'out.mp4'
    clip.write_videofile(out)
    return out


def process_video(video, progress=gr.Progress()):
    
    cap = cv2.VideoCapture(video)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get total frames
    writer = None
    #tmpname = random.randint(111111111, 999999999)
    tmpname = "output.mp4"
    processed_frames = 0
    start_time = time.time()
    i=0
    while cap.isOpened():
        ret, frame = cap.read()

        if ret is False:
            break

        if time.time() - start_time >= 20 * 60 - 5:
            print("GPU Timeout is coming")
            cap.release()
            writer.release()
            return tmpname
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame).convert('RGB')

        if writer is None:
            writer = cv2.VideoWriter(tmpname, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), img.size)

        processed_frames += 1
        print(f"Processing frame {processed_frames}")
        progress(processed_frames / total_frames, desc=f"Processing frame {processed_frames}/{total_frames}")
        out = process(np.array(img))
        writer.write(cv2.cvtColor(np.array(out), cv2.COLOR_BGR2RGB))

    cap.release()
    writer.release()
    return to_mp4(tmpname)

title = "Video Background Remover Green Screen"
description = """Removes your video background and adds green screen Using the briaai/RMBG-1.4 from HuggingFace """

examples = [['./input.mp4']]

iface = gr.Interface(
    fn=process_video,
    inputs=["video"],
    outputs="video",
    examples=examples,
    title=title,
    description=description
)
iface.launch()
