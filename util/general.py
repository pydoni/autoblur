import glob
from pathlib import Path
import os
import cv2
import sys
sys.path.append('../networks')
from networks.yolov5 import yolov5_inference
from tqdm import tqdm


def get_source(source):

    image_formats = "bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"  # include image suffixes
    video_formats = "asf", "avi", "gif", "m4v", "mkv", "mov", "mp4", "mpeg", "mpg", "ts", "wmv"  # include video suffixes

    is_image_file = Path(source).suffix[1:] in image_formats
    is_video_file = Path(source).suffix[1:] in video_formats
    is_image_glob = "*." in Path(source).name and Path(source).suffix[1:] in image_formats

    if is_image_glob:
        data = glob.glob(source)
        source_type = "image_glob"
        source_content = {"type": source_type, "data": data}
        return source_content
    elif is_image_file:
        source_type = "image"
        source_content = {"type": source_type, "data": source}
        return source_content
    elif is_video_file:
        source_type = "video"
        source_content = {"type": source_type, "data": source}
        return source_content
    elif source == "0":
        source_type = "webcam"
        source_content = {"type": source_type, "data": 0}
        return source_content


def blur_pipeline(source, model, blur_intensity, save_inplace):

    run_number = len(next(os.walk('results/'))[1])
    os.mkdir(f"results/run_{run_number}/")

    if source["type"] == "image_glob":
        

        for image_path in tqdm(source["data"],desc="Blurring images",position=0, leave=True):

            detections = yolov5_inference(image_path, model)
            if len(detections) > 0:
                blurred_image = blur(image_path, detections, blur_intensity, False, None)
                save_output(blurred_image, image_path, save_inplace, True, run_number)
        
        print(f"Results saved on results/run_{run_number}/")
        

    elif source["type"] == "image":
        image_path = source["data"]
        detections = yolov5_inference(image_path, model)
        if len(detections) > 0:
            blurred_image = blur(image_path, detections, blur_intensity, False, None)
            save_output(blurred_image, image_path, save_inplace, False, run_number)
            if save_inplace:
                print("Blur applied to original images")
            else:
                print(f"Results saved on results/run_{run_number}/")
        else:
            print("Nothing to blur")

        

    elif source["type"] == "video":
        video_inference(model, source["data"], False, blur_intensity, run_number)
        
    elif source["type"] == "webcam":
        video_inference(model, source["data"], True, blur_intensity, run_number)

def video_inference(model, video_path, is_webcam, blur_intensity, run_number):
    
    if is_webcam:
        video_capture = cv2.VideoCapture(0)
        output_path = f"results/{run_number}/webcam.avi"
    else:
        video_capture = cv2.VideoCapture(video_path)
        output_name = Path(video_path).name
        output_path = f"results/run_{run_number}/" + output_name[:-3]+"avi"

    fps = video_capture.get(cv2.CAP_PROP_FPS)
    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc("M", "J", "P", "G"), fps, (w, h))

    with tqdm(total=length, desc="Blurring Video") as pbar:
        while video_capture.isOpened():
            frame_is_read, frame = video_capture.read()
            if frame_is_read:
                detections = yolov5_inference(frame, model)
                if len(detections) > 0:
                    blurred_image = blur(None, detections, blur_intensity, True, frame)
                    out.write(blurred_image)
                else:
                    out.write(frame)
                pbar.update(1)
            else:
                break



def blur(image_path, detections, blur_intensity, is_video, frame):

    gaussian_kernel = (3+6*blur_intensity,3+6*blur_intensity)

    if not is_video:
        image = cv2.imread(image_path)
    else:
        image = frame

    for index, row in detections.iterrows():
        
        ROI = image[row["ymin"]:row["ymax"], row["xmin"]:row["xmax"]]
        blur = cv2.GaussianBlur(ROI, gaussian_kernel, 0)
        image[row["ymin"]:row["ymax"], row["xmin"]:row["xmax"]] = blur

    return image

def save_output(image, image_path, save_inplace, is_glob, run_number):
    if save_inplace:
        cv2.imwrite(image_path, image)
    elif is_glob:
        output_name = Path(image_path).name
        output_path = f"results/run_{run_number}/" + output_name
        cv2.imwrite(output_path, image)
    else:
        output_name = Path(image_path).name
        output_path = f"results/run_{run_number}/" + output_name
        cv2.imwrite(output_path, image)
