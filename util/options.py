import argparse

def parse_args():
    parser = argparse.ArgumentParser(formatter_class= argparse.RawTextHelpFormatter)

    parser.add_argument("--source", action = "store", dest = "source", type = str,
                           default = "demo/street.mp4", help = "Defines the content to be blurred\n\nExpected inputs:\n\n   /path/to/files/*.extension - glob of images\n   /path/to/file - individual image or video \n   0 - Webcam\n\n Accepted image formats: bmp, dng, jpeg, jpg, mpo, png, tif, tiff, webp, pfm\n Accepted Video formats: asf, avi, gif, m4v, mkv, mov, mp4, mpeg, mpg, ts, wmv\n\n (default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--blur-intensity", action = "store", dest = "blur_intensity", type = int,
                           default = 5, help = "Adjust the blur intensity, in a integer scale.\n\n(default: %(default)s) \n\n", required = False,)
    
    parser.add_argument("--save-inplace", action = "store_true", dest = "save_inplace", 
                           default = False, help = "If selected saves the blur on the original image, without creating a copy.\n\n(default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--model", action = "store", dest = "model", type = str,
                           default = "yolov5", help = "Specifies the model used on inference, weights must be of the same model.\n\nSupported models: YoloV5\n\n(default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--weights", action = "store", dest = "weights", type = str,
                           default = "weights/plates_and_faces.pt", help = "Defines the path of the weights used on inference.\nExpected input /path/to/weights.pt \n\n(default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--confidence", action = "store", dest = "confidence", type = float,
                           default = 0.4, help = "Specifies confidence threshold, from 0 to 1, in the YoloV5 inference.\n\n(default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--iou", action = "store", dest = "iou", type = float,
                           default = 0.45, help = "Specifies iou threshold, from 0 to 1, in the YoloV5 inference.\n\n(default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--classes2blur", dest = "classes", nargs="+", type=int,
                           default = [0, 1], help = "List of classes to detected and blurred. \n\nExpected input:\n\n   0 1 2 3 - Blur classes from 0 to 3\n\n(default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--max_det", action = "store", dest = "max_det", type = int,
                           default = 300, help = "Limits the number of detections per inference, in the YoloV5 inference.\n\n(default: %(default)s)\n\n", required = False,)
    
    parser.add_argument("--device", action = "store", dest = "device", type = str,
                           default = "GPU", help = "Device to be used on YoloV5 inference.\n\nExpected inputs:\n\n   GPU - runs inference in the first GPU of the system\n   CPU - runs inference on CPU\n   0,1,2... - runs inference on N GPU\n\n(default: %(default)s)\n\n", required = False,)
    
    args = parser.parse_args()
    return args