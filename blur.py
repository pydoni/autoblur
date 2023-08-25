from networks.yolov5 import yolov5_model_config
from util.general import blur_pipeline, get_source
from util.options import parse_args



if __name__ == "__main__":
    args = parse_args()
    source = get_source(args.source)

    print("Autoblur V1")
    print("Tool to automatic blur elements in images/videos")
    if args.model == "yolov5":
        print("Loading Model...")
        model = yolov5_model_config(args.weights, args.confidence, args.iou, args.classes, args.max_det, args.device)
        
    else:
        print("error: Model not supported yet, only Yolov5 is supported")
        raise KeyError

    blur_pipeline(source, model, args.blur_intensity, args.save_inplace)