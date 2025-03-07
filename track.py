import argparse
import os
import json

# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
from pathlib import Path
from lf.gst_loader import LoadGstAppSink

import torch
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'strong_sort'))  # add strong_sort ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, check_img_size, non_max_suppression, scale_coords, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
from strong_sort.utils.parser import get_config
from strong_sort.strong_sort import StrongSORT

# remove duplicated stream handler to avoid duplicated logging
logging.getLogger().removeHandler(logging.getLogger().handlers[0])


@torch.no_grad()
def run(
        source='0',
        gst_source=None, # gstreamer pipeline
        output_file=None,  # output file
        truncate_output_file=False,
        quite=False,
        yolo_weights=WEIGHTS / 'yolov5m.pt',  # model.pt path(s),
        strong_sort_weights=WEIGHTS / 'osnet_x0_25_msmt17.pt',  # model.pt path,
        config_strongsort=ROOT / 'strong_sort/configs/strong_sort.yaml',
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        show_vid=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        save_vid=False,  # save confidences in --save-txt labels
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/track',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        hide_class=False,  # hide IDs
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        frame_mod=1,  # show every n=th frame
        threads=1,  # inference threads
        scale=1.0,  # video display scale
):

    if gst_source:
        is_gst = True
        is_file = False
        is_url = False
        webcam = False
        source = gst_source
        LOGGER.info(gst_source)
    else:
        is_gst = False
        source = str(source)
        save_img = not nosave and not source.endswith('.txt')  # save inference images
        is_file = Path(source).suffix[1:] in (VID_FORMATS)
        is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        if is_url and is_file:
            source = check_file(source)  # download

    output_file_handle = None

    if output_file:
        if truncate_output_file:
            output_file_handle = open(output_file, 'w', newline='\n')
        else:
            output_file_handle = open(output_file, 'a', newline='\n')

    # Directories
    if not isinstance(yolo_weights, list):  # single yolo model
        exp_name = str(yolo_weights).rsplit('/', 1)[-1].split('.')[0]
    elif type(yolo_weights) is list and len(yolo_weights) == 1:  # single models after --yolo_weights
        exp_name = yolo_weights[0].split(".")[0]
    else:  # multiple models after --yolo_weights
        exp_name = 'ensemble'
    exp_name = name if name is not None else exp_name + "_" + str(strong_sort_weights).split('/')[-1].split('.')[0]
    save_dir = increment_path(Path(project) / exp_name, exist_ok=exist_ok)  # increment run
    (save_dir / 'tracks' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(yolo_weights, device=device, dnn=dnn, data=None, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    if is_gst:
        show_vid = show_vid and check_imshow()
        dataset = LoadGstAppSink(gst_source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    elif webcam:
        show_vid = show_vid and check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = len(dataset)
    else:
        show_vid = show_vid and check_imshow()
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        nr_sources = 1

    vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

    # initialize StrongSORT
    cfg = get_config()
    cfg.merge_from_file(opt.config_strongsort)

    # Create as many strong sort instances as there are video sources
    strongsort_list = []
    for i in range(nr_sources):
        strongsort_list.append(
            StrongSORT(
                strong_sort_weights,
                device,
                max_dist=cfg.STRONGSORT.MAX_DIST,
                max_iou_distance=cfg.STRONGSORT.MAX_IOU_DISTANCE,
                max_age=cfg.STRONGSORT.MAX_AGE,
                n_init=cfg.STRONGSORT.N_INIT,
                nn_budget=cfg.STRONGSORT.NN_BUDGET,
                mc_lambda=cfg.STRONGSORT.MC_LAMBDA,
                ema_alpha=cfg.STRONGSORT.EMA_ALPHA,

            )
        )
    outputs = [None] * nr_sources

    is_quit = False         # Used to signal that quit is called
    is_paused = False       # Used to signal that pause is called
    frame_counter = 0
    cap = False
    show_inf_input = False

    if webcam:
        height = dataset.imgs[0].shape[0]
        width = dataset.imgs[0].shape[1]
    else:
        cap = dataset.cap
        # Video information
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        no_of_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    torch.set_num_threads(threads)

    threads = torch.get_num_threads()
    num_inter_threads = torch.get_num_interop_threads()

    LOGGER.info("pt inter thread: %d intra threads: %d", num_inter_threads, threads)

    # video controller
    def quit_key_action(**params):
        nonlocal is_quit
        is_quit = True

    def zoom_in_key_action(**params):
        nonlocal scale
        scale = scale * 1.1
        LOGGER.info('Zoom: %f', scale)

    def zoom_out_key_action(**params):
        nonlocal scale
        scale = scale / 1.1
        LOGGER.info('Zoom: %f', scale)

    def less_threads_key_action(**params):
        nonlocal threads
        threads = max(1, threads - 1)
        torch.set_num_threads(threads)
        LOGGER.info("pt inter thread: %d intra threads: %d", num_inter_threads, threads)

    def more_threads_key_action(**params):
        nonlocal threads
        threads = max(1, threads + 1)
        torch.set_num_threads(threads)
        LOGGER.info("pt inter thread: %d intra threads: %d", num_inter_threads, threads)

    def rewind_key_action(**params):
        if cap:
            nonlocal frame_counter
            nonlocal video_fps
            frame_counter = max(0, int(frame_counter - (video_fps * 5)))
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

    def forward_key_action(**params):
        if cap:
            nonlocal frame_counter
            nonlocal video_fps
            nonlocal no_of_frames
            frame_counter = min(int(frame_counter + (video_fps * 5)), no_of_frames - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

    def pause_key_action(**params):
        nonlocal is_paused
        is_paused = not is_paused

    def toggle_show_inf_key_action(**params):
        nonlocal show_inf_input
        show_inf_input = not show_inf_input
        if not show_inf_input:
            cv2.destroyWindow("Inference Input")

    # Map keys to buttons
    key_action_dict = {
        ord('q'): quit_key_action,
        ord('r'): rewind_key_action,
        ord('f'): forward_key_action,
        ord('p'): pause_key_action,
        ord(' '): pause_key_action,
        ord('+'): zoom_in_key_action,
        ord('-'): zoom_out_key_action,
        ord('l'): less_threads_key_action,
        ord('m'): more_threads_key_action,
        ord('w'): toggle_show_inf_key_action,
    }

    def key_action(_key):
        if _key in key_action_dict:
            key_action_dict[_key]()

    # Run tracking
    model.warmup(imgsz=(1 if pt else nr_sources, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0, 0.0], 0
    curr_frames, prev_frames = [None] * nr_sources, [None] * nr_sources
    for frame_idx, (path, im, im0s, vid_cap, s) in enumerate(dataset):
        if len(im.shape) == 3:
            im_display = im[::-1].transpose(1,2,0)
        elif len(im.shape) == 4:
            im_display = im[::-1].transpose(0,2,3,1)

        if is_quit:
            break
        elif is_paused:
            while is_paused and not is_quit:
                key = cv2.waitKey(1)  # 1 millisecond
                if key >= 0:
                    key_action(key)


        if (frame_idx % frame_mod) != 0:
            continue

        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        visualize = increment_path(save_dir / Path(path[0]).stem, mkdir=True) if opt.visualize else False
        pred = model(im, augment=opt.augment, visualize=visualize)
        t3 = time_sync()
        dt[1] += t3 - t2

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, opt.classes, opt.agnostic_nms, max_det=opt.max_det)
        dt[2] += time_sync() - t3

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            seen += 1
            if webcam:  # nr_sources >= 1
                p, im0, _ = path[i], im0s[i].copy(), dataset.count
                p = Path(p)  # to Path
                s += f'{i}: '
                txt_file_name = p.name
                save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            else:
                p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
                p = Path(p)  # to Path
                # video file
                if source.endswith(VID_FORMATS):
                    txt_file_name = p.stem
                    save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
                # folder with imgs
                else:
                    txt_file_name = p.parent.name  # get folder name containing current img
                    save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            curr_frames[i] = im0

            txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            s += '{}x{} [{}] '.format(im.shape[2], im.shape[3], frame_idx)  # print string
            imc = im0.copy() if save_crop else im0  # for save_crop

            annotator = Annotator(im0, line_width=2, pil=not ascii)
            if cfg.STRONGSORT.ECC:  # camera motion compensation
                strongsort_list[i].tracker.camera_update(prev_frames[i], curr_frames[i])

            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]

                # pass detections to strongsort
                t4 = time_sync()
                outputs[i] = strongsort_list[i].update(xywhs.cpu(), confs.cpu(), clss.cpu(), im0)
                t5 = time_sync()
                dt[3] += t5 - t4

                # json out
                # draw boxes for visualization
                if len(outputs[i]) > 0:
                    persons = []

                    for j, (output, conf) in enumerate(zip(outputs[i], confs)):
    
                        bboxes = output[0:4]
                        id = output[4]
                        cls = output[5]

                        if save_txt or output_file_handle:
                            # to MOT format
                            bbox_left = output[0]
                            bbox_top = output[1]
                            bbox_w = output[2] - output[0]
                            bbox_h = output[3] - output[1]
                            if output_file_handle:
                                person = {
                                    "confidence": conf.cpu().numpy().item(),
                                    "identity": int(id),
                                    "bbox": {"x": bbox_left, "y": bbox_top, "w": bbox_w, "h": bbox_h},
                                }
                                persons.append(person)
                            if save_txt:
                                # Write MOT compliant results to file
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                                                                   bbox_top, bbox_w, bbox_h, -1, -1, -1, i))

                        if save_vid or save_crop or show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            label = None if hide_labels else (f'{id} {names[c]}' if hide_conf else \
                                (f'{id} {conf:.2f}' if hide_class else f'{id} {names[c]} {conf:.2f}'))
                            annotator.box_label(bboxes, label, color=colors(c, True))
                            if save_crop:
                                txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                save_one_box(bboxes, imc, file=save_dir / 'crops' / txt_file_name / names[c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                    if output_file_handle:
                        out = {
                            "timestamp": t1,
                            "frame_id": frame_idx,
                            "predictions": {
                                "persons": persons
                            }
                        }

                        line = json.dumps(out)
                        output_file_handle.write(f'{line}\n')

                if not quite:
                    log_line = f'{s}Done. YOLO:({t3 - t2:.3f}s), StrongSORT:({t5 - t4:.3f}s)'
                    LOGGER.info(log_line)
            else:
                strongsort_list[i].increment_ages()
                if not quite:
                    LOGGER.info('No detections')

            # Stream results
            im0 = annotator.result()
            if show_vid and (frame_idx % frame_mod) == 0:
                if scale != 1.0:
                    new_width = int(im0.shape[1] * scale)
                    new_height = int(im0.shape[0] * scale)
                    new_dim = (new_width, new_height)
                    new_im0 = cv2.resize(im0, new_dim)
                    cv2.imshow(str(p), new_im0)
                else:
                    cv2.imshow(str(p), im0)
                if show_inf_input:
                    cv2.imshow("Inference Input", im_display)

                key = cv2.waitKey(1)  # 1 millisecond
                if key >= 0:
                    key_action(key)

            # Save results (image with detections)
            if save_vid:
                if vid_path[i] != save_path:  # new video
                    vid_path[i] = save_path
                    if isinstance(vid_writer[i], cv2.VideoWriter):
                        vid_writer[i].release()  # release previous video writer
                    if vid_cap:  # video
                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    else:  # stream
                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                    save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                    vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                vid_writer[i].write(im0)

            prev_frames[i] = curr_frames[i]

        if output_file_handle:
            output_file_handle.flush()


    if output_file_handle:
        output_file_handle.close()

    # Print results
    t = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms strong sort update per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_vid:
        s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(yolo_weights)  # update model (to fix SourceChangeWarning)


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--yolo-weights', nargs='+', type=str, default=WEIGHTS / 'yolov5m.pt', help='model.pt path(s)')
    parser.add_argument('--strong-sort-weights', type=str, default=WEIGHTS / 'osnet_x0_25_msmt17.pt')
    parser.add_argument('--config-strongsort', type=str, default='strong_sort/configs/strong_sort.yaml')
    parser.add_argument('--source', type=str, default='0', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--gst-source', type=str, default=None, help='GStreamer pipeline')
    parser.add_argument('--output-file', type=str, default=None, help='file to append the predictions, create if not exist yet.')
    parser.add_argument('--truncate-output-file', default=False, action='store_true', help='truncate the output file if exists.')
    parser.add_argument('--quite', default=False, action='store_true', help='quite on stdout.')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--show-vid', action='store_true', help='display tracking video results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--save-vid', action='store_true', help='save video tracking results')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    # class 0 is person, 1 is bycicle, 2 is car... 79 is oven
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/track', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--hide-class', default=False, action='store_true', help='hide IDs')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--frame_mod', type=int, default=1, help='show every n-th frame')
    parser.add_argument('--threads', type=int, default=1, help='number of inference threads')
    parser.add_argument('--scale', type=float, default=1.0, help='video display scale')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt

    # Supported engines/weight:
    #   PyTorch:              weights = *.pt
    #   TorchScript:                    *.torchscript
    #   ONNX Runtime:                   *.onnx
    #   ONNX OpenCV DNN:                *.onnx with --dnn
    #   OpenVINO:                       *.xml
    #   CoreML:                         *.mlmodel
    #   TensorRT:                       *.engine
    #   TensorFlow SavedModel:          *_saved_model
    #   TensorFlow GraphDef:            *.pb
    #   TensorFlow Lite:                *.tflite
    #   TensorFlow Edge TPU:            *_edgetpu.tflite

def main(opt):
    check_requirements(requirements=ROOT / 'requirements.txt', exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)