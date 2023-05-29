from detectron2.data import MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import  default_argument_parser, default_setup, launch, DefaultPredictor
# from detectron2.config import CfgNode as CN
from detectron2.utils.visualizer import Visualizer
from ditod.config import add_vit_config
import cv2
# import pickle
import json
import torch
import glob


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    dic = {}
    dic[0] = '__background__'
    dic[1] = 'Text'
    dic[2] = 'Title'
    dic[3] = 'Figure'
    dic[4] = 'Figure caption'
    dic[5] = 'Table'
    dic[6] = 'Table caption'
    dic[7] = 'Header'
    dic[8] = 'Footer'
    dic[9] = 'Reference'
    dic[10] = 'Equation'

    for i, file in enumerate(glob.glob("./picture/*.jpg",)):
        # print(file)
        image = cv2.imread(file)
        print(file)

        # one slice each time
        predictor = DefaultPredictor(cfg)
        outputs = predictor(image)

        boxes = outputs["instances"].pred_boxes
        boxes = boxes.to('cpu')
        labels = outputs["instances"].pred_classes
        labels = labels.to('cpu').tolist()
        print(labels)
        print(dic)

        for j in range(0, len(labels)):
            labels[j] = dic[labels[j]]

        v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        out = v.overlay_instances(boxes=boxes, labels=labels)

        # # [optional] visualize the results and save
        # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
        # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("./output/{}.jpg".format(i), out.get_image()[:, :, ::-1])
    # print(file)
    # image = cv2.imread('./test.jpg')
    #
    # # one slice each time
    # predictor = DefaultPredictor(cfg)
    # outputs = predictor(image)
    #
    # boxes = outputs["instances"].pred_boxes
    # boxes = boxes.to('cpu')
    # labels = outputs["instances"].pred_classes
    # labels = labels.to('cpu').tolist()
    # print(labels)
    # print(dic)
    #
    # for j in range(0, len(labels)):
    #     labels[j] = dic[labels[j]]
    #
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # out = v.overlay_instances(boxes=boxes, labels=labels)
    #
    # # # [optional] visualize the results and save
    # # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    # # out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # cv2.imwrite("./out.jpg", out.get_image()[:, :, ::-1])
    # print("done!")



if __name__ == '__main__':

    parser = default_argument_parser()
    parser.add_argument("--debug", action="store_true", help="enable debug mode")
    args = parser.parse_args()
    print("Command Line Args:", args)

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )