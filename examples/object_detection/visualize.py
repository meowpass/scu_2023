import os
import itertools

import torch

from typing import Any, Dict, List, Set

from detectron2.data import build_detection_train_loader, MetadataCatalog, DatasetCatalog
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.solver.build import maybe_add_gradient_clipping

from ditod import add_vit_config
from ditod import DetrDatasetMapper

from detectron2.data.datasets import register_coco_instances
import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm
from detectron2.engine.defaults import create_ddp_model
import weakref
from detectron2.engine.train_loop import AMPTrainer, SimpleTrainer
from ditod import MyDetectionCheckpointer, ICDAREvaluator
from ditod import MyTrainer

from detectron2.utils.visualizer import Visualizer
import cv2
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from PIL import Image
from torchvision import transforms
import json

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # add_coat_config(cfg)
    add_vit_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)

    # model = MyTrainer.build_model(cfg)
    # DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
    #     cfg.MODEL.WEIGHTS, resume=args.resume
    # )

    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    sample_style = "choice"
    tfm_gens = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

    # image = utils.read_image("./datasets/CDLA/val/val_0014.jpg","RGB")
    # image = Image.open("./datasets/CDLA/val/val_0014.jpg")
    image = cv2.imread("./test.jpg")

    # pre_pross = transforms.Compose([transforms.Resize((224, 224)),
    #                                  transforms.ToTensor(),
    #                                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    #                                 )
    # image = pre_pross(image)


    image, transforms = T.apply_transform_gens(tfm_gens, image)
    print(image.shape)
    # 构建一个end-to-end单次预测一张图片的
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    # print(outputs["instances"].pred_classes)
    # print(outputs["instances"].pred_boxes)
    # print(len(outputs["instances"].pred_boxes))

    with open('test.json','w') as fp:
        json.dump(outputs, fp)



    v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite("test_out.jpg", out.get_image()[:, :, ::-1])

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