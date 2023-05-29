import sys
import os
from PyPDF2 import PdfFileWriter, PdfFileReader
import fitz  # package-name: PyMuPDF
from pdf2image import convert_from_path
import numpy as np
import cv2
from detectron2.data import transforms as T
from detectron2.utils.visualizer import Visualizer
from ditod import add_vit_config
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch, DefaultPredictor
from detectron2.data import MetadataCatalog

# inpfn = 'SVR.pdf'
# outfn = 'out1.pdf'

jpg_save_path = './pdf2jpg/'
pdf_path = './Crossformer.pdf'


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


def getFileList(dir, Filelist, ext=None):
    """
    获取文件夹及其子文件夹中文件列表
    输入 dir：文件夹根目录
    输入 ext: 扩展名
    返回： 文件路径列表
    """
    newDir = dir
    if os.path.isfile(dir):
        if ext is None:
            Filelist.append(dir)
        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):
        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def pdf_to_jpg(pdf_path, jpg_save_path, one_page=False):
    """
        将PDF文件转换为JPG图片
    :param pdf_path: PDF文件路径
    :param jpg_save_path: JPG图片保存路径， 若one_page为False，输入一个文件夹路径保存所有JPG图片
    :param one_page: True：返回 PDF首页的JPG；False：返回 PDF所有页的JPG
    :return:
    """
    pages = convert_from_path(pdf_path, 200)  # 若pdf有多页，则返回一个列表
    if one_page:
        pages[0].save(jpg_save_path, 'JPEG')
    else:
        if not os.path.exists(jpg_save_path):
            os.mkdir(jpg_save_path)     # 创建文件夹
        for i, page in enumerate(pages):
            page.save(jpg_save_path + str(i+1) + '.jpg', 'JPEG')


def main(args):

    # # 先将读入的pdf转化为jpg图片
    # pdf_to_jpg(pdf_path, jpg_save_path)
    #
    # imglist = getFileList(jpg_save_path, [], 'jpg')
    # print('本次执行检索到 ' + str(len(imglist)) + ' 张图像\n')


    cfg = setup(args)
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    sample_style = "choice"
    tfm_gens = [T.ResizeShortestEdge(min_size, max_size, sample_style)]

    # for imgpath in imglist:
    #     imgname = os.path.splitext(os.path.basename(imgpath))[0]
    #     image = cv2.imread(imgpath)
    #
    #     image, transforms = T.apply_transform_gens(tfm_gens, image)
    #     print(image.shape)
    #     # 构建一个end-to-end单次预测一张图片的
    #     predictor = DefaultPredictor(cfg)
    #     outputs = predictor(image)
    #     print(outputs["instances"].pred_classes)
    #     print(outputs["instances"].pred_boxes)
    #
    #     v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2)
    #     out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #     cv2.imwrite("./outputs/{}.jpg".format(imgname), out.get_image()[:, :, ::-1])

    image = cv2.imread("./test.jpg")

    image, transforms = T.apply_transform_gens(tfm_gens, image)
    print(image.shape)
    # 构建一个end-to-end单次预测一张图片的
    predictor = DefaultPredictor(cfg)
    outputs = predictor(image)
    print(outputs["instances"].pred_classes)
    print(outputs["instances"].pred_boxes)
    for i in range(outputs["instances"].pred_classes.shape[0]):
        temp = outputs["instances"].pred_classes[i]
        if( temp == 7 or temp == 8):
            # erase_box (左上角x, 左上角y, 宽度, 高度)
            erase_box = outputs["instances"].pred_boxes[i]
            print(erase_box)
            # x = erase_box[0]
            # y = erase_box[1]
            # for i in range(x, x+erase_box[2]+1):
            #     for j in range(y, y+erase_box[3]+1):
            #         # cv2中image坐标(y,x,通道数)
            #         image.itemset((j, i, 0), 0)
            #         image.itemset((j, i, 1), 0)
            #         image.itemset((j, i, 2), 0)

    # cv2.imwrite("./outputs/test_out.jpg", image)





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

# def imshow(mt):
# 	cv2.imshow('cc', mt)
# 	cv2.waitKey(0)
#
# # 返回宽和高
# def GetPageBox(page):
#     return page.mediaBox.getUpperRight_x(), page.mediaBox.getUpperRight_y()
#
# # 以test.pdf为例
# # numpages: 5
# # {(612, 792): [0, 1, 2, 3, 4]}
# def GetPageGroups(reader):
#     groups = {}
#     npages = reader.getNumPages()
#     for i in range(npages):
#         page = reader.getPage(i)
#         w, h = GetPageBox(page)
#         groups.setdefault((int(w), int(h)), []).append(i)
#     return groups
#
# def page2cv(doc, pid):
#     pix = doc.get_page_pixmap(pid, alpha=False)
#     im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
#     im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
#     return im
#
# def BinaryImg(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 25, 10)
#     return binary
#
#
# def compare_img_hist(img1, img2):
#     img1_hist = cv2.calcHist([img1], [1], None, [256], [0, 256])
#     img1_hist = cv2.normalize(img1_hist, img1_hist, 0, 1, cv2.NORM_MINMAX, -1)
#     img2_hist = cv2.calcHist([img2], [1], None, [256], [0, 256])
#     img2_hist = cv2.normalize(img2_hist, img2_hist, 0, 1, cv2.NORM_MINMAX, -1)
#     similarity = cv2.compareHist(img1_hist, img2_hist, 0)
#     return similarity
#
# def compare_img(img1, img2):
#     same = ((img1 - img2) == 0).sum()
#     return same / img1.size
#
#
# def TestMargin(imgs, func):
#     iims = list(map(func, imgs))
#     goods = []
#     for i in range(len(imgs)):
#         for j in range(i+1, i+4):
#             if j >= len(imgs): continue
#             im1, im2 = iims[i], iims[j]
#             sim = compare_img(im1, im2)
#             if sim > 0.99: goods.append(1)
#             if len(goods) > 5: return True
#     return False
#
# def GetCandPos(arr):
#     st = 0; ret = []
#     for i in range(3, len(arr)):
#         if arr[i] == 0 and arr[i-1] > 0: st = i
#         elif arr[i] > 0 and arr[i-1] == 0:
#             ret.append((st, i))
#     ret.append( (st, len(arr)+1) )
#     ret = [(u+v)//2 for u,v in ret]
#     return ret
#
# def FindCandicateLines(imgs):
#     if len(imgs) <= 1: return [-1]*4
#     bins = list(map(BinaryImg, imgs))
#     # 这步是在干啥?
#     if len(bins) > 10: bins = bins[:5] + bins[-5:]
#
#     bins = np.concatenate([x[None,...] for x in bins], 0)
#     bins = (bins == 0) * 1
#     cmap = bins.max(axis=0)
#     hh, vv = cmap.max(axis=1), cmap.max(axis=0)
#     hpos = GetCandPos(hh)
#     vpos = GetCandPos(vv)
#     img = imgs[-2]
#     ht, hb, vl, vr = [0], [img.shape[0]], [0], [img.shape[1]]
#     for h in hpos:
#         if TestMargin(imgs, lambda im:im[:h,:,:]): ht.append(h)
#         if TestMargin(imgs, lambda im:im[h:,:,:]): hb.append(h)
#     for v in vpos:
#         if TestMargin(imgs, lambda im:im[:,:v,:]): vl.append(v)
#         if TestMargin(imgs, lambda im:im[:,v:,:]): vr.append(v)
#
#     if verbose:
#         print(ht)
#         print(hb)
#         print(vl)
#         print(vr)
#         for h in ht+hb: cv2.line(img, (0, h), (img.shape[1], h), (0, 0, 255))
#         for v in vl+vr: cv2.line(img, (v, 0), (v, img.shape[0]), (0, 0, 255))
#         imshow(img)
#     return max(ht), min(hb), max(vl), min(vr)
#
# def CutGroup(imgs, groups, wh):
#     w, h = wh; group = groups[wh]
#     gimgs = [imgs[i] for i in group]
#     ht, hb, vl, vr = FindCandicateLines(gimgs)
#     if ht == -1: return None, None
#     lowerLeft = (vl, h-hb)
#     upperRight = (vr, h-ht)
#     return lowerLeft, upperRight
#
# verbose = False
#
# def CropPDF(inpfn, outfn):
#     doc = fitz.open(inpfn)
#     npages = doc.page_count
#     imgs = [page2cv(doc, i) for i in range(npages)]
#     with open(inpfn, 'rb') as fin:
#         reader = PdfFileReader(fin)
#         writer = PdfFileWriter()
#         npages = reader.getNumPages()
#         print(f'numpages: {npages}')
#         groups = GetPageGroups(reader)
#         for wh, iis in groups.items(): print(wh, iis)
#         allcuts = {}
#         for wh, iis in groups.items():
#             ll, ur = CutGroup(imgs, groups, wh)
#             if ll is None: continue
#             for ii in iis: allcuts[ii] = (ll, ur)
#         for i in range(npages):
#             page = reader.getPage(i)
#             if i in allcuts:
#                 ll, ur = allcuts[i]
#                 page.cropBox.lowerLeft = ll
#                 page.cropBox.upperRight = ur
#                 page.mediaBox = page.cropBox
#             writer.addPage(page)
#         with open(outfn, "wb") as fout: writer.write(fout)



