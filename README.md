## 安装依赖
pip install -r requirements.txt
pip install torch==1.10.0+cu111 torchvision==0.11.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.10/index.html
pip install -e .

## 训练
python train_net.py --config-file cascade_layoutlmv3.yaml --num-gpus 1 \
        MODEL.WEIGHTS ./models/layoutlmv3-base-finetuned-publaynet/model_final.pth \
        OUTPUT_DIR ./models/layoutlmv3-base-finetuned-publaynet
## 测试
python train_net.py --config-file cascade_layoutlmv3.yaml --eval-only --num-gpus 1 \
        MODEL.WEIGHTS ./models/layoutlmv3-base-finetuned-publaynet/model_final.pth \
        OUTPUT_DIR ./models/layoutlmv3-base-finetuned-publaynet
## 修改
\examples\object_detection\cascade_layoutlmv3.yaml中修改参数
