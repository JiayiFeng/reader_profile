export CUDA_VISIBLE_DEVICES=0
python trans_to_recordio.py
python recordio_vgg16.py
