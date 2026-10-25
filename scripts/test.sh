# python eval.py \
# --ckpt outputs/efficientnet_b0/best.pth \
# --image data/test/frame_000003.jpg \
# --image data/test/frame_000004.jpg

python eval.py \
--ckpt ckp-tag/best_weights_cls_mobilenet_v3.pth \
--config configs/mobilenet_v3_small.yaml \
--video /run/media/tao/56AE654DAE6526A9/迅雷下载/[电影天堂www.dytt89.com]血色将至-2007_蓝光中英双字.mp4.bt.xltd \
--frames-dir ./outputs/frames/there_will_be_blood \
--output outputs/cls_results_there_will_be_blood.jsonl \
--batch-size 256 \
--interval 0.1 