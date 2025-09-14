CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.run \
  --nproc_per_node=8 \
  --master_addr=127.0.0.1 \
  --master_port=29555 \
  src/llamafactory/launcher.py \
  examples/train_full/output_reverse.yaml \


###评估
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python eval.py --model_path /data