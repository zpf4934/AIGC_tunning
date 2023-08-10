
CUDA_VISIBLE_DEVICES="0,1" deepspeed train.py --deepspeed_config conf/zero2_offload.json --distributed deepspeed