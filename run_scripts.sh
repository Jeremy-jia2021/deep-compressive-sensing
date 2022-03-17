#export TMPDIR=/data/jia/Networks/ESRGAN/CK_ESRGAN/tb_logger
#tensorboard --logdir=/data/jia/Networks/ESRGAN/CK_ESRGAN/tb_logger --port=6020
http://localhost:16020/


tensorboard --logdir=/mnt/home/jeremy/networks/CS-GAN-test1-tju/tb_logger/c-net_21 --port=6001
tensorboard --logdir=/mnt/home/jeremy/networks/CS-GAN-test1-tju/tb_logger/optimization --port=6002

rm -r /data/jia/Networks/single-pixel/CS-GAN-test1-tju/tb_logger/optimization

python train.py -opt options/train.yml

python main_exp.py -opt ../options/train.yml
source activate Cuda_15
