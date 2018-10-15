#export PATH="/opt/cephfs1/asr/users/yun.tang/miniconda3/bin:$PATH"
echo $@
export PATH="/mnt/cephfs2/asr/users/fanlu/miniconda3/envs/py2/bin:$PATH"
tot_gpus=`nvidia-smi -q|grep "Attached GPUs"|awk '{print $NF-1}'`
gpu_ids=`nvidia-smi -q |grep -e "Minor" -e "Process ID" | grep -B 1 Process|grep Minor|awk '{print $NF}'`
use_gpu=-1
need_gpu=$1
echo "need_gpu:${need_gpu}"
for i in `seq 0 ${tot_gpus}` ; do
  if [[ ! "${gpu_ids}" =~ "${i}" ]]; then 
    if [ $use_gpu = -1 ] ; then
	use_gpu=$i
    else
    	use_gpu="$use_gpu,$i"
    fi
    need_gpu=$(( need_gpu - 1 ))
    if [ $need_gpu = 0 ]; then
    	break
    fi
  fi
done
if [ ! $need_gpu = 0 ]; then
  hostname
  echo "Error! There is not enough GPU cards available"
  exit 1
fi
echo use gpu card $use_gpu
export CUDA_VISIBLE_DEVICES=$use_gpu
shift
$@
