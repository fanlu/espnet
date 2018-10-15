. ./job_config.sh
LOGS=$PWD/logs3
OUTDIR=$PWD/output

sgeopt="-q g.q -l h_vmem=240G -l mem_free=40G" # -l h=\"GPU_172_21_*&!GPU_172_21_57_132\"" 
lr=0.001
mtype='lstm'
rsize=512
cost=xent
epochs=4
bs=512
ly=3

gn=2
cnorm="l2"


num_gpu=2
cxt="0"
dp=0
mtype='tdnn'

bs=512
logi=200
vchk=20000
mupdate=300000

epochs=8 
lrdecay=4
tdnn_def="512_5_1.512_3_2.512_3_3" 
dp=0  
lrr=0.7
cnorm="l2"
otype='ave'

#revisit lstm
cost='xent'
mtype='tdnn'

Resdir=./resources
for lr in 0.004 ;
do
    for init_factor in 0.5 ;
do  
    for dp in 0 ;
    do
    frmnn="512.R.B.1500.R.B"
    if [[ $otype == "end" ]]; then
        frmnn="None"
    fi
    num_worker=$(( num_gpu * 3 ))
    num_slots=$(( num_worker * 2 )) #$(( num_worker * 2 + $num_gpu ))
    gpuopt=" -l gpu=$num_gpu -pe smp $num_gpu"
    ekey=expt02.$num_gpu.pipe.$cost.mtype.$mtype.$tdnn_def.$otype.lr$lr.lrr$lrr.batchsize$bs.epochs$epochs.cosnorm$cnorm.gn$gn.cxt$cxt.dp$dp.init$init_factor.lrdecay$lrdecay
    mkdir -p $OUTDIR/$ekey
    if  [ ! -e $LOGS/$ekey.log ]; then
    perl ./utils/queue.pl $sgeopt $gpuopt $LOGS/$ekey.log bash scripts/run_mgpu.sh $num_gpu python scripts/spkid_trainer.py --lr $lr --lrr $lrr --optim sgd --out-type $otype  --feat-dim 23 --model-dir $OUTDIR/$ekey --rnn-layer $ly --train-data $Resdir/train.list  --valid-data  $Resdir/valid_large.list --model-type $mtype --rnn-size $rsize --epochs $epochs --cost $cost --costfactor 0.5 --grad_norm $gn --cos-norm $cnorm --batch-size $bs --num-spk $num_spk --num-load-workers $num_worker --num-gpu $num_gpu  --feat-cxt "\"[$cxt]\"" --dropout $dp --tdnn-def $tdnn_def --log-interval $logi --valid-check $vchk --max-update $mupdate --init-param-ratio $init_factor --lr-decay-check $lrdecay & 
    sleep 60 
    fi
    done
done

done
exit

for lr in 0.3 ;
do
    for init_factor in 1 ;
do
    for dp in 0.1 ;
    do
    frmnn="512.R.B.1500.R.B"
    if [[ $otype == "end" ]]; then
        frmnn="None"
    fi
    num_worker=$(( num_gpu * 3 ))
    num_slots=$(( num_worker * 2 )) #$(( num_worker * 2 + $num_gpu ))
    gpuopt=" -l gpu=$num_gpu -pe smp $num_gpu"
    ekey=expt02.$num_gpu.pipe.$cost.mtype.$mtype.$tdnn_def.$otype.lr$lr.lrr$lrr.batchsize$bs.epochs$epochs.cosnorm$cnorm.gn$gn.cxt$cxt.dp$dp.init$init_factor.lrdecay$lrdecay
    mkdir -p $OUTDIR/$ekey
    if  [ ! -e $LOGS/$ekey.log ]; then
    perl ./utils/queue.pl $sgeopt $gpuopt $LOGS/$ekey.log bash scripts/run_mgpu.sh $num_gpu python scripts/spkid_trainer.py --lr $lr --lrr $lrr --optim sgd --out-type $otype  --feat-dim 23 --model-dir $OUTDIR/$ekey --rnn-layer $ly --train-data $Resdir/train.list  --valid-data  $Resdir/valid_large.list --model-type $mtype --rnn-size $rsize --epochs $epochs --cost $cost --costfactor 0.5 --grad_norm $gn --cos-norm $cnorm --batch-size $bs --num-spk $num_spk --num-load-workers $num_worker --num-gpu $num_gpu  --feat-cxt "\"[$cxt]\"" --dropout $dp --tdnn-def $tdnn_def --log-interval $logi --valid-check $vchk --max-update $mupdate --init-param-ratio $init_factor --lr-decay-check $lrdecay &
    sleep 60
    fi
    done
done

done

