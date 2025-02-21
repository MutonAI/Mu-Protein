HERE=$(cd "$(dirname "$0")";pwd)

USER_DIR=$HERE/../src/protein

DATA_DIR=/mnt/data/generated_data/uniref50_2018_03/ # Path to the preprocess data bin files by fairseq

MAX_EPOCH=1000 # change this
WARMUP_UPDATES=20000
TOTAL_UPDATES=2000000
PEAK_LR=0.0001
TOKENS_PER_SAMPLE=768
MAX_POSITIONS=768
MAX_TOEKNS=768
UPDATE_FREQ=16
LOG_INTERVAL=10
SAVE_INTERVAL=1
NUM_WORKERS=0
VALID_SUBSET=valid
DDP_BACKEND=no_c10d
BEST_CHECKPOINT_METRIC=loss

TASK=prot_pmlm
CRIT=prot_pmlm
ARCH=prot_pmlm_1b

PRETRAIN_TASK=pcomb

LOG_DIR=$HERE/../log/

CHECKPOINT_DIR=$HERE/../ckpt/ur50-$PRETRAIN_TASK-$ARCH-ckpt/

rm -rf $LOG_DIR
mkdir -p $LOG_DIR

fairseq-train $DATA_DIR --fp16 \
  --fix-batches-to-gpus \
  --distributed-no-spawn \
  --task $TASK \
  --criterion $CRIT \
  --arch $ARCH \
  --optimizer adam \
  --adam-betas '(0.9,0.98)' \
  --adam-eps 1e-6 --clip-norm 1.0 \
  --lr-scheduler polynomial_decay \
  --lr $PEAK_LR \
  --total-num-update $TOTAL_UPDATES \
  --warmup-updates $WARMUP_UPDATES \
  --update-freq $UPDATE_FREQ \
  --dropout 0.1 \
  --weight-decay 0.01 \
  --tokens-per-sample $TOKENS_PER_SAMPLE \
  --max-positions $MAX_POSITIONS \
  --max-tokens $MAX_TOEKNS \
  --max-epoch $MAX_EPOCH \
  --log-format simple \
  --log-interval $LOG_INTERVAL \
  --valid-subset $VALID_SUBSET \
  --save-interval $SAVE_INTERVAL \
  --save-interval-updates 1000 \
  --keep-interval-updates 3 \
  --best-checkpoint-metric $BEST_CHECKPOINT_METRIC \
  --ddp-backend=$DDP_BACKEND \
  --tensorboard-logdir $LOG_DIR \
  --num-workers $NUM_WORKERS \
  --save-dir $CHECKPOINT_DIR \
  --sample-break-mode eos \
  --skip-invalid-size-inputs-valid-test \
  --pretrain-task $PRETRAIN_TASK \
  --user-dir $USER_DIR | tee $LOG_DIR/log.txt

