# IWSLT 2023 Simultaneous Speech Translation on MuST-C 2.0

This is a short tutorial on training an ST model and evaluating it using the wait-k policy

## Data Preparation
This section covers the data preparation required for training and evaluation
If you are only interested in model inference / evaluation, please jump to the [Inference & Evaluation](#inference--evaluation) section

[Download](https://ict.fbk.eu/must-c-release-v2-0/) and unpack the MuST-C 2.0 data to the path
`${MUSTC_ROOT}/en-${TARGET_LANG}`. Then run the following commands below to preprocess the data
```bash
# additional python packages for S2T data processing / model training
pip install pandas torchaudio sentencepiece

# generate TSV manifests
cd fairseq

python examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTC_ROOT} --task st \
  --vocab-type unigram --vocab-size 10000 \
  --use-audio-input
```

## Pretrained Encoder & Decoder
This section covers open-sourced pretrained encoders and decoders
If you already have your own pretrained encoder / decoder, please jump to the next section

For pretrained encoder, we used a [wav2vec 2.0 model](https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small_960h.pt) opensourced by the [original wav2vec 2.0 paper](https://arxiv.org/abs/2006.11477). Download and extract this model to `${MUSTC_ROOT}/en-${TARGET_LANG}/wav2vec_small_960h.pt`

For pretrained decoder, we used an [mBART model](https://dl.fbaipublicfiles.com/fairseq/models/mbart/mbart.cc25.v2.tar.gz) opensourced by the [original mBART paper](https://arxiv.org/abs/2001.08210). Download and extract the model to `${MUSTC_ROOT}/en-${TARGET_LANG}/model.pt`, the dict to `${MUSTC_ROOT}/en-${TARGET_LANG}/dict.txt` and the sentencepiece model to `${MUSTC_ROOT}/en-${TARGET_LANG}/sentence.bpe.model`

If using the above mBART model, in `${MUSTC_ROOT}/en-${TARGET_LANG}/config_st.yaml`, set the "sentencepiece_model" parameter (under "bpe_tokenizer") to "sentence.bpe.model" and the "vocab_filename" parameter to "dict.txt"

## Training
This section covers training an offline ST model
Set ${ST_SAVE_DIR} to be the save directory of the resulting ST model. This train command assumes that you are training on one GPU, so please adjust the "update-freq" command accordingly. 

```bash
 fairseq-train ${MUSTC_ROOT}/en-${TARGET_LANG} \
        --config-yaml config_st.yaml --train-subset train_st --valid-subset dev_st \
        --save-dir ${ST_SAVE_DIR} --num-workers 1  \
        --optimizer adam --lr 0.0001 --lr-scheduler inverse_sqrt --clip-norm 10.0 \
        --criterion label_smoothed_cross_entropy \
        --warmup-updates 2000 --max-update 30000 --max-tokens 1024 --seed 1 \
        --freeze-finetune-updates 0 \
        --w2v-path ${MUSTC_ROOT}/en-${TARGET_LANG}/wav2vec_small_960h.pt \
        --load-pretrained-decoder-from ${MUSTC_ROOT}/en-${TARGET_LANG}/model.pt \
        --decoder-normalize-before --share-decoder-input-output-embed \
        --finetune-w2v-params all --finetune-decoder-params encoder_attn,layer_norm,self_attn \
        --task speech_to_text  \
        --arch xm_transformer  \
        --adaptor-proj --fp16 \
        --update-freq 64 
```

## Inference & Evaluation (TODO: waiting to update --agent value)
This section covers simultaneous evaluation using the wait-k policy.
[SimulEval](https://github.com/facebookresearch/SimulEval) is used for evaluation. In the following command, we evaluate the best checkpoint from the [Training](#training) section. The init-target-token we used for training was "</s>". For the wait-k policy, we use k=8 and step=5. Evaluation results will be stored at ${OUTPUT_DIR}.

```
git clone https://github.com/facebookresearch/SimulEval.git
cd SimulEval
pip install -e .

k=8
step=5

simuleval \
    --agent test_time_waitk_s2t.py
    --dataloader fairseq_s2t \
    --fairseq-data ${MUSTC_ROOT}/en-${TARGET_LANG} \
    --fairseq-config ${MUSTC_ROOT}/en-${TARGET_LANG}/config_st.yaml \
    --fairseq-gen-subset tst_COMMON_st \
    --checkpoint ${ST_SAVE_DIR}/checkpoint_best.pt \
    --sentencepiece-model ${MUSTC_ROOT}/en-${TARGET_LANG}/sentence.bpe.model \
    --output ${OUTPUT_DIR} \
    --device cuda:0 \
    --source-segment-size `python -c "print(int(${step} * 40))"` \
    --waitk-lagging ${k} \
    --init-target-token "</s>" \
    --fixed-pre-decision-ratio ${step} \
```

The evaluation result on `tst-COMMON` is:
```bash
BLEU       AL    AP      DAL
13.762 5112.337 0.798 5061.954
```
