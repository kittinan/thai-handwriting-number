#!/bin/bash
export BUCKET_NAME=kittinan
export JOB_NAME="thai_number_train_$(date +%Y%m%d_%H%M%S)"
export JOB_DIR=gs://$BUCKET_NAME/$JOB_NAME
#export REGION=asia-east1
export REGION=us-east1

gcloud ml-engine jobs submit training $JOB_NAME \
    --job-dir $JOB_DIR \
    --runtime-version 1.2 \
    --module-name trainer.model \
    --package-path ./trainer \
    --region $REGION \
	--config config.yaml \
    -- \
    --train-file gs://$BUCKET_NAME/thainumber_28.pkl