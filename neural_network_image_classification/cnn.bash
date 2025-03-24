#!/bin/bash

model_name=b

echo "=== Starting CNN training process ==="
echo "Model name: ${model_name}"

echo "[1/4] Fitting initial model..."
time ./cnn_classification.py cnn-fit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 1

echo "[1/4] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib
mv plots/${model_name}.learning_curve.png plots/${model_name}.learning_curve-1.png

echo "[2/4] Refitting model with Batch 2..."
time ./cnn_classification.py cnn-refit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 2

echo "[2/4] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib
mv plots/${model_name}.learning_curve.png plots/${model_name}.learning_curve-2.png

echo "[3/4] Refitting model with Batch 3..."
time ./cnn_classification.py cnn-refit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 3

echo "[3/4] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib

mv plots/${model_name}.learning_curve.png plots/${model_name}..learning_curve-3.png

echo "[4/4] Refitting model with Batch 4..."
time ./cnn_classification.py cnn-refit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 4

echo "[4/4] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib
mv plots/${model_name}.blearning_curve.png plots/${model_name}.learning_curve-4.png

echo "Generating score"
time ./cnn_classification.py score \
     --model-file models/${model_name}.joblib \
     --batch-number 5

<<'COMMENT'
THIS IS THE TEST RUN TO USE AS REFERENC
echo "[1/3] Fitting initial model..."
time ./cnn_classification.py cnn-fit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 1

echo "[1/3] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib
mv plots/${model_name}.joblib.learning_curve.png plots/${model_name}.joblib.learning_curve-1.png

echo "[2/3] Refitting model with Batch 2..."
time ./cnn_classification.py cnn-refit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 2

echo "[2/3] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib
mv plots/${model_name}.joblib.learning_curve.png plots/${model_name}.joblib.learning_curve-2.png

echo "[3/3] Refitting model with Batch 3..."
time ./cnn_classification.py cnn-refit \
     --model-name ${model_name} --model-file models/${model_name}.joblib \
     --batch-number 3

echo "[3/3] Generating learning curve..."
time ./cnn_classification.py learning-curve \
     --model-file models/${model_name}.joblib
mv plots/${model_name}.joblib.learning_curve.png plots/${model_name}.joblib.learning_curve-3.png

echo "Generating score"
time ./cnn_classification.py score \
     --model-file models/${model_name}.joblib \
     --batch-number 6
COMMENT