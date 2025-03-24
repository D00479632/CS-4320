# Process I followed 

I got all the code with the dataset fashion mnist working and I ran the cnn.bash
to see if it works and how long it takes.

```bash
./cnn.bash
=== Starting CNN training process ===
Model name: a
[1/3] Fitting initial model...
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 28, 28, 64)          │           3,200 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 14, 14, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 14, 14, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 14, 14, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 7, 7, 128)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 6272)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │         401,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │             650 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 626,762 (2.39 MB)
 Trainable params: 626,762 (2.39 MB)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 31s 4ms/step - accuracy: 0.5988 - loss: 1.2080 - val_accuracy: 0.8095 - val_loss: 0.5380
Epoch 2/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 31s 4ms/step - accuracy: 0.7715 - loss: 0.5946 - val_accuracy: 0.8300 - val_loss: 0.4844
Epoch 3/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 32s 4ms/step - accuracy: 0.8148 - loss: 0.5212 - val_accuracy: 0.8265 - val_loss: 0.4905
Epoch 4/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 32s 4ms/step - accuracy: 0.8322 - loss: 0.4765 - val_accuracy: 0.8355 - val_loss: 0.4536
Epoch 5/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 36s 4ms/step - accuracy: 0.8329 - loss: 0.4737 - val_accuracy: 0.8625 - val_loss: 0.4319
Epoch 6/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 40s 5ms/step - accuracy: 0.8524 - loss: 0.4229 - val_accuracy: 0.8675 - val_loss: 0.3883
Epoch 7/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 39s 5ms/step - accuracy: 0.8518 - loss: 0.4208 - val_accuracy: 0.8600 - val_loss: 0.4005
Epoch 8/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.8499 - loss: 0.4020 - val_accuracy: 0.8670 - val_loss: 0.4102
Epoch 9/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 37s 5ms/step - accuracy: 0.8657 - loss: 0.3803 - val_accuracy: 0.8620 - val_loss: 0.4182
Epoch 10/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 37s 5ms/step - accuracy: 0.8631 - loss: 0.3917 - val_accuracy: 0.8590 - val_loss: 0.4304

real    5m55.662s
user    17m6.411s
sys     5m31.401s
[1/3] Generating learning curve...

real    0m3.373s
user    0m2.487s
sys     0m0.284s
mv: rename plots/a.joblib.learning_curve.png to plots/a.joblib.learning_curve-1.png: No such file or directory
[2/3] Refitting model with Batch 2...
Epoch 1/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 39s 5ms/step - accuracy: 0.8273 - loss: 0.5328 - val_accuracy: 0.8515 - val_loss: 0.4034
Epoch 2/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 41s 5ms/step - accuracy: 0.8489 - loss: 0.4436 - val_accuracy: 0.8690 - val_loss: 0.3942
Epoch 3/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.8616 - loss: 0.4025 - val_accuracy: 0.8680 - val_loss: 0.3807
Epoch 4/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.8573 - loss: 0.4007 - val_accuracy: 0.8700 - val_loss: 0.3951
Epoch 5/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 40s 5ms/step - accuracy: 0.8595 - loss: 0.3868 - val_accuracy: 0.8675 - val_loss: 0.5108
Epoch 6/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 41s 5ms/step - accuracy: 0.8667 - loss: 0.3930 - val_accuracy: 0.8680 - val_loss: 0.4024
Epoch 7/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 41s 5ms/step - accuracy: 0.8760 - loss: 0.3864 - val_accuracy: 0.8765 - val_loss: 0.4013
Epoch 8/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 40s 5ms/step - accuracy: 0.8696 - loss: 0.3580 - val_accuracy: 0.8625 - val_loss: 0.4844

real    5m20.639s
user    14m49.912s
sys     5m6.014s
[2/3] Generating learning curve...

real    0m3.543s
user    0m2.557s
sys     0m0.305s
mv: rename plots/a.joblib.learning_curve.png to plots/a.joblib.learning_curve-2.png: No such file or directory
[3/3] Refitting model with Batch 3...
Epoch 1/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.8288 - loss: 0.5205 - val_accuracy: 0.8475 - val_loss: 0.4125
Epoch 2/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 39s 5ms/step - accuracy: 0.8477 - loss: 0.4454 - val_accuracy: 0.8620 - val_loss: 0.4203
Epoch 3/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 39s 5ms/step - accuracy: 0.8464 - loss: 0.4504 - val_accuracy: 0.8425 - val_loss: 0.4271
Epoch 4/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 37s 5ms/step - accuracy: 0.8467 - loss: 0.4510 - val_accuracy: 0.8475 - val_loss: 0.4545
Epoch 5/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.8546 - loss: 0.4005 - val_accuracy: 0.8570 - val_loss: 0.5016
Epoch 6/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.8587 - loss: 0.4239 - val_accuracy: 0.8690 - val_loss: 0.4280

real    3m52.331s
user    11m32.877s
sys     3m18.525s
[3/3] Generating learning curve...
Model name:  

real    0m3.455s
user    0m2.484s
sys     0m0.292s
mv: rename plots/a.joblib.learning_curve.png to plots/a.joblib.learning_curve-3.png: No such file or directory
Generating score
313/313 ━━━━━━━━━━━━━━━━━━━━ 5s 14ms/step 

models/a.joblib: train: 

+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
| 741 |   4 |  18 |  49 |   8 |   0 | 183 |   0 |   6 |   0 |
|   1 | 996 |   4 |  12 |   1 |   0 |   5 |   0 |   0 |   0 |
|   8 |   1 | 838 |  10 |  82 |   0 |  41 |   0 |   0 |   0 |
|  13 |  28 |  12 | 860 |  37 |   0 |  30 |   0 |   2 |   0 |
|   2 |   3 | 161 |  63 | 691 |   0 |  73 |   0 |   0 |   0 |
|   0 |   0 |   0 |   0 |   0 | 981 |   0 |  28 |   3 |   5 |
|  81 |   7 | 164 |  32 |  87 |   0 | 615 |   0 |  10 |   0 |
|   0 |   0 |   0 |   0 |   0 |  37 |   0 | 934 |   3 |  23 |
|   3 |   0 |   8 |   6 |  10 |   5 |   9 |   0 | 948 |   1 |
|   0 |   0 |   0 |   1 |   0 |  13 |   0 |  47 |   0 | 956 |
+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

              precision    recall  f1-score   support

           0       0.87      0.73      0.80      1009
           1       0.96      0.98      0.97      1019
           2       0.70      0.86      0.77       980
           3       0.83      0.88      0.85       982
           4       0.75      0.70      0.72       993
           5       0.95      0.96      0.96      1017
           6       0.64      0.62      0.63       996
           7       0.93      0.94      0.93       997
           8       0.98      0.96      0.97       990
           9       0.97      0.94      0.96      1017

    accuracy                           0.86     10000
   macro avg       0.86      0.86      0.85     10000
weighted avg       0.86      0.86      0.86     10000



real    0m7.473s
user    0m32.374s
sys     0m1.047s
```

It used in between 50 and 70% of my CPU, maybe I should figure out how to get
tensorflow to use the gpu.
The approximate time that this took is around 15 minutes, we will see when it starts getting bigger with the new dataset and models