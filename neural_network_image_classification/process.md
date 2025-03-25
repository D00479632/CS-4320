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

``` bash
./cnn.bash
=== Starting CNN training process ===
Model name: b
[1/4] Fitting initial model...
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 32, 32, 64)          │           9,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 16, 16, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 16, 16, 128)         │          73,856 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 16, 16, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 8, 8, 128)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 8192)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 64)                  │         524,352 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 64)                  │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │             650 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 755,914 (2.88 MB)
 Trainable params: 755,914 (2.88 MB)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 39s 5ms/step - accuracy: 0.1158 - loss: 2.3079 - val_accuracy: 0.3220 - val_loss: 1.8583
Epoch 2/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.2900 - loss: 1.9218 - val_accuracy: 0.4410 - val_loss: 1.6276
Epoch 3/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 38s 5ms/step - accuracy: 0.3853 - loss: 1.6878 - val_accuracy: 0.4585 - val_loss: 1.5125
Epoch 4/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 40s 5ms/step - accuracy: 0.4491 - loss: 1.5220 - val_accuracy: 0.5110 - val_loss: 1.3923
Epoch 5/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 39s 5ms/step - accuracy: 0.4821 - loss: 1.4116 - val_accuracy: 0.5440 - val_loss: 1.3302
Epoch 6/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 42s 5ms/step - accuracy: 0.5195 - loss: 1.3096 - val_accuracy: 0.5635 - val_loss: 1.2529
Epoch 7/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 44s 6ms/step - accuracy: 0.5610 - loss: 1.2157 - val_accuracy: 0.5760 - val_loss: 1.2333
Epoch 8/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 48s 6ms/step - accuracy: 0.6071 - loss: 1.0957 - val_accuracy: 0.5605 - val_loss: 1.2268
Epoch 9/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 46s 6ms/step - accuracy: 0.6290 - loss: 1.0280 - val_accuracy: 0.5845 - val_loss: 1.2061
Epoch 10/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 47s 6ms/step - accuracy: 0.6576 - loss: 0.9575 - val_accuracy: 0.5880 - val_loss: 1.2164

real    7m7.165s
user    21m42.977s
sys     6m53.427s
[1/4] Generating learning curve...
Model name:  

real    0m3.464s
user    0m2.477s
sys     0m0.293s
mv: rename plots/b.learning_curve.png to plots/b.learning_curve-1.png: No such file or directory
[2/4] Refitting model with Batch 2...
Epoch 1/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 46s 6ms/step - accuracy: 0.5075 - loss: 1.4217 - val_accuracy: 0.5620 - val_loss: 1.2470
Epoch 2/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 47s 6ms/step - accuracy: 0.5359 - loss: 1.2850 - val_accuracy: 0.5850 - val_loss: 1.1504
Epoch 3/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 45s 6ms/step - accuracy: 0.5930 - loss: 1.1391 - val_accuracy: 0.5965 - val_loss: 1.1190
Epoch 4/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 46s 6ms/step - accuracy: 0.6274 - loss: 1.0406 - val_accuracy: 0.6200 - val_loss: 1.0947
Epoch 5/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 46s 6ms/step - accuracy: 0.6565 - loss: 0.9406 - val_accuracy: 0.5885 - val_loss: 1.1601
Epoch 6/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 46s 6ms/step - accuracy: 0.6890 - loss: 0.8657 - val_accuracy: 0.6205 - val_loss: 1.1871
Epoch 7/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 47s 6ms/step - accuracy: 0.7114 - loss: 0.8010 - val_accuracy: 0.6185 - val_loss: 1.1264
Epoch 8/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 47s 6ms/step - accuracy: 0.7449 - loss: 0.7176 - val_accuracy: 0.6180 - val_loss: 1.1438
Epoch 9/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 47s 6ms/step - accuracy: 0.7586 - loss: 0.6793 - val_accuracy: 0.6080 - val_loss: 1.2505

real    7m0.626s
user    22m36.186s
sys     6m6.759s
[2/4] Generating learning curve...
Model name:  

real    0m3.420s
user    0m2.460s
sys     0m0.292s
mv: rename plots/b.learning_curve.png to plots/b.learning_curve-2.png: No such file or directory
[3/4] Refitting model with Batch 3...
Epoch 1/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 49s 6ms/step - accuracy: 0.5478 - loss: 1.3100 - val_accuracy: 0.6150 - val_loss: 1.1324
Epoch 2/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 51s 6ms/step - accuracy: 0.5822 - loss: 1.1643 - val_accuracy: 0.6280 - val_loss: 1.1145
Epoch 3/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.6290 - loss: 1.0423 - val_accuracy: 0.6255 - val_loss: 1.0850
Epoch 4/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.6581 - loss: 0.9731 - val_accuracy: 0.6250 - val_loss: 1.0704
Epoch 5/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 51s 6ms/step - accuracy: 0.6866 - loss: 0.8995 - val_accuracy: 0.6515 - val_loss: 1.0855
Epoch 6/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 52s 6ms/step - accuracy: 0.7140 - loss: 0.7870 - val_accuracy: 0.6425 - val_loss: 1.0914
Epoch 7/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 51s 6ms/step - accuracy: 0.7302 - loss: 0.7503 - val_accuracy: 0.6510 - val_loss: 1.1427
Epoch 8/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 49s 6ms/step - accuracy: 0.7575 - loss: 0.6592 - val_accuracy: 0.6585 - val_loss: 1.1242
Epoch 9/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 48s 6ms/step - accuracy: 0.7878 - loss: 0.5936 - val_accuracy: 0.6535 - val_loss: 1.1931

real    7m32.903s
user    23m23.645s
sys     6m59.363s
[3/4] Generating learning curve...
Model name:  

real    0m3.414s
user    0m2.458s
sys     0m0.289s
mv: rename plots/b.learning_curve.png to plots/b..learning_curve-3.png: No such file or directory
[4/4] Refitting model with Batch 4...
Epoch 1/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.5742 - loss: 1.2600 - val_accuracy: 0.6315 - val_loss: 1.0960
Epoch 2/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 51s 6ms/step - accuracy: 0.6229 - loss: 1.0977 - val_accuracy: 0.6250 - val_loss: 1.0740
Epoch 3/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.6528 - loss: 0.9920 - val_accuracy: 0.6535 - val_loss: 1.0633
Epoch 4/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.6791 - loss: 0.9017 - val_accuracy: 0.6645 - val_loss: 1.0450
Epoch 5/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.7175 - loss: 0.7878 - val_accuracy: 0.6610 - val_loss: 1.0056
Epoch 6/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.7442 - loss: 0.7175 - val_accuracy: 0.6650 - val_loss: 0.9959
Epoch 7/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 50s 6ms/step - accuracy: 0.7690 - loss: 0.6520 - val_accuracy: 0.6605 - val_loss: 1.0820
Epoch 8/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 49s 6ms/step - accuracy: 0.7864 - loss: 0.6125 - val_accuracy: 0.6575 - val_loss: 1.0364
Epoch 9/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 51s 6ms/step - accuracy: 0.7940 - loss: 0.5656 - val_accuracy: 0.6640 - val_loss: 1.1867
Epoch 10/10
8000/8000 ━━━━━━━━━━━━━━━━━━━━ 51s 6ms/step - accuracy: 0.8197 - loss: 0.5063 - val_accuracy: 0.6495 - val_loss: 1.0700

real    8m25.302s
user    26m21.755s
sys     7m39.301s
[4/4] Generating learning curve...
Model name:  

real    0m3.444s
user    0m2.481s
sys     0m0.291s
mv: rename plots/b.blearning_curve.png to plots/b.learning_curve-4.png: No such file or directory
Generating score
313/313 ━━━━━━━━━━━━━━━━━━━━ 7s 23ms/step 

models/b.joblib: train: 

+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
| 769 |  20 |  28 |  22 |  13 |   9 |   5 |  12 |  81 |  51 |
|  36 | 787 |   1 |   8 |   5 |   1 |   9 |   2 |  32 | 108 |
| 124 |   9 | 436 |  84 | 139 |  75 |  63 |  34 |  16 |  13 |
|  24 |   9 |  48 | 492 |  83 | 174 |  58 |  37 |  13 |  29 |
|  45 |   2 |  68 |  83 | 659 |  26 |  30 |  74 |  14 |  10 |
|   9 |   3 |  52 | 235 |  62 | 546 |  23 |  58 |   7 |   9 |
|   7 |  11 |  42 | 109 |  84 |  24 | 711 |  13 |  13 |  19 |
|  32 |   1 |  17 |  56 | 106 |  78 |   2 | 726 |   2 |  26 |
| 107 |  26 |   4 |  21 |   4 |   4 |   5 |   8 | 801 |  24 |
|  38 | 125 |   1 |  21 |   4 |   9 |   8 |  12 |  27 | 698 |
+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

              precision    recall  f1-score   support

           0       0.65      0.76      0.70      1010
           1       0.79      0.80      0.79       989
           2       0.63      0.44      0.52       993
           3       0.44      0.51      0.47       967
           4       0.57      0.65      0.61      1011
           5       0.58      0.54      0.56      1004
           6       0.78      0.69      0.73      1033
           7       0.74      0.69      0.72      1046
           8       0.80      0.80      0.80      1004
           9       0.71      0.74      0.72       943

    accuracy                           0.66     10000
   macro avg       0.67      0.66      0.66     10000
weighted avg       0.67      0.66      0.66     10000



real    0m10.807s
user    0m49.015s
sys     0m1.647s
```

This took about 30 minutes to finish and it ended up with an accuracy of 0.6495 in the last training batch and when tested with the 
validation batch we got an accuracy of 0.66. This is all right. We dont meet the requirements of: 
- Validation data accuracy ('val_categorical_accuracy'): > 0.75
- Testing data accuracy ('test_categorical_accuracy'): > 0.75
- abs(val_categorical_accuracy - test_categorical_accuracy): < 0.03
since we have val_accurac: 0.6495 ( not > 0.75)
and a testing (not with the test data just validation but I will use the validation for now) score of 0.66
We do have an abs(0.6495-0.66) of 0.01 which is < 0.03 so that's good.

I just figured out that I had the batch size = 1 when I was fitting the model, this slows down the process a lot so I will try to do 32 and maybe that will be faster. 
Another way I could make it faster is by training on multiple data batches at the same time instead of refitting.

```bash
./cnn.bash
=== Starting CNN training process ===
Model name: c
[1/4] Fitting initial model...
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 32, 32, 64)          │           9,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 16, 16, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 16, 16, 128)         │         204,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 16, 16, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 8, 8, 128)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_3 (Conv2D)                    │ (None, 8, 8, 256)           │         295,168 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_2 (MaxPooling2D)       │ (None, 4, 4, 256)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 4096)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │         524,416 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,182,858 (4.51 MB)
 Trainable params: 1,182,858 (4.51 MB)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 22s 87ms/step - accuracy: 0.1571 - loss: 2.2843 - val_accuracy: 0.3475 - val_loss: 1.9368
Epoch 2/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 23s 91ms/step - accuracy: 0.2872 - loss: 1.9577 - val_accuracy: 0.4035 - val_loss: 1.7591
Epoch 3/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 23s 93ms/step - accuracy: 0.3623 - loss: 1.8049 - val_accuracy: 0.4625 - val_loss: 1.6058
Epoch 4/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 23s 93ms/step - accuracy: 0.4010 - loss: 1.6864 - val_accuracy: 0.4635 - val_loss: 1.5213
Epoch 5/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 24s 96ms/step - accuracy: 0.4282 - loss: 1.5987 - val_accuracy: 0.4750 - val_loss: 1.4741
Epoch 6/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 24s 96ms/step - accuracy: 0.4614 - loss: 1.5163 - val_accuracy: 0.5090 - val_loss: 1.4434
Epoch 7/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 24s 97ms/step - accuracy: 0.4808 - loss: 1.4411 - val_accuracy: 0.5245 - val_loss: 1.3669
Epoch 8/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 27s 106ms/step - accuracy: 0.5107 - loss: 1.3752 - val_accuracy: 0.5335 - val_loss: 1.3442
Epoch 9/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 124ms/step - accuracy: 0.5317 - loss: 1.3192 - val_accuracy: 0.5405 - val_loss: 1.3184
Epoch 10/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 34s 135ms/step - accuracy: 0.5594 - loss: 1.2451 - val_accuracy: 0.5455 - val_loss: 1.2778

real    4m21.268s
user    26m20.848s
sys     1m38.423s
[1/4] Generating learning curve...

real    0m3.526s
user    0m2.481s
sys     0m0.297s
[2/4] Refitting model with Batch 2...
Epoch 1/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 118ms/step - accuracy: 0.4862 - loss: 1.4398 - val_accuracy: 0.5535 - val_loss: 1.2875
Epoch 2/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 120ms/step - accuracy: 0.5268 - loss: 1.3526 - val_accuracy: 0.5540 - val_loss: 1.2546
Epoch 3/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 119ms/step - accuracy: 0.5418 - loss: 1.3010 - val_accuracy: 0.5505 - val_loss: 1.2460
Epoch 4/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 122ms/step - accuracy: 0.5692 - loss: 1.2242 - val_accuracy: 0.5615 - val_loss: 1.1950
Epoch 5/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 120ms/step - accuracy: 0.5773 - loss: 1.1793 - val_accuracy: 0.5770 - val_loss: 1.1707
Epoch 6/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 120ms/step - accuracy: 0.5962 - loss: 1.1337 - val_accuracy: 0.5840 - val_loss: 1.1613
Epoch 7/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 29s 118ms/step - accuracy: 0.6110 - loss: 1.0824 - val_accuracy: 0.5755 - val_loss: 1.1464
Epoch 8/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 29s 117ms/step - accuracy: 0.6420 - loss: 1.0092 - val_accuracy: 0.5835 - val_loss: 1.1249
Epoch 9/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 29s 118ms/step - accuracy: 0.6478 - loss: 0.9786 - val_accuracy: 0.5965 - val_loss: 1.1348
Epoch 10/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 122ms/step - accuracy: 0.6746 - loss: 0.9028 - val_accuracy: 0.5865 - val_loss: 1.1488

real    5m2.073s
user    32m0.440s
sys     1m38.817s
[2/4] Generating learning curve...

real    0m3.449s
user    0m2.470s
sys     0m0.287s
[3/4] Refitting model with Batch 3...
Epoch 1/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 28s 112ms/step - accuracy: 0.5546 - loss: 1.2851 - val_accuracy: 0.6265 - val_loss: 1.1072
Epoch 2/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 118ms/step - accuracy: 0.5841 - loss: 1.1839 - val_accuracy: 0.6100 - val_loss: 1.1168
Epoch 3/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 121ms/step - accuracy: 0.6010 - loss: 1.1347 - val_accuracy: 0.6305 - val_loss: 1.0860
Epoch 4/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 122ms/step - accuracy: 0.6134 - loss: 1.0941 - val_accuracy: 0.6280 - val_loss: 1.1083
Epoch 5/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 122ms/step - accuracy: 0.6389 - loss: 1.0325 - val_accuracy: 0.6235 - val_loss: 1.0946
Epoch 6/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 125ms/step - accuracy: 0.6647 - loss: 0.9478 - val_accuracy: 0.6235 - val_loss: 1.0601
Epoch 7/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 123ms/step - accuracy: 0.6852 - loss: 0.8901 - val_accuracy: 0.6405 - val_loss: 1.0354
Epoch 8/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 125ms/step - accuracy: 0.6993 - loss: 0.8600 - val_accuracy: 0.6385 - val_loss: 1.0608
Epoch 9/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 125ms/step - accuracy: 0.7118 - loss: 0.8029 - val_accuracy: 0.6425 - val_loss: 1.0406
Epoch 10/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 33s 133ms/step - accuracy: 0.7212 - loss: 0.7773 - val_accuracy: 0.6460 - val_loss: 1.0315

real    5m10.799s
user    32m1.247s
sys     1m44.556s
[3/4] Generating learning curve...

real    0m3.536s
user    0m2.534s
sys     0m0.306s
[4/4] Refitting model with Batch 4...
Epoch 1/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 124ms/step - accuracy: 0.6011 - loss: 1.1624 - val_accuracy: 0.6315 - val_loss: 1.0803
Epoch 2/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 33s 133ms/step - accuracy: 0.6346 - loss: 1.0717 - val_accuracy: 0.6445 - val_loss: 1.0223
Epoch 3/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 32s 127ms/step - accuracy: 0.6513 - loss: 0.9795 - val_accuracy: 0.6590 - val_loss: 1.0157
Epoch 4/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 123ms/step - accuracy: 0.6811 - loss: 0.9214 - val_accuracy: 0.6380 - val_loss: 1.0281
Epoch 5/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 123ms/step - accuracy: 0.7072 - loss: 0.8426 - val_accuracy: 0.6560 - val_loss: 1.0012
Epoch 6/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 119ms/step - accuracy: 0.7219 - loss: 0.7956 - val_accuracy: 0.6660 - val_loss: 0.9887
Epoch 7/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 121ms/step - accuracy: 0.7456 - loss: 0.7389 - val_accuracy: 0.6625 - val_loss: 1.0248
Epoch 8/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 30s 120ms/step - accuracy: 0.7569 - loss: 0.6850 - val_accuracy: 0.6510 - val_loss: 1.0160
Epoch 9/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 31s 126ms/step - accuracy: 0.7795 - loss: 0.6231 - val_accuracy: 0.6570 - val_loss: 1.0192
Epoch 10/10
250/250 ━━━━━━━━━━━━━━━━━━━━ 32s 128ms/step - accuracy: 0.8036 - loss: 0.5752 - val_accuracy: 0.6620 - val_loss: 1.0518

real    5m14.861s
user    32m26.284s
sys     1m38.678s
[4/4] Generating learning curve...

real    0m3.459s
user    0m2.463s
sys     0m0.293s
mv: rename plots/c.blearning_curve.png to plots/c.learning_curve-4.png: No such file or directory
Generating score
313/313 ━━━━━━━━━━━━━━━━━━━━ 10s 31ms/step

models/c.joblib: train: 

+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
| 754 |  30 |  53 |   6 |  22 |   8 |  18 |  20 |  69 |  30 |
|  41 | 818 |   6 |   6 |   4 |   2 |  15 |   4 |  17 |  76 |
|  87 |   9 | 563 |  54 |  95 |  42 |  72 |  37 |  19 |  15 |
|  31 |  11 |  90 | 436 |  76 | 145 |  94 |  41 |  11 |  32 |
|  36 |   7 | 117 |  52 | 596 |  26 |  57 | 108 |   8 |   4 |
|   7 |   6 | 101 | 220 |  59 | 476 |  45 |  76 |   7 |   7 |
|  10 |  14 |  53 |  52 |  66 |  11 | 793 |   9 |   8 |  17 |
|  18 |   4 |  46 |  39 |  91 |  51 |   5 | 778 |   1 |  13 |
| 105 |  66 |  15 |  18 |  13 |   2 |  11 |   3 | 746 |  25 |
|  46 | 155 |  11 |   9 |   4 |   9 |  15 |  30 |  28 | 636 |
+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

              precision    recall  f1-score   support

           0       0.66      0.75      0.70      1010
           1       0.73      0.83      0.78       989
           2       0.53      0.57      0.55       993
           3       0.49      0.45      0.47       967
           4       0.58      0.59      0.59      1011
           5       0.62      0.47      0.54      1004
           6       0.70      0.77      0.73      1033
           7       0.70      0.74      0.72      1046
           8       0.82      0.74      0.78      1004
           9       0.74      0.67      0.71       943

    accuracy                           0.66     10000
   macro avg       0.66      0.66      0.66     10000
weighted avg       0.66      0.66      0.66     10000



real    0m13.546s
user    1m7.127s
sys     0m2.051s
```

We got the same accuracy as in the past model. However, the graphs are very different. I noticed that this one got bad at 
generalizing after the first fit and I can also see in c.learning_curve-1 that it could've gone for more epochs so for the next model 
I will change the epochs to be 20 for the cnn_fit and see how it does.

For this next try I changed the architecture of my model, and also the batch size (from 32 to 16). I also installed 
the necessary packages for tensorflow to use my gpu on mac.
Here are the websites:
https://developer.apple.com/metal/tensorflow-plugin/
https://blog.fotiecodes.com/install-tensorflow-on-your-mac-m1m2m3-with-gpu-support-clqs92bzl000308l8a3i35479
Lets see if it works and if it runs faster.

```bash
./cnn.bash
=== Starting CNN training process ===
Model name: d
[1/4] Fitting initial model...
GPU is available
2025-03-24 18:43:35.204206: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2025-03-24 18:43:35.204243: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2025-03-24 18:43:35.204257: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2025-03-24 18:43:35.204292: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-03-24 18:43:35.204304: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ Layer (type)                         ┃ Output Shape                ┃         Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                      │ (None, 32, 32, 64)          │           9,472 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d (MaxPooling2D)         │ (None, 16, 16, 64)          │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_1 (Conv2D)                    │ (None, 16, 16, 128)         │         204,928 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ conv2d_2 (Conv2D)                    │ (None, 16, 16, 128)         │         147,584 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ max_pooling2d_1 (MaxPooling2D)       │ (None, 8, 8, 128)           │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ flatten (Flatten)                    │ (None, 8192)                │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 128)                 │       1,048,704 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_1 (Dense)                      │ (None, 128)                 │          16,512 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dropout (Dropout)                    │ (None, 128)                 │               0 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense_2 (Dense)                      │ (None, 10)                  │           1,290 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 1,428,490 (5.45 MB)
 Trainable params: 1,428,490 (5.45 MB)
 Non-trainable params: 0 (0.00 B)
None
Epoch 1/10
2025-03-24 18:43:36.613893: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
500/500 ━━━━━━━━━━━━━━━━━━━━ 18s 28ms/step - accuracy: 0.1134 - loss: 5.3609 - val_accuracy: 0.2420 - val_loss: 2.1239
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - accuracy: 0.1714 - loss: 5.2471 - val_accuracy: 0.3245 - val_loss: 2.6237
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2203 - loss: 7.1075 - val_accuracy: 0.3555 - val_loss: 2.9990
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 29ms/step - accuracy: 0.2336 - loss: 8.6853 - val_accuracy: 0.3835 - val_loss: 3.9924
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2640 - loss: 11.7143 - val_accuracy: 0.4125 - val_loss: 5.2111
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2774 - loss: 12.9118 - val_accuracy: 0.4470 - val_loss: 4.6469

real    1m35.390s
user    0m58.482s
sys     0m26.142s
[1/4] Generating learning curve...
GPU is available

real    0m3.640s
user    0m4.561s
sys     0m1.576s
[2/4] Refitting model with Batch 2...
GPU is available
2025-03-24 18:45:12.464174: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2025-03-24 18:45:12.464201: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2025-03-24 18:45:12.464209: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2025-03-24 18:45:12.464227: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-03-24 18:45:12.464241: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Epoch 1/10
2025-03-24 18:45:12.915832: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
500/500 ━━━━━━━━━━━━━━━━━━━━ 15s 28ms/step - accuracy: 0.1600 - loss: 5.7435 - val_accuracy: 0.3185 - val_loss: 2.5008
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 27ms/step - accuracy: 0.1893 - loss: 10.6767 - val_accuracy: 0.3065 - val_loss: 3.7573
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2058 - loss: 13.7324 - val_accuracy: 0.3525 - val_loss: 4.9334
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2293 - loss: 15.7300 - val_accuracy: 0.3685 - val_loss: 5.9665
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2400 - loss: 21.2437 - val_accuracy: 0.3695 - val_loss: 8.5377
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 27ms/step - accuracy: 0.2675 - loss: 21.3590 - val_accuracy: 0.4095 - val_loss: 8.3419

real    1m27.529s
user    0m56.775s
sys     0m27.115s
[2/4] Generating learning curve...
GPU is available

real    0m3.568s
user    0m4.622s
sys     0m1.428s
[3/4] Refitting model with Batch 3...
GPU is available
2025-03-24 18:46:43.528082: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2025-03-24 18:46:43.528110: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2025-03-24 18:46:43.528118: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2025-03-24 18:46:43.528135: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-03-24 18:46:43.528149: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Epoch 1/10
2025-03-24 18:46:43.988951: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
500/500 ━━━━━━━━━━━━━━━━━━━━ 16s 30ms/step - accuracy: 0.1917 - loss: 11.0506 - val_accuracy: 0.3045 - val_loss: 3.8676
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 15s 29ms/step - accuracy: 0.2005 - loss: 13.2955 - val_accuracy: 0.3760 - val_loss: 4.7399
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 15s 29ms/step - accuracy: 0.2250 - loss: 17.6846 - val_accuracy: 0.3715 - val_loss: 5.7227
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2419 - loss: 25.9772 - val_accuracy: 0.3705 - val_loss: 10.9437
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2560 - loss: 26.3172 - val_accuracy: 0.3875 - val_loss: 8.7684
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 15s 29ms/step - accuracy: 0.2620 - loss: 28.2883 - val_accuracy: 0.4005 - val_loss: 11.4892

real    1m32.145s
user    0m58.049s
sys     0m27.338s
[3/4] Generating learning curve...
GPU is available

real    0m3.454s
user    0m4.385s
sys     0m1.748s
[4/4] Refitting model with Batch 4...
GPU is available
2025-03-24 18:48:19.132002: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2025-03-24 18:48:19.132025: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2025-03-24 18:48:19.132031: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2025-03-24 18:48:19.132046: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-03-24 18:48:19.132058: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
Epoch 1/10
2025-03-24 18:48:19.587213: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
500/500 ━━━━━━━━━━━━━━━━━━━━ 15s 29ms/step - accuracy: 0.2020 - loss: 14.8183 - val_accuracy: 0.3425 - val_loss: 5.7657
Epoch 2/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2145 - loss: 22.0295 - val_accuracy: 0.3840 - val_loss: 8.4517
Epoch 3/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 27ms/step - accuracy: 0.2422 - loss: 24.5268 - val_accuracy: 0.4050 - val_loss: 9.1871
Epoch 4/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 28ms/step - accuracy: 0.2406 - loss: 30.6841 - val_accuracy: 0.4030 - val_loss: 16.1064
Epoch 5/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 14s 27ms/step - accuracy: 0.2539 - loss: 41.0464 - val_accuracy: 0.4185 - val_loss: 14.7644
Epoch 6/10
500/500 ━━━━━━━━━━━━━━━━━━━━ 13s 27ms/step - accuracy: 0.2587 - loss: 45.4614 - val_accuracy: 0.4170 - val_loss: 19.5652

real    1m27.841s
user    0m59.325s
sys     0m26.441s
[4/4] Generating learning curve...
GPU is available

real    0m3.476s
user    0m4.068s
sys     0m2.145s
mv: rename plots/d.blearning_curve.png to plots/d.learning_curve-4.png: No such file or directory
Generating score
GPU is available
2025-03-24 18:49:50.399278: I metal_plugin/src/device/metal_device.cc:1154] Metal device set to: Apple M2
2025-03-24 18:49:50.399304: I metal_plugin/src/device/metal_device.cc:296] systemMemory: 16.00 GB
2025-03-24 18:49:50.399312: I metal_plugin/src/device/metal_device.cc:313] maxCacheSize: 5.33 GB
2025-03-24 18:49:50.399335: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:305] Could not identify NUMA node of platform GPU ID 0, defaulting to 0. Your kernel may not have been built with NUMA support.
2025-03-24 18:49:50.399349: I tensorflow/core/common_runtime/pluggable_device/pluggable_device_factory.cc:271] Created TensorFlow device (/job:localhost/replica:0/task:0/device:GPU:0 with 0 MB memory) -> physical PluggableDevice (device: 0, name: METAL, pci bus id: <undefined>)
2025-03-24 18:49:50.759252: I tensorflow/core/grappler/optimizers/custom_graph_optimizer_registry.cc:117] Plugin optimizer for device_type GPU is enabled.
313/313 ━━━━━━━━━━━━━━━━━━━━ 2s 6ms/step    

models/d.joblib: train: 

+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+
| 406 |  26 |  43 |   5 |  14 |  18 |  71 | 130 | 131 | 166 |
|  22 | 166 |   2 |   5 |   2 |   6 | 112 |  65 |  26 | 583 |
|  93 |   4 |  77 |   7 |  53 |  68 | 399 | 198 |  29 |  65 |
|  16 |   4 |  28 |  23 |   5 | 155 | 390 | 245 |   8 |  93 |
|  36 |   3 |  40 |   5 |  53 |  30 | 515 | 266 |  14 |  49 |
|   9 |   1 |  23 |  20 |   5 | 259 | 377 | 254 |   3 |  53 |
|   6 |   7 |  19 |   1 |  10 |  17 | 772 | 129 |   8 |  64 |
|  21 |  11 |   8 |  10 |   9 |  43 | 220 | 621 |   6 |  97 |
| 189 |  33 |  26 |  14 |   3 |  35 |  57 |  39 | 321 | 287 |
|  21 |  21 |   2 |   6 |   2 |   9 |  69 |  73 |  11 | 729 |
+-----+-----+-----+-----+-----+-----+-----+-----+-----+-----+

              precision    recall  f1-score   support

           0       0.50      0.40      0.44      1010
           1       0.60      0.17      0.26       989
           2       0.29      0.08      0.12       993
           3       0.24      0.02      0.04       967
           4       0.34      0.05      0.09      1011
           5       0.40      0.26      0.32      1004
           6       0.26      0.75      0.38      1033
           7       0.31      0.59      0.41      1046
           8       0.58      0.32      0.41      1004
           9       0.33      0.77      0.47       943

    accuracy                           0.34     10000
   macro avg       0.38      0.34      0.29     10000
weighted avg       0.38      0.34      0.30     10000



real    0m5.957s
user    0m4.174s
sys     0m1.926s
```

It for sure ran faster and was easier for my computer (temperature wise). However, the way I changed my model made it really bad.
My accuracy now is 0.33 and the model thought it was going to be 0.4170 which is way off. I will go back to my other model and make some other changes to see if that
makes it better 