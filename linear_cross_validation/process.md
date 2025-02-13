# LINNEAR REGRESSION WITH CROSS VALIDATION 

## BIG PICTURE

Following the flow chart from the powerpoint I first went into [kaggle](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/overview) 
and I read about the dataset, the description and the key features. They even offer some histograms with the data. 

## GET DATA

Once I felt like I understood the data enough I downloaded it and placed it in my `data` directory where I plan to have all the csv files so that 
everything is a little bit more organized. The data comes already in two parts, one for training and the other one for testing. There is also a txt file with more details about the data and the values.

## EXPLORE/VISUALIZE

I am now working on the scatter plots and histograms but I cannot seem to generate the histograms since we have some data that is not integers 
and its just string values. For example: MSZoning has these values: RL, RH, RM, C(all), and FV. For some reason when doing the histograms numpy doesn't like 
the categorical data:

`./display_data.py feature-histograms
Traceback (most recent call last):
  File "/Users/paulalozanogonzalo/spring2025/CS-4320/linear_cross_validation/./display_data.py", line 178, in <module>
    main(sys.argv)  
    ^^^^^^^^^^^^^^
  File "/Users/paulalozanogonzalo/spring2025/CS-4320/linear_cross_validation/./display_data.py", line 170, in main
    display_feature_histograms(my_args, data) 
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/paulalozanogonzalo/spring2025/CS-4320/linear_cross_validation/./display_data.py", line 74, in display_feature_histograms
    n, _ = np.histogram(col_df[feature_column], bins=20)  # Get histogram counts
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/paulalozanogonzalo/.pyenv/versions/3.11.0/lib/python3.11/site-packages/numpy/lib/_histograms_impl.py", line 796, in histogram
    bin_edges, uniform_bins = _get_bin_edges(a, bins, range, weights)
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/paulalozanogonzalo/.pyenv/versions/3.11.0/lib/python3.11/site-packages/numpy/lib/_histograms_impl.py", line 429, in _get_bin_edges
    first_edge, last_edge = _get_outer_edges(a, range)
                            ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/paulalozanogonzalo/.pyenv/versions/3.11.0/lib/python3.11/site-packages/numpy/lib/_histograms_impl.py", line 321, in _get_outer_edges
    if not (np.isfinite(first_edge) and np.isfinite(last_edge)):
            ^^^^^^^^^^^^^^^^^^^^^^^
TypeError: ufunc 'isfinite' not supported for the input types, and the inputs could not be safely coerced to any supported types according to the casting rule ''safe''`

For now I have decided to let it be and only use the scatter plots.

There are plenty of features (80) so using the `sklearn.preprocessing.PolynomialFeatures` will be very slow. 

## PREPARE DATA

This data set had the data ready to start with the pipeline since they give you a test.csv and train.csv.
One thing that is different here is that there is numerical data but also categorical data. Because of this we will have two pipelines to take care of each separately
and then we will fork them back together so that we can train the model only with one.

## PIPELINE

This is the part that took me the longest, getting everything setup and the categorical and numerical pipelines and then doing the different model options. Once
I got everything setup it was my time to guess and check and I logged all that in my commands.bash 

## TABLE TRIES

We need to record a table with the model we used, the cross validation R^2 we got, the train R^2 we got and if we submit it the score after submitting.