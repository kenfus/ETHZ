# Task 4

## Task
We had to create a multiclass-classification to predict 5 classes (0-4) based on 120 features. We recieved a training and test file in hdr-format and a sample submission file in CSV-format. The training file contained 45'325 rows of data and 121 columns, y and x_1 to x_120.

## Hyperas
After trying different parameters and doing feature engineering, I arrived at an accuracy on the validation-set of 0.932.. (Right below the hard baseline of 0.93401). Optimizing with Hyperas returned the following improved parameters:

```
Best performing model chosen hyper-parameters:
{'Dense': 3, 'Dense_1': 5, 'Dense_2': 4, 'Dense_3': 3, 'Dense_4': 4, 'Dense_5': 4, 'Dense_6': 4, 'Dropout': 0.7341756871860086, 'Dropout_1': 0.012436751978174343, 'Dropout_2': 0.4424884329057396, 'Dropout_3': 0.8782007274402505, 'Dropout_4': 0.22323041287770126, 'Dropout_5': 0.7126274394639049, 'Dropout_6': 0.8985695947818465, 'activation_function': 0, 'lr': 1, 'nr_hlayers': 2, 'optimizer': 0}
```

and an accuracy on the validation-set of 0.9346227385766682. 

## Uploading sample solution
The final evaluation was done on a hidden evaluation set. My final score was an accuracy of 0.935460478608.

## Euler
The python-script was run for 13 hours and 42 minutes on EULER (Swiss Supercluster) with 32 cores and 32 GB Memory.
