# Task 4

## Task
We had to create a multiclass-classification to predict 5 classes (0-4) based on 120 features. We recieved a training and submission file in the hdr-format and a sample submission file in CSV-format. The training file contained 45325 rows of data and 121 columns, y and x_1 to x_120.

## Hyperas
After trying around and getting an accuracy on the test set of 0.932.. (Right below the hard baseline of 0.934010152284). With hyperas I got the following Parameters:

```
Best performing model chosen hyper-parameters:
{'Dense': 3, 'Dense_1': 5, 'Dense_2': 4, 'Dense_3': 3, 'Dense_4': 4, 'Dense_5': 4, 'Dense_6': 4, 'Dropout': 0.7341756871860086, 'Dropout_1': 0.012436751978174343, 'Dropout_2': 0.4424884329057396, 'Dropout_3': 0.8782007274402505, 'Dropout_4': 0.22323041287770126, 'Dropout_5': 0.7126274394639049, 'Dropout_6': 0.8985695947818465, 'activation_function': 0, 'lr': 1, 'nr_hlayers': 2, 'optimizer': 0}
```

and an accuracy of 0.9346227385766682. 

## Uploading sample solution
The final evaluation was done hidden on a uknown evaluation set. My solution there gave me a final score of 0.935460478608.

## Euler
The file was run for 13 hours 42 minutes on the ETHZ supercluster Euler on 32 cores.
