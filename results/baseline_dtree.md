# data split
    train/val/test = 6:2:2
    result on validation set
# hyperparameters
build_decision_tree(
    random_state=42,
    其余参数为 sklearn 默认值)


# evaluation
## confusion matrix
[[ 150   20   13    3    1   36   27   65   71]
 [  20 2246  620  166   14   30   72   29   27]
 [   7  619  746  123    1   17   62   14   12]
 [   3  159  106  226    2   18   15    3    6]
 [   1   13    6    5  515    0    3    0    5]
 [  44   32   13   28    6 2492   52  106   54]
 [  33   73   52   20    6   51  248   58   27]
 [  59   35   25    6    9   90   57 1335   77]
 [  96   34   13   11    6   59   24   56  692]]
## accuracy
0.6989334195216548

## classification report
              precision    recall  f1-score   support

     Class_1       0.36      0.39      0.38       386
     Class_2       0.70      0.70      0.70      3224
     Class_3       0.47      0.47      0.47      1601
     Class_4       0.38      0.42      0.40       538
     Class_5       0.92      0.94      0.93       548
     Class_6       0.89      0.88      0.89      2827
     Class_7       0.44      0.44      0.44       568
     Class_8       0.80      0.79      0.79      1693
     Class_9       0.71      0.70      0.71       991

    accuracy                           0.70     12376
   macro avg       0.63      0.64      0.63     12376
weighted avg       0.70      0.70      0.70     12376