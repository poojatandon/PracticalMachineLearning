---
title: "Activity Recognition Using Predictive Analytics"
author: "Pooja Tandon"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---
## Introduction
Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, our goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har

## Data Preprocessing 

The training data for this project are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:

https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

The data for this project come from this source: http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har.

Load Libraries required:

```r
library(knitr)
```

```
## Warning: package 'knitr' was built under R version 3.6.2
```

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.6.2
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.6.2
```

```r
library(rpart)
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.6.2
```

```r
library(rattle)
```

```
## Warning: package 'rattle' was built under R version 3.6.2
```

```
## Rattle: A free graphical interface for data science with R.
## Version 5.3.0 Copyright (c) 2006-2018 Togaware Pty Ltd.
## Type 'rattle()' to shake, rattle, and roll your data.
```

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.6.2
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:rattle':
## 
##     importance
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
library(RColorBrewer)
library(RGtk2)
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.6.2
```

```
## Loaded gbm 2.1.5
```

## Loading Data

```r
train_url<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
test_url<- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
training_data<- read.csv(url(train_url))
testing_data<- read.csv(url(test_url))
dim(training_data)
```

```
## [1] 19622   160
```

```r
dim(testing_data)
```

```
## [1]  20 160
```

##Data Cleansing
Removing Variables which are having Nearly Zero Variance.

```r
nzv <- nearZeroVar(training_data)

train_data <- training_data[,-nzv]
test_data <- testing_data[,-nzv]

dim(train_data)
```

```
## [1] 19622   100
```

```r
dim(test_data)
```

```
## [1]  20 100
```

```r
na_val_col <- sapply(train_data, function(x) mean(is.na(x))) > 0.95
train_data <- train_data[,na_val_col == FALSE]
test_data <- test_data[,na_val_col == FALSE]

dim(train_data)
```

```
## [1] 19622    59
```

```r
dim(test_data) 
```

```
## [1] 20 59
```

```r
train_data<- train_data[, 8:59]
test_data<- test_data[, 8:59]
dim(train_data)
```

```
## [1] 19622    52
```

```r
dim(test_data)
```

```
## [1] 20 52
```

## Data Partioning
In this we will seggregate our train_data in two parts “training”(60% of data) and “testing”(40% of data)/ Validateion set.


```r
inTrain<- createDataPartition(train_data$classe, p=0.6, list=FALSE)
inTrain<- createDataPartition(train_data$classe, p=0.6, list=FALSE)
training<- train_data[inTrain,]
testing<- train_data[-inTrain,]
dim(training)
```

```
## [1] 11776    52
```

## Construct the Model using Cross Validation-
Decision Tree Model and Prediction

```r
library(rattle)
DT_model<- train(classe ~. , data=training, method= "rpart")
fancyRpartPlot(DT_model$finalModel)
```

![](project_files/figure-html/unnamed-chunk-5-1.png)<!-- -->


```r
set.seed(21243)
DT_prediction<- predict(DT_model, testing)
confusionMatrix(DT_prediction, testing$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2017  647  613  579  351
##          B   35  511   48  221  281
##          C  140  306  563  182  325
##          D   38   54  144  304   57
##          E    2    0    0    0  428
## 
## Overall Statistics
##                                           
##                Accuracy : 0.4873          
##                  95% CI : (0.4761, 0.4984)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.329           
##                                           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9037  0.33663  0.41155  0.23639  0.29681
## Specificity            0.6099  0.90755  0.85289  0.95534  0.99969
## Pos Pred Value         0.4794  0.46624  0.37137  0.50921  0.99535
## Neg Pred Value         0.9409  0.85081  0.87283  0.86453  0.86327
## Prevalence             0.2845  0.19347  0.17436  0.16391  0.18379
## Detection Rate         0.2571  0.06513  0.07176  0.03875  0.05455
## Detection Prevalence   0.5362  0.13969  0.19322  0.07609  0.05480
## Balanced Accuracy      0.7568  0.62209  0.63222  0.59586  0.64825
```


From the Decision Tree Model we see the prediction accuracy is 57% which is not upto satisfactory level.


## Random Forest Model and Prediction


```r
set.seed(26817)
###Fit the model   
RF_model<- train(classe ~. , data=training, method= "rf", ntree=100)
###Prediction  
RF_prediction<- predict(RF_model, testing)
RF_cm<-confusionMatrix(RF_prediction, testing$classe)
RF_cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2224   21    0    0    0
##          B    6 1490   10    0    0
##          C    1    6 1353   19    2
##          D    0    1    5 1267    2
##          E    1    0    0    0 1438
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9906          
##                  95% CI : (0.9882, 0.9926)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9881          
##                                           
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9964   0.9816   0.9890   0.9852   0.9972
## Specificity            0.9963   0.9975   0.9957   0.9988   0.9998
## Pos Pred Value         0.9906   0.9894   0.9797   0.9937   0.9993
## Neg Pred Value         0.9986   0.9956   0.9977   0.9971   0.9994
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2835   0.1899   0.1724   0.1615   0.1833
## Detection Prevalence   0.2861   0.1919   0.1760   0.1625   0.1834
## Balanced Accuracy      0.9963   0.9895   0.9924   0.9920   0.9985
```

```r
###plot    
plot(RF_cm$table, col=RF_cm$byClass, main="Random Forest Accuracy")
```

![](project_files/figure-html/unnamed-chunk-7-1.png)<!-- -->

From the Random Forest Model we see the prediction accuracy is 99% which is close to perfect accuracy level.

## Gradient Boosting Model and Prediction


```r
set.seed(25621)
gbm_model<- train(classe~., data=training, method="gbm", verbose= FALSE)
gbm_model$finalmodel
```

```
## NULL
```

```r
###Prediction    

gbm_prediction<- predict(gbm_model, testing)
gbm_cm<-confusionMatrix(gbm_prediction, testing$classe)
gbm_cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2197   60    0    1    3
##          B   25 1412   45    5   16
##          C    8   43 1313   42   16
##          D    0    1    8 1228   17
##          E    2    2    2   10 1390
## 
## Overall Statistics
##                                           
##                Accuracy : 0.961           
##                  95% CI : (0.9565, 0.9652)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9506          
##                                           
##  Mcnemar's Test P-Value : 1.146e-11       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9843   0.9302   0.9598   0.9549   0.9639
## Specificity            0.9886   0.9856   0.9832   0.9960   0.9975
## Pos Pred Value         0.9717   0.9395   0.9233   0.9793   0.9886
## Neg Pred Value         0.9937   0.9833   0.9914   0.9912   0.9919
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2800   0.1800   0.1673   0.1565   0.1772
## Detection Prevalence   0.2882   0.1916   0.1812   0.1598   0.1792
## Balanced Accuracy      0.9865   0.9579   0.9715   0.9755   0.9807
```

From the Gradient Boosting Model we see the prediction accuracy is 96% which is satisfied.


```r
##we have taken Random Forest and Gradient Boosting Model because it reach to satisfied prediction level. we are compairing the both model which is more accurate.    
RF_cm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##      0.9905684      0.9880679      0.9881738      0.9925871      0.2844762 
## AccuracyPValue  McnemarPValue 
##      0.0000000            NaN
```

```r
gbm_cm$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##   9.609992e-01   9.506462e-01   9.564771e-01   9.651747e-01   2.844762e-01 
## AccuracyPValue  McnemarPValue 
##   0.000000e+00   1.145522e-11
```
## Conclusion
we conclude that, Random Forest is more accurate than Gradient Boosting Model at upto 99% of accuracy level

## Prediction - using Random Forest MOdel on testing data.


```r
prediction_test<- predict(RF_model, test_data)
prediction_test
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
