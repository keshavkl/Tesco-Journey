### This code is to predict whether a customer will buy a product or not when 
### he/she next visits the store.

#### Strategy: Use transaction data to see whether a customer bought a product or not
####           on a perticular day. Use that data as the dependent variable. 
####           Then use Household statistics to predict whether the customer will 
####           buy a product in his/her next visit. 

##### Loading Data ######

library(readr)
library(dplyr)
library(Matrix)
library(xgboost)

t_data <- read_csv("transaction_data.csv")
hh_data <- read_csv("hh_demographic.csv")


### Take the first 400 households as sample for analysis and merge them by household_key
## note that household_keys which have no household data are omitted.
t_data_subset <- subset(t_data, household_key <= 400)
hh_data_subset <- subset(hh_data, household_key <= 400)

newdata <- merge.data.frame(t_data_subset, hh_data_subset, by = "household_key")

### define the dependent variable
## If the user bought a product, the QUANTITY would be 1 or more. So we 
## code a variable to denote 0 for not bought and 1 for bought.

newdata$Bought[newdata$QUANTITY == 0] <- 0
newdata$Bought[newdata$QUANTITY > 0] <- 1

table(newdata$Bought) 
### There is considerable skew as seen in the table. This will be handled later

### Remove all transaction data that directly denote whether the user bought the product or not
newdata$QUANTITY <- NULL
newdata$BASKET_ID <- NULL
newdata$SALES_VALUE <- NULL
newdata$RETAIL_DISC <- NULL
newdata$COUPON_DISC <- NULL
newdata$COUPON_MATCH_DISC <- NULL
newdata$WEEK_NO <- NULL
newdata$TRANS_TIME <- NULL
newdata$DAY <- NULL
newdata$PRODUCT_ID <- NULL

# Change character variables into factors
newdata[sapply(newdata, is.character)] <- lapply(newdata[sapply(newdata, is.character)], 
                                                 as.factor)

newdata$STORE_ID <- as.factor(newdata$STORE_ID)
newdata$household_key <- NULL

###Split dataset into training and testing sets
library(caTools)
set.seed(999)
sample <- sample.split(newdata, SplitRatio = .70)

train <- subset(newdata, sample == T)
test <- subset(newdata, sample == F)

### Using ROSE package to balance the dependent variable  using Oversampling for
### minority output(0) and undersampling the majority output(1)

library(ROSE)
train <- ovun.sample(Bought ~ ., data = train, 
                       method = "both", p=0.5, N=245932, seed = 1)$data
table(train$Bought)

### remove dependent variable from test. 
train.y <- train$Bought
test.y <- test$Bought

test$Bought <- NULL

train_new <- train
test_new <- test


### Model Building using Extreme Gradient Boosting.
### Used step size = 0.05, sample within training data is set at 75%
train_new$Bought <- train.y
train_new <- sparse.model.matrix(Bought ~ ., data = train_new)

dtrain <- xgb.DMatrix(data=train_new, label=train.y)
watchlist <- list(train_new = dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.3,
                max_depth           = 5,
                subsample           = 0.75,
                colsample_bytree    = 0.7
)

### Cross Validation to see the mose optimal iteration.
xcv <- xgb.cv(  params = param,
                data = dtrain,
                nrounds = 2000,
                nfold = 2,
                metrics = {'auc'}
)
which.max(xcv$test.auc.mean)
plot(xcv$test.auc.mean)

### Actual Model optimized using the best iteration  from CV.
clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = which.max(xcv$test.auc.mean), 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)
### Importance Matrix to understand feature importance
importance_matrix <- xgb.importance(train_new@Dimnames[[2]], model = clf)
xgb.plot.importance(importance_matrix)

##### PREDICTIONS####
### Owing to the use of Oversampling and undersampling techniques, It is generally found that
### in-sample prediction is considerably lower. However, out-of-sample prediction gives a good prediction.

### Prediction on the Training Set (Threshold set at 0.50)
tpred <- predict(clf, train_new)
tpred[tpred <= 0.50] <- 0
tpred[tpred > 0.50] <- 1
table(tpred, train.y)
train_acc <- sum(diag(table(tpred, train.y)))/sum(table(tpred, train.y))

### Prediction on the testing set.
test_new$Bought <- -1
test_new <- sparse.model.matrix(Bought ~ ., data = test_new)

preds <- predict(clf, test_new)

##Threshold set at 0.50
preds[preds <= 0.50] <- 0
preds[preds >0.50] <- 1
table(preds, test.y)
test_acc <- sum(diag(table(preds, test.y)))/sum(table(preds, test.y))

## ________________________________________________________________________##

####  ***** Testing the accuracy on out-of-sample set 1 ******

#### Setting up data
t_data_subset1 <- subset(t_data, household_key > 400 & household_key <= 800)
hh_data_subset1 <- subset(hh_data, household_key > 400 & household_key <= 800)

newdata1 <- merge.data.frame(t_data_subset1, hh_data_subset1, by = "household_key")

newdata1$Bought[newdata1$QUANTITY == 0] <- 0
newdata1$Bought[newdata1$QUANTITY > 0] <- 1

actual1 <- newdata1$Bought

newdata1$QUANTITY <- NULL
newdata1$Bought <- NULL
newdata1$BASKET_ID <- NULL
newdata1$SALES_VALUE <- NULL
newdata1$RETAIL_DISC <- NULL
newdata1$COUPON_DISC <- NULL
newdata1$COUPON_MATCH_DISC <- NULL
newdata1$WEEK_NO <- NULL
newdata1$TRANS_TIME <- NULL
newdata1$DAY <- NULL
newdata1$household_key <- NULL

### Converting classes
newdata1[sapply(newdata1, is.character)] <- lapply(newdata1[sapply(newdata1, is.character)], 
                                                   as.factor)
newdata1$STORE_ID <-  as.factor(newdata1$STORE_ID)

#### Prediction using Xtreme Gradient Boosting
newdata1$Bought <- -1
newdata1 <- sparse.model.matrix(Bought ~ ., data = newdata1)

preds1 <- predict(clf, newdata1)
preds1[preds1 <= 0.50] <- 0
preds1[preds1 >0.50] <- 1
### Confusion Matrix
table(preds1, actual1)
### Accuracy
os_acc_1 <- sum(diag(table(preds1, actual1)))/sum(table(preds1, actual1))

## __________________________________________________________________________##

####  ***** Testing the accuracy on out-of-sample set 2 ******

#### Setting up data
t_data_subset2 <- subset(t_data, household_key > 800 & household_key <= 1600)
hh_data_subset2 <- subset(hh_data, household_key > 800 & household_key <= 1600)

newdata2 <- merge.data.frame(t_data_subset2, hh_data_subset2, by = "household_key")

newdata2$Bought[newdata2$QUANTITY == 0] <- 0
newdata2$Bought[newdata2$QUANTITY > 0] <- 1

actual3 <- newdata2$Bought

newdata2$QUANTITY <- NULL
newdata2$Bought <- NULL
newdata2$BASKET_ID <- NULL
newdata2$SALES_VALUE <- NULL
newdata2$RETAIL_DISC <- NULL
newdata2$COUPON_DISC <- NULL
newdata2$COUPON_MATCH_DISC <- NULL
newdata2$WEEK_NO <- NULL
newdata2$TRANS_TIME <- NULL
newdata2$DAY <- NULL
newdata2$household_key <- NULL

### Converting classes
newdata2[sapply(newdata2, is.character)] <- lapply(newdata2[sapply(newdata2, is.character)], 
                                                   as.factor)
newdata2$STORE_ID <-  as.factor(newdata2$STORE_ID)

#### Prediction using Xtreme Gradient Boosting
newdata2$Bought <- -1
newdata2 <- sparse.model.matrix(Bought ~ ., data = newdata2)

preds2 <- predict(clf, newdata2)
preds2[preds2 <= 0.50] <- 0
preds2[preds2 >0.50] <- 1
os_acc_2 <- sum(diag(table(preds2, actual3)))/sum(table(preds2, actual3))


####  ***** Testing the accuracy on out-of-sample set 3 ******

#### Setting up data
t_data_subset3 <- subset(t_data, household_key > 1601 & household_key <= 2499)
hh_data_subset3 <- subset(hh_data, household_key > 1601 & household_key <= 2499)

newdata3 <- merge.data.frame(t_data_subset3, hh_data_subset3, by = "household_key")

newdata3$Bought[newdata3$QUANTITY == 0] <- 0
newdata3$Bought[newdata3$QUANTITY > 0] <- 1

actual3 <- newdata3$Bought

newdata3$QUANTITY <- NULL
newdata3$Bought <- NULL
newdata3$BASKET_ID <- NULL
newdata3$SALES_VALUE <- NULL
newdata3$RETAIL_DISC <- NULL
newdata3$COUPON_DISC <- NULL
newdata3$COUPON_MATCH_DISC <- NULL
newdata3$WEEK_NO <- NULL
newdata3$TRANS_TIME <- NULL
newdata3$DAY <- NULL
newdata3$household_key <- NULL

### Converting classes
newdata3[sapply(newdata3, is.character)] <- lapply(newdata3[sapply(newdata3, is.character)], 
                                                   as.factor)
newdata3$STORE_ID <-  as.factor(newdata3$STORE_ID)

#### Prediction using Xtreme Gradient Boosting
newdata3$Bought <- -1
newdata3 <- sparse.model.matrix(Bought ~ ., data = newdata3)

preds3 <- predict(clf, newdata3)
preds3[preds3 <= 0.50] <- 0
preds3[preds3 >0.50] <- 1
os_acc_3 <- sum(diag(table(preds3, actual3)))/sum(table(preds3, actual3))








