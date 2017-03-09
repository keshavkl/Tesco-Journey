library(readr)
library(dplyr)
library(Matrix)
library(xgboost)

t_data <- read_csv("transaction_data.csv")
hh_data <- read_csv("hh_demographic.csv")

t_data_subset <- subset(t_data, household_key <= 400)
hh_data_subset <- subset(hh_data, household_key <= 400)

t_data_subset$RETAIL_DISC <- sqrt((t_data_subset$RETAIL_DISC)^2)
t_data_subset$COUPON_MATCH_DISC <- sqrt((t_data_subset$COUPON_MATCH_DISC)^2)
t_data_subset$COUPON_DISC <- sqrt((t_data_subset$COUPON_DISC)^2)

t_data_subset$TotalDiscount <- t_data_subset$COUPON_DISC + t_data_subset$COUPON_MATCH_DISC + 
                        t_data_subset$RETAIL_DISC

t_data_subset$RETAIL_DISC <- NULL
t_data_subset$COUPON_MATCH_DISC <- NULL
t_data_subset$COUPON_DISC <- NULL

t_data_subset$WEEK_NO <- NULL

newdata <- merge.data.frame(t_data_subset, hh_data_subset, by = "household_key")

newdata[sapply(newdata, is.character)] <- lapply(newdata[sapply(newdata, is.character)], 
                                       as.factor)
newdata$Bought[newdata$QUANTITY == 0] <- 0
newdata$Bought[newdata$QUANTITY > 0] <- 1
newdata$QUANTITY <- NULL

newdata$BASKET_ID <- NULL
newdata$PRODUCT_ID <- NULL
newdata$STORE_ID <- NULL
newdata$TRANS_TIME <- as.numeric(newdata$TRANS_TIME)


library(caTools)
set.seed(999)
sample <- sample.split(newdata$household_key, SplitRatio = .70)
train <- subset(newdata, sample == TRUE)
test <- subset(newdata, sample == FALSE)

train.y <- train$Bought
test.y <- test$Bought
train$Bought <- NULL
test$Bought <- NULL


train_new <- train
test_new <- test

train_new$Bought <- train.y
train_new <- sparse.model.matrix(Bought ~ ., data = train_new)

dtrain <- xgb.DMatrix(data=train_new, label=train.y)
watchlist <- list(train_new = dtrain)

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.03,
                max_depth           = 5,
                subsample           = 0.75,
                colsample_bytree    = 0.7
)

xcv <- xgb.cv(  params = param,
                data = dtrain,
                nrounds = 50,
                nfold = 2,
                metrics = {'auc'}
)
which.min(xcv$test.auc.mean)
plot(xcv$test.auc.mean)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = which.min(xcv$test.auc.mean), 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)
importance_matrix <- xgb.importance(train_new@Dimnames[[2]], model = clf)
xgb.plot.importance(importance_matrix)







