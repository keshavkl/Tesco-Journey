library(readr)
library(dplyr)
library(Matrix)
library(xgboost)

t_data <- read_csv("transaction_data.csv")

new <- t_data %>% group_by(household_key) %>%
  summarize(number = n())

data <- subset(t_data, household_key <= 150)
data <- data %>% arrange(desc(household_key))
data$Bought <- 1

library(caTools)
set.seed(999)
sample <- sample.split(data, SplitRatio = .75)
train <- subset(data, sample == TRUE)
test <- subset(data, sample == FALSE)

train.y <- vector(train$Bought)
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
                nrounds = 800,
                nfold = 2,
                metrics = {'auc'}
)
which.min(xcv$test.auc.mean)
plot(xcv$test.auc.mean)

clf <- xgb.train(   params              = param, 
                    data                = dtrain, 
                    nrounds             = which.min(xcv$test.logloss.mean), 
                    verbose             = 1,
                    watchlist           = watchlist,
                    maximize            = FALSE
)
importance_matrix <- xgb.importance(train_new@Dimnames[[2]], model = clf)
#co_occurence <- xgb.importance(train_new@Dimnames[[2]], model = clf, 
#              data = train_new, label = target.y)
xgb.plot.importance(importance_matrix)


#Accuracy on Training Set
tpred <- predict(clf, train_new)
table(round(tpred), target.y)
sum(diag(table(round(tpred), target.y)))/sum(table(round(tpred), target.y))

test_new$target <- -1
test_new <- sparse.model.matrix(target ~ ., data = test_new)

preds <- predict(clf, test_new)
submission <- data.frame(ID = test$ID, PredictedProb = preds)
write.csv(submission, "submission.csv", row.names = F)




