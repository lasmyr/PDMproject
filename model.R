setwd("~/Coursera Files/09 Practical Data Modelling")
library(caret)
library(dplyr)
library(randomForest)
library(pROC)


# Read Data
data <- read.csv("pml-training.csv", na.strings = c("NA", "#DIV/0", ""))
testdata <- read.csv("pml-testing.csv",na.strings = c("NA", "#DIV/0", ""))

summary(data)
summary(testdata)

# Delete NAs
navals <- data.frame(matrix(ncol = 2, nrow = 0))
for (i in 1:ncol(data)) {
  if(sum(is.na(data[,i]))>0) {
    navals <- rbind(navals, data.frame(sum(is.na(data[,i])), colnames(data)[i]))
  }
}
navals
navalscols <-as.character(navals[,2])
uselesscols <- colnames(data)[grepl("^X|timestamp|window|user|problem", names(data))]
dropcols <- append(navalscols, uselesscols)

# Create Train, Validate, Test data
data_clean<- select(data, -dropcols)
inTrain <- createDataPartition(y = data_clean$classe, p=0.7)[[1]]

for (i in 1:ncol(data_clean)){
  if (colnames(data_clean)[i] == "classe") {
    data_clean[,i] <- as.factor(data_clean[,i])
  } else {data_clean[,i] <- as.numeric(data_clean[,i])
  }
}

train <- data_clean[inTrain,]
valid <- data_clean[-inTrain,]
test <- select(testdata, -dropcols)

# Simple Decision Tree
model_simple <- train(classe ~ ., data = train, method = "rpart")
pred1 <- predict(model_simple, newdata = valid)
pred1_proba <- predict(model_simple, newdata = valid, type = "prob")
confusionMatrix(pred1, valid$classe)
asda<-multiclass.roc(pred1_proba$A, as.numeric(valid$classe))

# Random Forest, paramters tuned
controlRf <- trainControl(method="cv", 5)
train_x <- select(train, -classe)
train_y <- select(train, classe)

model_complex <- train(x = train_x, y = train_x,
                       method="rf", trControl=controlRf, 
                       importance=TRUE, ntree=100)

valid_x <- select(valid, -classe)
pred2 <- predict(model_complex, newdata = valid_x)
pred2_proba <- predict(model_complex, newdata = valid, type = "prob")
confusionMatrix(pred2, valid$classe)
multiclass.roc(pred2_proba$A, as.numeric(valid$classe))

?train

pred2_train <- predict(model_complex, newdata = train)
pred2_train_prob <- predict(model_complex, newdata = train, type = "prob")
confusionMatrix(pred2_train, train$classe)

multiclass.roc(pred2_train$A, as.numeric(train$classe))
barplot(table(pred2_train))

# Predicting test set
test_pred <- predict(model_complex, newdata = test)
test_pred_proba <- predict(model_complex, newdata = test, type = "prob")
test_pred_result <- data.frame(test_pred, test_pred_proba)
colnames(test_pred_result) <- c("prediction", "probability")
View(test_pred_result)

# Plot
par(mfrow=c(2,2))
barplot(table(train$classe), main = "Train Data")
barplot(table(pred1), main = "Simple Decision Tree Predictions")
barplot(table(pred2), main = "Tuned Random Forest Predictions")
barplot(table(test_pred), main = "Test Predictions")
