#Import the data set
data <- read.csv("data.csv")
str(data)
summary(data)
#

data["Attrition"] =ifelse(data["Attrition"]=="Yes",1,0)
data["Gender"] =ifelse(data["Gender"]=="Male",1,0)
data["OverTime"]=ifelse(data["OverTime"]=="Yes",1,0)
data["MaritalStatus"] =ifelse(data["MaritalStatus"] =="Divorced",0, ifelse(data["MaritalStatus"]=="Married",1,2))
data["Department"] =ifelse(data["Department"] =="Human Resources",0, ifelse(data["Department"]=="Sales",2,1))

#Since there are lot of categorical variables in the data
#make the Algorithm to treatt them as factors


#Make sure about the categorical variables in the data set
data$BusinessTravel <- factor(data$BusinessTravel)
data$Department <- factor(data$Department)
data$Education <- factor(data$Education)
data$EducationField <- factor(data$EducationField)
data$EnvironmentSatisfaction <- factor(data$EnvironmentSatisfaction)
data$Gender <- factor(data$Gender)
data$JobInvolvement <- factor(data$JobInvolvement)
data$JobLevel <- factor(data$JobLevel)
data$JobSatisfaction <- factor(data$JobSatisfaction)
data$OverTime <- factor(data$OverTime)
data$PerformanceRating <- factor(data$PerformanceRating)
data$RelationshipSatisfaction <- factor(data$RelationshipSatisfaction)
data$StockOptionLevel <- factor(data$StockOptionLevel)
data$TrainingTimesLastYear <- factor(data$TrainingTimesLastYear)
data$WorkLifeBalance <- factor(data$WorkLifeBalance)
data$MaritalStatus <- factor(data$MaritalStatus)
data$Department <- factor(data$Department)

#
summary(data)
set.seed(123)
class1 = subset(data,data["Attrition"] == 1)
class0 = subset(data,data["Attrition"] == 0)
smp_size1 <- floor(0.7 * nrow(class1))
smp_size0 <- floor(0.7 * nrow(class0))
train_ind1 <- sample(seq_len(nrow(class1)),size = smp_size1)
train_ind0 <- sample(seq_len(nrow(class0)),size = smp_size0)
train0 <- class0[train_ind0,]
train1 <- class1[train_ind1,]
test0 <- class0[-train_ind0,]
test1 <- class1[-train_ind1,]
#The train and test samples are allocated
train = rbind(train0,train1)
test = rbind(test0,test1)
#
#
str(train)
summary(train)
head(train)
################################
#Now Data preprocessing is over
#There are no Outliers or missing values in the data
#
#
#Train the model using logistic regression
mylogit <-glm(Attrition ~ Age + DailyRate +DistanceFromHome
              + Education+EnvironmentSatisfaction +
                Gender+ HourlyRate + JobInvolvement +
                JobLevel+JobSatisfaction+MonthlyIncome
              +MonthlyRate+NumCompaniesWorked +MaritalStatus
              +OverTime+PercentSalaryHike+ Department+
                PerformanceRating+RelationshipSatisfaction
              +StockOptionLevel + TotalWorkingYears + TrainingTimesLastYear +
                WorkLifeBalance + YearsAtCompany +
                YearsInCurrentRole + YearsSinceLastPromotion + 
                YearsWithCurrManager, data=train, family="binomial")

summary(mylogit)
####
#
#Remove the attributes which have low information
mylogit2 <-glm(Attrition ~ Age + DistanceFromHome
              + EnvironmentSatisfaction + 
                Gender+ JobInvolvement +
                JobSatisfaction+MonthlyIncome +NumCompaniesWorked
              +OverTime+RelationshipSatisfaction
              +StockOptionLevel + TotalWorkingYears + TrainingTimesLastYear +
                WorkLifeBalance + YearsAtCompany +
                YearsInCurrentRole + YearsSinceLastPromotion + 
                YearsWithCurrManager, data=train, family="binomial")

summary(mylogit2)

#See the training accuracy and test accuracy
train_prob <- predict(mylogit2, train,type = "response")
train_out <- ifelse(train_prob>0.5,1,0)
table(train_out, train$Attrition)
#
#Accuracy of training data 
train_logit_accuracy <- (841+70)/(841+95+22+70)
train_logit_accuracy

#Predict the test dataset
test_prob <- predict(mylogit2, test, type = "response")
test_out <-ifelse(test_prob>0.5,1,0)
table(test_out, test$Attrition)
#Caluclate the test set accuracy
test_logit_accuracy=(357+28)/(357+44+13+28)
test_logit_accuracy

#
#Now plot the ROC curve
library(ROCR)
pred <- prediction( test_prob, test$Attrition)
perf <- performance(pred,"tpr","fpr")
plot(perf)
abline(a=0,b=1)


###########
#Use Random forest
library(randomForest)
myrf <-randomForest(as.factor(Attrition) ~ Age + DistanceFromHome
               + EnvironmentSatisfaction + 
                 Gender+ JobInvolvement +
                 JobSatisfaction+MonthlyIncome +NumCompaniesWorked
               +OverTime+RelationshipSatisfaction
               +StockOptionLevel + TotalWorkingYears + TrainingTimesLastYear +
                 WorkLifeBalance + YearsAtCompany +
                 YearsInCurrentRole + YearsSinceLastPromotion + 
                 YearsWithCurrManager, data=train)
summary(myrf)
#See the importance of each variable
importance(myrf)
varImpPlot(myrf,type=2)

#
train_rf <-predict(myrf, train, type = "class")
library(caret)
confusionMatrix(train_rf,train$Attrition)
#
#predict the test data
test_rf <- predict(myrf, test,type = "class")
confusionMatrix(test_rf,test$Attrition)
#########
