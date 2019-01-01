NBA_train = read.csv("NBA_train.csv")
str(NBA_train)
########################
table(NBA_train$W, NBA_train$Playoffs)
##
#it seems that the teams which wins more thana 42 likely seems to qualify for playoffs
#so making use of it find how team wins using points scored and points allowed
NBA_train$diff = NBA_train$PTS - NBA_train$oppPTS
#make a scatter plot to see any relation blw points diff and wins
plot(NBA_train$diff, NBA_train$W)
###
#develop a model on Wins depending on difference
WinsReg = lm(W ~ diff, data = NBA_train)
summary(WinsReg)
################
PointsReg = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB +DRB + TOV + STL + BLK, data =  NBA_train)
summary(PointsReg)
PointsReg$residuals
SSE = sum(PointsReg$residuals ^2)
SSE
RMSE = sqrt(SSE/nrow(NBA_train))
RMSE
mean(NBA_train$PTS)
####################
#
#
#remove the not significant values
#remove TOV because it has high p-values
PointsReg2 = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB +DRB + STL + BLK, data =  NBA_train)
summary(PointsReg2)
######
#remone one more
PointsReg3 = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB + STL + BLK, data =  NBA_train)
summary(PointsReg3)
###########
#remove one more
PointsReg4 = lm(PTS ~ X2PA + X3PA + FTA + AST + ORB + STL, data =  NBA_train)
summary(PointsReg4)
SSE4 = sum(PointsReg4$residuals ^2)
RMSE4 = sqrt(SSE4/nrow(NBA_train))
SSE4
RMSE4
##########################################
#                                        #
# Now make predictions on test data      #
#                                        #
##########################################
NBA_test = read.csv("NBA_test.csv")
PointsPredictions = predict(PointsReg4, newdata = NBA_test)
SSE = sum((PointsPredictions - NBA_test$PTS)^2)
SST = sum((mean(NBA_test$PTS) - NBA_test$PTS)^2)
R2 = 1 - SSE/SST
R2

