#SInce we are working with the text data, we add stringsASFactors 
tweets = read.csv("tweets.csv", stringsAsFactors = FALSE)
str(tweets)
#since we are interested in negative review we add extra colum 
#this sets Negative score to TRUE if Avg is lessthan -1 and FALSE if Avg is more than -1
tweets$Negative = as.factor(tweets$Avg <= -1)
table(tweets$Negative)
library(tm)
#install snowballC also
library(SnowballC)
corpus = Corpus(VectorSource(tweets$Tweet))
corpus
corpus[[1]]
corpus = tm_map(corpus, tolower)
#convert all documents in the corpus to the PlainTextDocument type
corpus = tm_map(corpus, PlainTextDocument)
corpus[[1]]
#remove punchuations
corpus = tm_map(corpus, removePunctuation)
corpus[[1]]
#we want to remove stop words in our tweets
stopwords("english")[1:10]
# we are removing stopwords and apple word also,
#    since apple word repeats many times so not useful every time in text
corpus = tm_map(corpus, removeWords, c("apple", stopwords("english")))
corpus[[1]]
# using stemmer document
corpus = tm_map(corpus, stemDocument)
corpus[[1]]
 
#
frequencies = DocumentTermMatrix(corpus)
frequencies
inspect(frequencies[1000:1005, 505:515])
findFreqTerms(frequencies, lowfreq = 20)
#threshold 0.995 which keep only those words which are 0.05% significant
sparse = removeSparseTerms(frequencies, 0.995)
sparse
#covert sparse to data frmae
tweetsSparse = as.data.frame(as.matrix(sparse)) 
colnames(tweetsSparse) = make.names(colnames(tweetsSparse))
tweetsSparse$Negative = tweets$Negative
#set training set and testing set
set.seed(123)
ind <- sample(2, nrow(tweetsSparse), replace = TRUE, prob = c(0.7,0.3))
trainData <- tweetsSparse[ind==1,]
testData <- tweetsSparse[ind==2,]
#now built sentiment model
library(rpart)
library(rpart.plot)
tweetCart = rpart(Negative ~ ., data = trainData, method = "class")
prp(tweetCart)
predictCart = predict(tweetCart, newdata = testData, type = "class")
table(testData$Negative, predictCart)
library(randomForest)
set.seed(123)
tweetRF = randomForest(Negative ~., data = trainData)
predictRF = predict(tweetRF, newdata = testData)
table(testData$Negative, predictRF)
