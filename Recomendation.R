movies = read.table("MovieLens.txt", header = FALSE, sep = "|", quote = "\"")
str(movies)
#we can see that there are 24 variable   #these are comments 
#since the header is not there in text
# we add the variable names to the data
colnames(movies) = c("ID", "Title", "ReleaseDate", "VideoReleaseDate", 
                     "IMDB", "Unknown", "Action", "Adventure", 
                     "Animation", "Childrens", "Comedy", "Crime", 
                     "Documentary", "Drama", "Fantasy", "FilmNoir", 
                     "Horror", "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western")
str(movies)
#We don't want the ID, Release Date, Video release date, IMDB
#    , so remove them
# so assigning them to null will remove them
movies$ID = NULL
movies$ReleaseDate = NULL
movies$VideoReleaseDate = NULL
movies$IMDB = NULL
#there are duplicate data so we remove them using unique function
movies = unique(movies)
str(movies)
#
#We use hierarchial cluster to cluster movies by gener
#first we have to find differnces blw the points and then we have to cluster the points
#we cluster on gener not on title
distances = dist(movies[2:20], method = "euclidean")
clustermovies = hclust(distances, method = "ward.D")
#now plot dentrogram
plot(clustermovies)
#You see the graph is messy since of names will be written of 1600 observations
#Choose cluster size as 10
#that means we have 10 different types of clusters
clusterGroups = cutree(clustermovies, k=10)
tapply(movies$Action, clusterGroups, mean)
#it caluclates the avg value of action variable for each cluster
tapply(movies$Romance, clusterGroups, mean)
#
#How how do u apply recommendation system
#just check where the Men in Black movie belongs to
subset(movies, Title =="Men in Black (1997)")
#you can see that Men in black is numbered at 257, so go check what cluster group it is
clusterGroups[257]
#now extract all movies which are under cluster2
cluster2 = subset(movies, clusterGroups ==2)
cluster2$Title[1:10]
