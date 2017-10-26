rm(list=ls())

#load the necessary packages
library(datasets)
library(stats)
library(dbscan)
library(cluster)

#load the iris dataset
iris_copy <- iris[,c(1:4)]
k2 <- kmeans(iris_copy, 2)
k2$size
#for k=2, the size of the clusters is 53 and 97.
k2clust <- k2$cluster
sil2 <- silhouette(k2$cluster, dist(iris_copy))
y2 <- summary(sil2)[[4]] #average silhouette width
#the f-measure is 77.6%

k3 <- kmeans(iris_copy, 3)
sil3 <- silhouette(k3$cluster, dist(iris_copy))
y3 <- summary(sil3)[[4]]
#for k=3, the size of the clusters is 62, 38, and 50. 
#The f-measure is 88.4%

k4 <- kmeans(iris_copy, 4)
sil4 <- silhouette(k4$cluster, dist(iris_copy))
y4 <- summary(sil4)[[4]]
#For k=4, the size of the clusters is 27, 28, 50, and 45.
#The f-measure is 91.6%

kmeans(iris_copy, 5)
#For k=5, the size of the clusters is 38, 19, 62, 8, and 23.
#The F-measure is 89.8%

kmeans(iris_copy, 7)
#For k=7, the size of the clusters is 50, 22, 19, 19, 12, 21, and 7.
#The F-measure is 94.5%

kmeans(iris_copy, 9)
#For k=9, the size of the clusters is 23, 22, 28, 8, 19, 12, 19, 8, and 11.
#The F-measure is 95.4%

kmeans(iris_copy, 11)
#For k=11, the size of the clusters is 3, 4, 12, 12, 11, 19, 17, 19, 12, 22, and 19.
#The F-measure is 96.3%

#The value of k that produces the highest F-score is k=11.
#I think it is interesting the value of k=5 produced a lower 
#F-measure than most of the other values of k. 
#That amount of centroids proved to be the least accurate. 


###Cool automatic k-means clustering program

k <- c(2, 3, 4, 5, 7, 9, 11)
km.out <- list()
sil.out <- list()
x <- vector()
y <- vector()
for (i in k){
  set.seed(5)
  km.out[i] <- list(kmeans(iris_copy, centers=i))
  sil.out[i] <- list(silhouette(km.out[[i]]$cluster, dist(iris_copy)))
  
  
  x[i] <- i
  y[i] <- summary(sil.out[[i]])[[4]]
}

#only get the silhouette widths for the k values we are working with. 
y <- y[!is.na(y)]
y <- y[y>0]
y

#Get the F-Measure scores
classes <- as.integer(iris$Species)
#the answers are above. 

###############
#Problem 4
#Hierarchical agglomerative clustering algorithm
###############
agglom <- hclust(dist(iris_copy))

#plot the dendogram
plot(agglom, xlab="Distances")
agglom$height



