rm(list=ls())

atusact.dat <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusact_2017/atusact_2017.dat", header = T, sep = ",", as.is = T)
atuscps_2017.dat <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atuscps_2017/atuscps_2017.dat", header = T, sep = ",")
atusresp_2017.dat <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusresp_2017/atusresp_2017.dat", header = T, sep = ",")
atusrost_2017.dat <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusrost_2017/atusrost_2017.dat", header = T, sep = ",")
atussum_2017.dat <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atussum_2017/atussum_2017.dat", header = T, sep = ",")
atuswho_2017.dat <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atuswho_2017/atuswho_2017.dat", header = T, sep = ",")

#####
#merge response and summaries together to get daily average estimates
#by day. 
atusresp_2017.dat <- atusresp_2017.dat[order(atusresp_2017.dat$TUCASEID),]
atussum_2017.dat <- atussum_2017.dat[order(atussum_2017.dat$TUCASEID),]
#I don't like this way yet. 
respSum <- cbind(atussum_2017.dat, atusresp_2017.dat)

sleeping <- atussum_2017.dat$t010101
sleeplessness <- atussum_2017.dat$t010102
tapply(respSum$t010101, respSum$TUDIARYDATE, mean)/60
tapply(respSum$t010102, respSum$TUDIARYDATE, mean)/60

#compare sleep to sleeplesness 
sum(sleeping)/sum(sleeplessness)
#participants got about 97 more times sleep than sleeplessness

#what is the average amount of sleep people got? 
mean(atusact.dat$TUACTDUR[which(atusact.dat$TRCODE == 010101)], na.rm = T)/60
quantile((atusact.dat$TUACTDUR[which(atusact.dat$TRCODE == 010101)])/60, c(0.025, 0.25, 0.5, 0.75, 0.975))
min(sleeping)/60
max(sleeping)/60
avgEstSleep <- 

#what is the average amount of sleeplessness people got? 
mean(atusact.dat$TUACTDUR[which(atusact.dat$TRCODE == 010102)], na.rm = T)/60
quantile((atusact.dat$TUACTDUR[which(atusact.dat$TRCODE == 010102)])/60, c(0.025, 0.25, 0.5, 0.75, 0.975))
min(sleeplessness)/60
max(sleeplessness)/60

#how many participants are there?
length(unique(atusact.dat$TUCASEID)) #10,223


#what activities get the most time?
timeSpent <- apply(atussum_2017.dat[,25:421], 2, sum)/60
timesSpent.dat <- data.frame(timeSpent)
#t010101 = sleeping = 89742.9 hours
#t120303 = "Television and movies (not religious)" = 31,008.33 hrs
#t050101 = "Work, main job" = 25,371.52 hours
#t110101 = "Eating and drinking" = 11,297.48 hours
#t010201 = "Washing, dressing and grooming oneself" = 6,821.73 hours
#t120101 = "Socializing and communicating with others" = 6,189.8 hours
#t020201 = "Food and drink preparation" = 5,213.38 hours
#t020101 = "Interior cleaning" = 4,060.48 hours
#t120301 = "Relaxing, thinking" = 3,858.17 hours
#t120312 = "Reading, for personal interest = 3,286.8 hours

#####
#how many activities did someone do in a day?
#####
numActivities <- data.frame(tapply(atusact.dat$TUACTIVITY_N, atusact.dat$TUCASEID, length))
colnames(numActivities) <- "numActivities"
numActivities <- cbind(numActivities$numActivities, rownames(numActivities))
colnames(numActivities) <- c("numActivities", 'TUCASEID')
respSum <- cbind(respSum, numActivities)
respSum$numActivities <- as.numeric(respSum$numActivities)
sleepSummaries <- respSum[,c(1, 2, 4, 5, 11, 13:14, 19, 445, 439, 22, 24, 17, 18, 521, 597, 516, 25, 26)]
sleepSummaries$t010101 <- sleepSummaries$t010101/60
sleepSummaries$t010102 <- sleepSummaries$t010102/60
sleepSummaries$sleeplessInd <- as.numeric(sleepSummaries$t010102 != 0)
####
#response meanings
#TESEX: 2 = women
#TEMJOT: -1 = blank, 1="yes", 2="no"
#TRSPFTPT:  -1 = blank, 1='Full Time', 2='Part Time', 3='Hours vary'
#TELFS: 1 = 'employed - at work', 2='employed, absent', 3='unemp, layoff', 4='unemp, looking', 5=not in labor force
#TEHRUSLT: -1 = blank
#TRTHH: time spent taking care of children < 13 years old
#TRERNWA: weekly earnings from work
#TRChildNum: Number of children
#TUECYTD: Did you take care of the elderly yesterday? -1 = 'blank', -2='idk', 1='yes', 2='no'
#TESCHENR = "Are you enrolled in high school, college, or university?";
#TESCHLVL = "Would that be high school, college, or university?";

#####
#Plot each covariate with the response
#####
library(MASS) 
library(lmtest) 
library(car) 
library(bestglm)

windows() 
par(mfrow=c (2,3)) 
for (i in(3:13)) { scatter.smooth(sleepSummaries[,i] , sleepSummaries$sleeplessInd, xlab=colnames(sleepSummaries)[i] , ylab="Sleeplessness")
}

#numActivities
par(mfrow=c(1,2))
hist(sleepSummaries$numActivities[which(sleepSummaries$sleeplessInd==0)], xlab = "No Sleeplessness", main = "", freq = F)
hist(sleepSummaries$numActivities[which(sleepSummaries$sleeplessInd==1)], col="lightblue", xlab="Yes", main="", freq = F)

#Gender
tapply(sleepSummaries$sleeplessInd[which(sleepSummaries$sleeplessInd==1)], sleepSummaries$TESEX[which(sleepSummaries$sleeplessInd==1)], length)
tapply(sleepSummaries$sleeplessInd, sleepSummaries$TESEX, length)
297/4642 #for men
435/5581 #for women
#more sleeplessness in women

#Age
par(mfrow=c(1,2))
hist(sleepSummaries$TEAGE[which(sleepSummaries$sleeplessInd==0)], xlab = "No Sleeplessness", main = "", freq = F)
hist(sleepSummaries$TEAGE[which(sleepSummaries$sleeplessInd==1)], col="lightblue", xlab="Yes", main="", freq = F)

#Hours worked
par(mfrow=c(1,2))
hist(sleepSummaries$TEHRUSLT[which(sleepSummaries$sleeplessInd==0)], xlab = "No Sleeplessness", main = "", freq = F, breaks = 20)
hist(sleepSummaries$TEHRUSLT[which(sleepSummaries$sleeplessInd==1)], col="lightblue", xlab="Yes", main="", freq = F, breaks = 20)

#Number of Children
tapply(sleepSummaries$sleeplessInd[which(sleepSummaries$sleeplessInd==1)], sleepSummaries$TRCHILDNUM[which(sleepSummaries$sleeplessInd==1)], length)
tapply(sleepSummaries$sleeplessInd, sleepSummaries$TRCHILDNUM, length)
470/6131 #for 0 children
116/1708 #for 1 child
101/1572 #for 2 children
34/576 # for 3 children
11/171 #for 4 children

#for weekly earnings
par(mfrow=c(1,2))
hist(sleepSummaries$TRERNWA[which(sleepSummaries$sleeplessInd==0)], xlab = "No Sleeplessness", main = "", freq = F, breaks = 20)
hist(sleepSummaries$TRERNWA[which(sleepSummaries$sleeplessInd==1)], col="lightblue", xlab="Yes", main="", freq = F, breaks = 20)

#Elderly Care
tapply(sleepSummaries$sleeplessInd[which(sleepSummaries$sleeplessInd==1)], sleepSummaries$TUECYTD[which(sleepSummaries$sleeplessInd==1)], length)
tapply(sleepSummaries$sleeplessInd, sleepSummaries$TUECYTD, length)
33/489 #for yes
86/1298 #for no

######
#Consider a logistic model
######
#take out diarydate
sleepLogisticData <- sleepSummaries[,c(3:16, 20)]

######
#take out unecessary variables
######
bestglmEDA <- bestglm(sleepLogisticData, IC='BIC')
bestglmEDA$BestModel
best.glm <- bestglmEDA$BestModel
summary(best.glm)
coeffBIC <- cbind(best.glm$coefficients, confint(best.glm))


bestglmAIC <- bestglm(sleepLogisticData, IC='AIC')
bestglmAIC$BestModel
best.glmAIC <- bestglmAIC$BestModel

summary(best.glmAIC)
coeff <- cbind(best.glmAIC$coefficients, confint(best.glmAIC))
#none of these models explain the data well. I wonder if 
#I need more data or a different approach. 

####
#read in 2016 and 2015 data
####
resp2016 <- read.csv("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusresp_2016/atusresp_2016.dat", header = T, sep = ",")
act2016 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusact_2016/atusact_2016.dat", header = T, sep = ",", as.is = T)
sum2016 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atussum_2016/atussum_2016.dat", header = T, sep = ",")
resp2016 <- resp2016[order(resp2016$TUCASEID),]
sum2016 <- sum2016[order(sum2016$TUCASEID),]
respSum2016 <- cbind(sum2016, resp2016)

resp2015 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusresp_2015.dat", header = T, sep=",")
act2015 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusact_2015.dat", header = T, sep=",")
sum2015 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atussum_2015.dat", header = T, sep=",")
resp2015 <- resp2015[order(resp2015$TUCASEID),]
sum2015 <- sum2015[order(sum2015$TUCASEID),]
respSum2015 <- cbind(sum2015, resp2015)

resp2014 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusresp_2014.dat", header = T, sep=",")
act2014 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atusact_2014.dat", header = T, sep=",")
sum2014 <- read.table("C:/Users/kevin/Desktop/Career Prep/Kevin work Portfolio/Personal Project/SleepData/atussum_2014.dat", header = T, sep=",")
resp2014 <- resp2014[order(resp2014$TUCASEID),]
sum2014 <- sum2014[order(sum2014$tucaseid),]
respSum2014 <- cbind(sum2014, resp2014)

library(gtools)
respSumPrev <- smartbind(respSum2016, respSum2015, respSum2014)

respSumBig <- smartbind(respSum, respSumPrev)
sleepSumBig <- respSumBig[,c(1, 2, 500, 4, 5, 11, 13:14, 19, 10, 20, 22, 24, 17, 18, 504, 579, 25, 26)]
sleepSumBig$t010101 <- sleepSumBig$t010101/60
sleepSumBig$t010102 <- sleepSumBig$t010102/60
sleepSumBig$sleeplessInd <- as.numeric(sleepSumBig$t010102 != 0)


######
#Now, consider a logistic model
######
#take out diarydate
sleepLogisticDataBig <- sleepSummaries[,c(4:17, 20)]

######
#take out unecessary variables
######
bestglmEDABig <- bestglm(sleepLogisticDataBig, IC='BIC')
bestglmEDABig$BestModel
best.glmBig <- bestglmEDABig$BestModel
summary(best.glmBig)
coeffBig <- cbind(best.glmBig$coefficients, confint(best.glmBig))


bestglmAICBig <- bestglm(sleepLogisticDataBig, IC='AIC')
bestglmAICBig$BestModel
best.glmAICBig <- bestglmAICBig$BestModel

summary(best.glmAICBig)


###########
#consider PCA
###########
#PCA only should use predictors that are numeric and standardized
#I can do a log transformation and then
#use center and scale as parameters in prcomp()
sleepSumBigNumerics <- sleepSumBig[,c(4, 9: 12, 14)]
sleepSumBigNumerics <- cbind(sleepSumBigNumerics, respSumBig[,27:421])
sleepSumBigNumerics$SleeplessInd <- sleepSumBig$sleeplessInd

samp <- sample(nrow(sleepSumBigNumerics), nrow(sleepSumBigNumerics)*0.8)
sleepTrain <- sleepSumBigNumerics[samp,]
sleepTest <- sleepSumBigNumerics[-samp,]

sleep.pca <- prcomp(na.omit(sleepTrain[, which(apply(sleepTrain, 2, var) !=0)]), scale. = T, center=T)
print(sleep.pca)

par(mfrow=c(1,1))
plot(sleep.pca, type="l", xlim=c(0, 10))
summary(sleep.pca)
pca.xs <- sleep.pca$x

#use pcas to predict
predict(sleep.pca, newdata = sleepTest)


#preprocess the data and make it ready for any machine
#learning work. 
library(caret)
PC <- preProcess(sleepTrain[, which(apply(sleepTrain, 2, var) !=0)], method=c("BoxCox", "center", 
                                      "scale", "zv", "pca"))
summary(PC)
head(PC)
predictorsTrans = data.frame(
  trans = predict(PC, sleepTrain[, which(apply(sleepTrain, 2, var) !=0)]))
