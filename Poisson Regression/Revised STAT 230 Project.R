rm(list=ls())
library(ggplot2)

#My data is following a binomial distribution.
#I also have a sample size of 20 for each of the 20 shooters
#That sample size is enough for binomial regression.

#variances, in regards to p, will variable. p, the proportion of
#each shooter will be a random variable. log tranformations
#help account for the variability. 
#each five people in a group will have a p. value. 
#CLT theorem for each group 5*20 shots will 
#help estimate p values. Variance of a proportion is
#pi *(1-pi)/n. Variance of proportion is .05/10. 

freethrows = read.table("Free Throws.csv", header=T,sep=",") 
#the reponse variable has a unit of makes per attempts
#since I divided the makes by 20 attempts, the amount of significant digits decreased from 4
#I think I use proportions/percentages
quantile(freethrows$Response, 0.5)
#I found the median to be 0.475.

######## EXPLORATORY DATA ANALYSIS######### 

freethrows$Group = as.factor(freethrows$Group) 
freethrows$Noise = as.factor(freethrows$Noise) 
freethrows$Inflation = as.factor(freethrows$Inflation) 

hist(freethrows$Response)
dotchart(freethrows$Response, groups=freethrows$Group)
qqnorm(freethrows$Response)
#are my ANOVA assumptions met from using a log transformation? Yes
logsurv = log(freethrows$Response)##Log Transformation## 
qqnorm(logsurv)
hist(logsurv)
##Feel free to make captions 3-4 sentences long
##so that you clear up any questions a reader may have about the units in a table.
##make a table self sustaining. 
##Consider including the table from Free Throws.csv into the report. Or I can use the histogram we got

hist(logsurv,main="Histogram of Free Throw Percentage", xlab="Percentage of Free Throws Made", col="lightblue",breaks=5) 
ggplot(freethrows, aes(freethrows$Response)) + geom_histogram(binwidth = 0.05) + ggtitle("Shooting Percentages") + xlab("Shooting Percentages") + ylab("Frequency")

#Assumptions for Binomial regression:
#1. proportions are the same

boxplot(Response ~ Group, names = c("Inflated","Deflated","Inflated w/ Noise","Deflated w/ Noise"), xlab="Group",ylab="Free Throw Percentage",data=freethrows) 
#need to show them the raw data, not the log transformations
boxplot(logsurv ~ Group, names = c("Inflated","Deflated","Inflated w/ Noise","Deflated w/ Noise"), xlab="Group", ylab="Free Throw Percentage", data=freethrows) 
ggplot(freethrows, aes(x=Group, y=Response)) + geom_boxplot(fill=c("green", "pink", "lightblue", "orange")) + labs(title="Group Distributions", y = "Shooting Percentage")
ggplot(freethrows, aes(x=Group, y=logsurv)) + geom_boxplot(fill=c("green", "pink", "lightblue", "orange")) + labs(title="Group Distributions", y = "Log Transformations")

#I should use my log transformation instead of my raw data.
aggregate(logsurv ~ Group, sd, data=freethrows)
aggregate(logsurv ~ Inflation+Noise, mean, data=freethrows) 

###### TREATMENT GROUP MEANS###### 
mean11 = mean(logsurv[freethrows$Noise==1 & freethrows$Inflation==1]) 
mean10 = mean(logsurv[freethrows$Noise==1 & freethrows$Inflation==0]) 
mean01 = mean(logsurv[freethrows$Noise==0 & freethrows$Inflation==1]) 
mean00 = mean(logsurv[freethrows$Noise==0 & freethrows$Inflation==0]) 

###### INTERACTION PLOT ########## 
plot(c(1,2),c(mean11,mean01),type="l", ylim=c(-2,0), xlim=c(0.5,2.5), xlab="", ylab="Free Throw Percentage", xaxt="n", lwd=3, col="red", cex.lab=1.4) 
axis(1,at=c(1,2),labels=c("No Noise","Noise"),cex.axis=1.4) 
lines(c(1,2),c(mean10,mean00),col="blue",lwd=3) 
legend(x=2,y=0,legend=c("Inflated","Deflated"), col=c("red","blue"),lty=c(1,1),lwd=c(3,3)) 

plot(c(1,2),c(mean11,mean10),type="l", ylim=c(-2,0), xlim=c(0.5,2.5), xlab="", ylab="Free Throw Percentage", xaxt="n", lwd=3, col="red", cex.lab=1.4) 
axis(1,at=c(1,2),labels=c("Inflated","Deflated"),cex.axis=1.4) 
lines(c(1,2),c(mean01,mean00),col="blue",lwd=3) 
legend(x=2,y=0,legend=c("No Noise","Noise"),col=c("red","blue"),lty=c(1,1),lwd=c(3,3)) 

#binomial regression
reg1 <- glm(freethrows$Response~Inflation+Noise+Inflation*Noise, family=binomial, data=freethrows)
summary(reg1)

#anova test on a log transformation
anova(lm(logsurv~Inflation+Noise+Inflation*Noise, data=freethrows))
#anova(lm(freethrows$Response~Inflation+Noise+Inflation*Noise, data=freethrows))
TukeyHSD(aov(lm(logsurv ~ Group, data=freethrows)))

#null deviance is the base model. 
#16 degrees of freedom come after taking the three factors. 
#3 degree of freedom and a chi square of 1.2. 

#try running a poisson regression
#I am running GLM on the regular data, not log transformations
reg_pois <- glm(freethrows$Response*20~Inflation+Noise+Inflation*Noise, family=poisson, data=freethrows)
summary(reg_pois)
pois_diff <- 35.945 - 22.174
1 - pchisq(pois_diff, df=3)
predict.glm(reg_pois, type = "link")
#The counts are not the same = conclusion. 
#use the counts because they have more variability n*p*q. 

reg_pois <- glm(freethrows$Response*20~Inflation+Noise, family=poisson, data=freethrows)
summary(reg_pois)
pois_diff_2 <- 35.945 - 25.802
1 - pchisq(pois_diff_2, df=2)
#The interaction and the main effects are significant. 

reg_pois <- glm(freethrows$Response*20~Inflation, family=poisson, data=freethrows)
summary(reg_pois)
pois_diff_3 <- 25.825 - 25.802
1 - pchisq(pois_diff_3, df=1)
predict.glm(reg_pois, type = "link")

reg_pois <- glm(freethrows$Response*20~Noise, family=poisson, data=freethrows)
summary(reg_pois)
pois_diff_4 <- 35.922 - 25.802
1 - pchisq(pois_diff_4, df=1)
predict.glm(reg_pois, type = "link")
#I am trying the predict function for a glm object
#What are the results telling me? 


#I'll also try the linear model for log transformed data.

#the difference from when we threw out the interaction (25.802) is significantly different
#The difference from when we threw out the inflaction factor is 35.922. There is a significant difference.
#the difference from the base model of 25.802 when we threw out noise, is less than 1. 25.825-25.802 isn't a difference. 
