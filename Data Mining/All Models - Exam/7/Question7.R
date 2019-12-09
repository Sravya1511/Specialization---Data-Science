getwd()
setwd("F:/MSIT/Year 2/Specialization - Data Science/Data Mining/Final Exam - Extended/7")


data = read.csv("bupa.csv", header = FALSE, col.names = c("1", "2", "3", "4", "5", "6", "7"))


str(data)
summary(data)


x = data[,1:5]
y = data[,6]
fit = kmeans(x,4)
library(class)
knnfit = knn(fit$centers,x,as.factor(c(-2,-1,1,2)))
error = 1-sum(knnfit == y)/length(y)
error
#Misclassification Error Rate = 0.9536232
