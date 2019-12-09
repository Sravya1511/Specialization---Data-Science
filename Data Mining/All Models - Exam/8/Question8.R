getwd()
setwd("F:/MSIT/Year 2/Specialization - Data Science/Data Mining/Final Exam - Extended/8")


stockdata = read.csv("Stock_Index_table.csv", header = TRUE)

#stockdata$sgr = append(diff(stockdata$Close, lag=1, differences = 1)/ stockdata$Close, 0, after = 0)

#getting sgr column
library(dplyr)
new_stockdata = mutate(stockdata, sgr = lead((lag(Close) - Close) / Close))
#new_stockdata$sgr = format(new_stockdata$sgr, scientific = FALSE, digits = 4)

#last row in sgr column
lastcol = nrow(new_stockdata)
a = new_stockdata$sgr[lastcol-1]
b = new_stockdata$sgr[lastcol-2]
c = new_stockdata$sgr[lastcol-3]
new_stockdata$sgr[lastcol] = mean(a,b,c)


#calculating z-scores
sgrmean <- mean(new_stockdata$sgr, na.rm=TRUE)
sgrsd <- sd(new_stockdata$sgr,na.rm=TRUE)
z<-(new_stockdata[,8] - sgrmean) / sgrsd
sort(z)
new_stockdata$zscores <- z

#Dates of the outliers 
dates<-subset(new_stockdata[,1],  new_stockdata[,"zscores"] >= 3.0 | new_stockdata[,"zscores"] <= -3.0)
View(dates)

