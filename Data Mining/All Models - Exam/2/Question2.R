getwd()
setwd("F:/MSIT/Year 2/Specialization - Data Science/Data Mining/Final exam/2")


ap = read.csv("apriori_data.csv", header = TRUE);

ap$TID <- NULL
library(arules)



transactions = read.transactions("ItemList.csv", sep=',', rm.duplicates = TRUE)
rules <- apriori(transactions, parameter = list(sup = 0.03, conf = 0.5,target="rules"))

inspect(sort(basket_rules, by = 'lift')[1:15])

itemFrequencyPlot(transactions, topN = 5)
