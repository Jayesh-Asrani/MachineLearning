#Simple Linear Regression

library(caTools)
library(ggplot2)
#Importing the DataSet
Salary_DataSet=read.csv('Salary_Data.csv')

#Splitting DataSet 

set.seed()
split=sample.split(Salary_DataSet$Salary,2/3)
training_set=subset(Salary_DataSet,split == TRUE)
test_set=subset(Salary_DataSet,split == FALSE)  

# Fitting Simple Linear Model
LR_Model=lm(formula = Salary ~ YearsExperience,
            data=training_set)

# Prediting 
y_pred = predict(LR_Model,newdata=test_set)

# Visualising the results
#install.packages("ggplot2")

ggplot() +
  geom_point(aes(x = training_set$YearsExperience ,y = training_set$Salary),
             color='Red') +
  geom_line(aes(x = training_set$YearsExperience , y = predict(LR_Model,newdata=training_set)),
            color='Blue') + 
  ggtitle('Salary vs Exp (Training Set)') +
  xlab("Years of Exp") +
  ylab("Salary")

ggplot() +
  geom_point(aes(x = test_set$YearsExperience ,y = test_set$Salary),
             color='Red') +
  geom_line(aes(x = training_set$YearsExperience , y = predict(LR_Model,newdata=training_set)),
            color='Blue') + 
  ggtitle('Salary vs Exp (Test Set)') +
  xlab("Years of Exp") +
  ylab("Salary")
