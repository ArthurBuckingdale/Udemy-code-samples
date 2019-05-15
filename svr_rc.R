# Regression Template

# Importing the dataset
dataset = read.csv('Position_Salaries.csv')
dataset = dataset[2:3]


# Fitting the Regression Model to the dataset
#install.packages('e1071')
regressor = svm(formula = Salary ~ .,
                data = dataset,
                type = 'eps-regression')


# Predicting a new result
y_pred = predict(regressor, data.frame(Level = ))
 
# Visualising the Regression Model results
# install.packages('ggplot2')
library(ggplot2)
ggplot() +
  geom_point(aes(x = dataset$Level, y = dataset$Salary),
             colour = 'red') +
  geom_line(aes(x = dataset$Level, y = predict(regressor, newdata = dataset)),
            colour = 'blue') +
  ggtitle('Truth or Bluff (SVR Model)') +
  xlab('Level') +
  ylab('Salary')

