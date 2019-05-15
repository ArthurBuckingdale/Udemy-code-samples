# Data Preprocessing

# Importing the dataset
dataset = read.csv('50_Startups.csv')

# Encoding categorical data
dataset$State = factor(dataset$State,
                         levels = c('New York', 'California', 'Florida'),
                         labels = c(1, 2, 3))


# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Profit, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
# training_set = scale(training_set)
# test_set = scale(test_set)

#FItting Multiple linear refression to the training set
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend + State, 
               data = dataset)
summary(regressor)

#Predicting the test set results
y_pred = predict(regressor, newdata = test_set)


#building the optimal model using backward elimination
#first we are going to eliminate the state variable
regressor = lm(formula = Profit ~ R.D.Spend + Administration + Marketing.Spend, 
               data = dataset)
summary(regressor)
#second we are going to get rid of the administration
regressor = lm(formula = Profit ~ R.D.Spend  + Marketing.Spend, 
               data = dataset)
summary(regressor)









