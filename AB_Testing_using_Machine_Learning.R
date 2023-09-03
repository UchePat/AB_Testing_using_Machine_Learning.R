# A/B Testing with Machine Learning
# With the rise of digital marketing led by tools including Google Analytics, Google Adwords, and Facebook Ads, a key competitive advantage for businesses is using A/B testing to determine effects of digital marketing efforts. 
# Why? In short, small changes can have big effects. This is why A/B testing is a huge benefit.
# A/B Testing enables us to determine whether changes in landing pages, popup forms, article titles, and other digital marketing decisions improve conversion rates and ultimately customer purchasing behavior. 
# A successful A/B Testing strategy can lead to massive gains - more satisfied users, more engagement, and more sales 

# A key competitive advantage for businesses is using A/B testing
# A major issue with traditional, statistical-inference approaches to A/B Testing is that it only compares 2 variables - an experiment/control to an outcome. 
# The problem is that customer behavior is vastly more complex than this. Customers take different paths, spend different amounts of time on the site, come from different backgrounds (age, gender, interests), and more. 
# This is where Machine Learning excels - generating insights from complex systems.

# In this article you will experience how to implement Machine Learning for A/B Testing step-by-step. 
# After reading this post you will:
# Understand what A/B Testing is
# Understand why Machine Learning is a better approach for performing A/B Testing versus traditional statistical inference (e.g. z-score, t-test)
# Get a Step-by-Step Walkthrough for implementing machine learning for A/B Testing in R using 3 different algorithms:
# Linear Regression
# Decision Trees
# XGBoost
# Develop a Story for what contributes to the goal of gaining Enrollments

# What is A/B Testing?
# A/B Testing is a tried-and-true method commonly performed using a traditional statistical inference approach grounded in a hypothesis test (e.g. t-test, z-score, chi-squared test). 
# In plain English, 2 tests are run in parallel:
# - Treatment Group (Group A) - This group is exposed to the new web page, popup form, etc.
# - Control Group (Group B) - This group experiences no change from the current setup.
# The goal of the A/B is then to compare the conversion rates of the two groups using statistical inference.
# The problem is that the world is not a vacuum involving only the experiment (treatment vs control group) and effect. The situation is vastly more complex and dynamic. 
# Consider these situations:
# Users have different characteristics: Different ages, genders, new vs returning, etc
# Users spend different amounts of time on the website: Some hit the page right away, others spend more time on the site
# Users are find your website differently: Some come from email or newsletters, others from web searches, others from social media
# Users take different paths: Users take actions on the website going to different pages prior to being confronted with the event and goal

# Often modeling an A/B test in this vacuum can lead to misunderstanding of the true story.
# This is where Machine Learning can help.

# We can use an example from Udacity's A/B Testing Course, but apply the applied Machine Learning techniques from our Business Analysis with R Course to gain better insights into the inner-workings of the system rather than simply comparing an experiment and control group in an A/B Test.
# We'll implement Machine Learning to perform the A/B test using the R statistical programming language
# Experiment Name: "Free Trial" Screener
 
# In the experiment, Udacity tested a change where if the student clicked "start free trial", they were asked how much time they had available to devote to the course.
# If the student indicated 5 or more hours per week, they would be taken through the checkout process as usual. If they indicated fewer than 5 hours per week, a message would appear indicating that Udacity courses usually require a greater time commitment for successful completion.
# Why Implement the Form?
# The goal with this popup was that this might set clearer expectations for students upfront, thus reducing the number of frustrated students who left the free trial because they didn't have enough time.
# However, what Udacity wants to avoid is "significantly" reducing the number of students that continue past the free trial and eventually complete the course.

# Project Goal
# In this analysis, we will investigate which features are contributing to enrollments and determine if there is an impact on enrollments from the new "Setting Expectations" form.
# The users that experience the form will be denoted as "Experiment = 1"
# The control group (users that don't see the form) will be denoted as "Experiment = 0".

# Get the Data
# The data set for this A/B Test can be retrieved from Kaggle Data Sets.

# Core packages
library(tidyverse)
library(tidyquant)

# Modeling packages
library(parsnip)
library(recipes)
library(rsample)
library(yardstick)
library(broom)

# Connector packages
library(rpart)
library(rpart.plot)
library(xgboost)

# Import data
control_tbl <- read_csv("control_data.csv")

experiment_tbl <- read_csv("experiment_data.csv")

# 3.4 Investigate the Data
control_tbl %>% head(5)
# We have 5 columns consisting of:
# Date: a character formatted Day, Month, and Day of Month
# Pageviews: An aggregated count of Page Views on the given day
# Clicks: An aggregated count of Page Clicks on the given day for the page in question
# Enrollments: An aggregated count of Enrollments by day.
# Payments: An aggregated count of Payments by day.

# Next, we inspect the "Control Group" using glimpse() the data to get an idea of the data format (column classes and data size).
str(control_tbl)

control_tbl %>% glimpse()

# Last, we can check the "Experiment" group (i.e. the Treatment Group) with glimpse() as well to make sure it's in the same format.
str(experiment_tbl)

experiment_tbl %>% glimpse()
# Key Points:
# 37 total observations in the control set and 37 in the experiment set

# The Dataset is time-based and aggregated by day - This isn't the best way to understand complex user behavior, but we'll go with it
# We can see that Date is formatted as a character data type. This will be important when we get to data quality. We'll extract day of the week features out of it.
# Data between the experiment group and the control group is in the same format. Same number of observations (37 days) since the groups were tested in parallel.


# 3.5 Data Quality Check
# Next, let's check the data quality. We'll go through a process involves:
# Check for Missing Data - Are values missing? What should we do?
# Check Data Format - Is data in correct format for analysis? Are all features created and in the right class?

# 3.5.1 Check for Missing Data
# The next series of operations calculates the count of missing values in each column with map(~ sum(is.na(.))), converts to long format with gather(), and arranges descending with arrange(). 
control_tbl %>%
  map_df(~ sum(is.na(.))) %>%  # checks each column for NAs
  gather(key = "feature", value = "missing_count") %>%  # creates a new columns - feature (with all the column-headers as values) and missing_count(showing the corresponding total NAs in each column-header)
  arrange(desc(missing_count))
# Key Point: We have 14 days of missing observations that we need to investigate

# Let's see if the missing data (NA) is consistent in the experiment set.
experiment_tbl %>% 
  map_df(~ sum(is.na(.))) %>%
  gather(key = "feature", value = "missing_count") %>%
  arrange(desc(missing_count))
# Key Point: The count of missing data is consistent (a good thing). We still need to figure out what's going on though.


# Let's see which specific rows of values are missing in a particular column using the filter().
control_tbl %>%
  filter(is.na(Enrollments))
# Key Point: We don't have Enrollment information from November 3rd on. We will need to remove these observations.


# 3.5.2 Check Data Format
# We'll just check the data out, making sure its in the right format for modeling.
control_tbl %>% glimpse()
# Key Points:
# Date is in character format. It doesn't contain year information. Since the experiment was only run for 37 days, we can only realistically use the "Day of Week" as a predictor.
# The other columns are all numeric, which is OK. We will predict the number of Enrollments (regression) 
# Payments is an outcome of Enrollments so this should be removed.


# 3.6 Format Data
# Now that we understand the data, let's put it into the format we can use for modeling. We'll do the following:
# Combine the control_tbl and experiment_tbl, adding an "id" column indicating if the data was part of the experiment or not
# Add a "row_id" column to help for tracking which rows are selected for training and testing in the modeling section
# Create a "Day of Week" feature from the "Date" column
# Drop the unnecessary "Date" column and the "Payments" column
# Handle the missing data (NA) by removing these rows.
# Shuffle the rows to mix the data up for learning
# Reorganize the columns
set.seed(123)

data_formatted_tbl <- control_tbl %>%
  # Combine with Experiment data
  bind_rows(experiment_tbl, .id = "Experiment") %>%  # the Experiment column uses values 1 and 2 to indicate if a row of data is from control_tbl or experiment_tbl object 
  mutate(Experiment = as.numeric(Experiment) - 1) %>%  # changes Experiment values to 0 and 1
  # Add row id/index column
  mutate(row_id = row_number()) %>%   
  # Create a Day of Week feature
  mutate(DOW = str_sub(Date, start = 1, end = 3) %>%   # using str_sub() to extract the 1st 3 letters/characters(which are the days of the week) from Date column and display them separately under a new column
           factor(levels = c("Sun", "Mon", "Tue", "Wed", 
                             "Thu", "Fri", "Sat"))
  ) %>%
  select(-Date, -Payments) %>%
  # Remove missing data
  filter(!is.na(Enrollments)) %>%
  # re-shuffle the whole data (note that set.seed is used to make reproducible)
  sample_frac(size = 1) %>%
  # Reorganize columns
  select(row_id, Enrollments, Experiment, everything())

head(data_formatted_tbl)

data_formatted_tbl %>% glimpse()

data_formatted_tbl$Experiment <- as_factor(data_formatted_tbl$Experiment)


# 3.7 Training and Testing Sets
# With the data formatted properly for analysis, we can now separate into training and testing sets using an 80% / 20% ratio. 
# We can use the initial_split() function from rsample to create a split object, then extracting the training() and testing() sets.
set.seed(123)

split_obj <- data_formatted_tbl %>%
  initial_split(prop = 0.8, strata = "Experiment")

train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

# We can take a quick glimpse of the training data. 
train_tbl %>% glimpse()
# It's 36 observations randomly selected.

# And, we can take a quick glimpse of the testing data. 
test_tbl %>% glimpse()
# It's the remaining 10 observations.


# 3.8 Implement Machine Learning Algorithms
# We'll implement the new parsnip R package.
# Our strategy will be to implement 3 modeling approaches:
# - Linear Regression - Linear, Explainable (Baseline)
# - Decision Tree
# Pros: Non-Linear, Explainable.
# Cons: Lower Performance
# - XGBoost
# Pros: Non-Linear, High Performance
# Cons: Less Explainable
 
# # 3.8.1 Linear Regression (Baseline)
model_01_lm <- linear_reg("regression") %>%
  set_engine("lm") %>%
  fit(Enrollments ~ ., data = train_tbl %>% select(-row_id))

summary(model_01_lm)

model_01_lm_2$results 

# Next, we can make predictions on the test set using predict(). 
# We bind the predictions with the actual values ("Enrollments" from the test set). 
# Then we calculate the error metrics using metrics() from the yardstick package. 
# knitr::kable() used for pretty tables
model_01_lm %>%
  predict(new_data = test_tbl) %>%   # use d model to predict on test data
  bind_cols(test_tbl %>% select(Enrollments)) %>%   # combine the predicted Enrollment and actual Enrollment values for comparison
  metrics(truth = Enrollments, estimate = .pred) %>%
  knitr::kable()
# We can see that the model is off by about +/-19 Enrollments on average(based on MAE value). 
# We can see that the R-squared value is very very low which is a problem and the RMSE and MAE values are high


# We can investigate the predictions by visualizing them using ggplot2. 
# This is the best way to view/analyse the actual values and d predicted values side by side using visualization
model_01_lm %>%
  # Format Data
  predict(test_tbl) %>%  # use d model to predict on test data
  bind_cols(test_tbl %>% select(Enrollments)) %>%   # combine the predicted Enrollment and actual Enrollment values for comparison
  mutate(observation = row_number() %>% as.character()) %>%   # creates an index column
  gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%  # change .pred and Enrollment columns into values under key column and their corresponding values is under value column
  
  # Visualize
  ggplot(aes(x = observation, y = value, color = key)) +
  geom_point() +
  expand_limits(y = 0) +  # starting the y-axis from value 0
  theme_tq() +
  scale_color_tq() +
  labs(title = "Enrollments: Prediction vs Actual",
       subtitle = "Model 01: Linear Regression (Baseline)")
# After formatting and plotting the data, we can see that the model had an issue with Observation 1,2,4,8(bcus the predicted and actual values are not close at all) which is likely the reason for the low R-squared value (test set).


# We can use tidy() from broom package to view the model stats(such as std.error, p-values of all columns) to understand what is driving this model
# We will arrange by 'p-value' to see how important the model terms are. 
linear_regression_model_terms_tbl <- model_01_lm$fit %>%
  tidy() %>%   # we view the model stats including p-values
  arrange(p.value) %>%
  mutate(term = as_factor(term) %>% fct_rev())

linear_regression_model_terms_tbl

# creating a nice table
linear_regression_model_terms_tbl %>%
  knitr::kable()
# Clicks, Pageviews, DOWMon are judged strong predictors with p-values < 0.05. However we wanto try out other modelling techniques to judge this.
# We note that d coefficient of Experiment is -14.2(under estimate column), and bcus the term is binary(0/1) this can be interpreted as decreasing Enrollments by -14.2 per day when the Experiment is run.


# We can visualize the importance by separating "p.values" of < 0.05 from "p.values" of >= 0.05 with a red dotted line.
linear_regression_model_terms_tbl %>%
  ggplot(aes(x = p.value, y = term)) +
  geom_point(color = "#2C3E50") +
  geom_vline(xintercept = 0.05, linetype = 2, color = "red") +   # creates a vertical line at x-axis value stated
  theme_tq() +
  labs(title = "Feature Importance",
       subtitle = "Model 01: Linear Regression (Baseline)")
# Key Points:
# Our model is on average off by +/-19 enrollments (which is the MAE value). The test set R-squared is quite low at 0.329.
# We investigated the predictions to see if there is anything that jumps out at us. The model had an issue with observation 1,2,4,8, which is likely throwing off the R-squared value.
# We investigated feature importance. Clicks, Pageviews, and DOWMon are the most important features. Experiment is 5th, with a p.value 0.100. Typically this is considered insignificant.
# We can also see the term coefficient for Experiment is -14.2 indicating as decreasing Enrollments by -14.2 per day when the Experiment is run.
 
# Before we move onto the next model, we can setup some helper functions to reduce repetitive code.

#--> 3.8.2 Helper Functions
# We'll make some helper functions to reduce repetitive code and increase readability. 
# First, we'll make a simplified metric reporting function, calc_metrics().
calc_metrics <- function(model, new_data) {
  model %>%
    predict(new_data = new_data) %>%
    bind_cols(new_data %>% select(Enrollments)) %>%
    metrics(truth = Enrollments, 
            estimate = .pred)
}

# Next we can make a simplified visualization function, plot_predictions().
plot_predictions <- function(model, new_data) {
  
  g <- predict(model, new_data) %>%
    bind_cols(new_data %>% select(Enrollments)) %>%
    mutate(observation = row_number() %>% as.character()) %>%
    gather(key = "key", value = "value", -observation, factor_key = TRUE) %>%
    
    # Visualize
    ggplot(aes(x = observation, y = value, color = key)) +
    geom_point() +
    expand_limits(y = 0) +
    theme_tq() +
    scale_color_tq()
  
  return(g)
}


# 3.8.3 Decision Trees
# Decision Trees are excellent models that can pick up on non-linearities and often make very informative models that compliment linear models by providing a different way of viewing the problem. 
# We can implement a decision tree with decision_tree(). 
# We'll set the engine to "rpart", a popular decision tree package. There are a few key tunable parameters:
# cost_complexity: A cutoff for model splitting based on increase in explainability
# tree_depth: The max tree depth
# min_n: The minimum number of observations in terminal (leaf) nodes
# The parameters selected for the model were determined using 5-fold cross validation to prevent over-fitting. This is discussed in Important Considerations. 
model_02_decision_tree <- decision_tree(
  mode = "regression",
  cost_complexity = 0.001, 
  tree_depth = 5, 
  min_n = 4) %>%
  set_engine("rpart") %>%
  fit(Enrollments ~ ., data = train_tbl %>% select(-row_id))

# Next, we can calculate the metrics on this model using our helper function, calc_metrics(). 
# knitr::kable() used for pretty tables
model_02_decision_tree %>% 
  calc_metrics(test_tbl) %>%
  knitr::kable()
# We have a better RSQ value and the RMSE and MAE values are lower
# The MAE of the predictions is lower than that of the linear model at +/-14.3 Enrollments per day.


# We can visualize how its performing on the observations using our helper function, plot_predictions(). 
model_02_decision_tree %>% 
  plot_predictions(test_tbl) +
  labs(title = "Enrollments: Prediction vs Actual",
       subtitle = "Model 02: Decision Tree")
# The model is having issues with Observations 4,5,6


# And finally, we can use rpart.plot() to visualize the decision tree rules. 
# Note that we need to extract the underlying "rpart" model from the parsnip model object using the model_02_decision_tree$fit.
model_02_decision_tree$fit %>%
  rpart.plot(
    roundint = FALSE, 
    cex = 0.8, 
    fallen.leaves = TRUE,
    extra = 101, 
    main = "Model 02: Decision Tree")
# Interpreting the decision tree is straightforward: Each decision is a rule, and Yes is to the left, No is to the right. 
# The top features are the most important to the model ("Pageviews" and "Clicks"). The decision tree shows that "DOW and Experiment" is involved in the decision rules. 
# The rules indicate a when Experiment >= 0.5, there is a drop in enrollments.
# Key Points:
# Our new model has a higher accuracy to +/-14 enrollments (MAE value) than the linear regression model.
# Experiment shows up towards the bottom of the tree. The rules indicate a when Experiment >= 0.5, there is a drop in enrollments.


# 3.8.4 XGBoost
# The final model we'll implement is an xgboost model. Several key tuning parameters include:
# mtry: The number of predictors that will be randomly sampled at each split when creating the tree models.
# trees: The number of trees contained in the ensemble.
# min_n: The minimum number of data points in a node that are required for the node to be split further.
# tree_depth: The maximum depth of the tree (i.e. number of splits).
# learn_rate: The rate at which the boosting algorithm adapts from iteration-to-iteration.
# loss_reduction: The reduction in the loss function required to split further.
# sample_size: The amount of data exposed to the fitting routine.
# Understanding these parameters is critical to building good models. 
# The parameters selected for the model were determined using 5-fold cross validation to prevent over-fitting. This is discussed in Important Considerations. 
set.seed(123)

model_03_xgboost <- boost_tree(
  mode = "regression",
  mtry = 100, 
  trees = 1000, 
  min_n = 8, 
  tree_depth = 6, 
  learn_rate = 0.2, 
  loss_reduction = 0.01, 
  sample_size = 1) %>%
  set_engine("xgboost") %>%
  fit(Enrollments ~ ., data = train_tbl %>% select(-row_id))

# We can get the Testing set performance using our custom calc_metrics() function. 
# knitr::kable() used for pretty tables
model_03_xgboost %>% 
  calc_metrics(test_tbl) %>%
  knitr::kable()
# We can see that the MAE is 10.6 indicating the model is off by an average 10.6 enrollments per day on the test set.


# We can visualize how its performing on the observations using our helper function, plot_predictions(). 
model_03_xgboost %>% 
  plot_predictions(test_tbl) +
  labs(title = "Enrollments: Prediction vs Actual",
       subtitle = "Model 02: Decision Tree")
# We can see that it's performing poorly only on Observation 4.


# We want to understand which features are important to the XGBoost model. 
# We can get the global feature importance from the model by extracting the underlying model from the parsnip object using model_03_xgboost$fit and piping this into the function xgb.importance().
xgboost_feature_importance_tbl <- model_03_xgboost$fit %>%
  xgb.importance(model = .) %>%
  as_tibble() %>%
  mutate(Feature = as_factor(Feature) %>% fct_rev())

xgboost_feature_importance_tbl %>% knitr::kable()
# We can see the top 3 variables important to d model


# Next, we can plot the Feature Importance.
# We can see that the model is largely driven by Pageviews and Clicks.
xgboost_feature_importance_tbl %>%
  ggplot(aes(x = Gain, y = Feature)) +
  geom_point(color = "#2C3E50") +
  geom_label(aes(label = scales::percent(Gain)), 
             hjust = "inward", color = "#2C3E50") +
  expand_limits(x = 0) +
  theme_tq() +
  labs(title = "XGBoost Feature Importance") 
# The information gain is 95% from Pageviews and Clicks combined. Experiment has about a 5% contribution to information gain, indicating it's still predictive (just not nearly as much as Pageviews). 
# This tells a story that if Enrollments are critical, Udacity should focus on getting Pageviews.
# Key Points:
# The XGBoost model error has dropped to +/-10.6 Enrollments.
# The XGBoost shows that Experiment provides an information gain of 5%
# The XGBoost model tells a story that Udacity should be focusing on Page Views and secondarily Clicks to maintain or increase Enrollments. The features drive the system.

 
# 3.10 Business Conclusions - Key Benefits to Machine Learning
# There are several key benefits to performing A/B Testing using Machine Learning. These include:
# Understanding the Complex System - We discovered that the system is driven by Pageviews and Clicks. Statistical Inference would not have identified these drivers. Machine Learning did.
# Providing a direction and magnitude of the experiment - We saw that Experiment = 1 drops enrollments by -19.6 Enrollments Per Day in the Linear Regression. 
# We saw similar drops in the Decision Tree rules. Statistical inference would not have identified magnitude and direction. Only whether or not the Experiment had an effect.

# What Should Udacity Do?
# If Udacity wants to maximimize enrollments, it should focus on increasing Page Views from qualified candidates. Page Views is the most important feature in 2 of 3 models.
# If Udacity wants to alert people of the time commitment, the additional popup form is expected to decrease the number of enrollments. 
# The negative impact can be seen in the decision tree (when Experiment <= 0.5, Enrollments go down) and in the linear regression model term (-19.6 Enrollments when Experiment = 1). Is this OK? It depends on what Udacity's goals are.
# But this is where the business and marketing teams can provide their input developing strategies to maximize their goals - More users, more revenue, and/or more course completions.


# 3.11 Important Considerations: Cross Validation and Improving Modeling Performance
# Two important further considerations when implementing an A/B Test using Machine Learning are:
# How to Improve Modeling Performance
# The need for Cross-Validation for Tuning Model Parameters


# 3.11.1 How to Improve Modeling Performance
# A different test setup would enable significantly better understanding and modeling performance. Why?
# The data was AGGREGATED - To truly understand customer behavior, we should run the analysis on unaggregated data to determine probability of an individual customer enrolling.
# There are NO features related to the Customer in the data set - The customer journey and their characteristics are incredibly important to understanding complex purchasing behavior. Including GOOD features is the best way to improving model performance, and thus insights into customer behavior.


#----------------------------------------------------------------------------------------------- 


####------> 3.11.2 Need for Cross-Validation for Tuning Models <-----####
# Note: use 5-fold cross-validation on the 3 Models and compare their MAE values in a Tabular form
# In practice, we need to perform cross-validation to prevent the models from being tuned to the test data set.
# The parameters for the Decision Tree and XGBoost Models were selected using 5-Fold Cross Validation. The results are as follows.

# 

# 3.7 Training and Testing Sets
# With the data formatted properly for analysis, we can now separate into training and testing sets using an 80% / 20% ratio. 
# We can use the initial_split() function from rsample to create a split object, then extracting the training() and testing() sets.
set.seed(123)

split_obj <- data_formatted_tbl %>%
  initial_split(prop = 0.8, strata = "Experiment")

train_tbl <- training(split_obj)
test_tbl  <- testing(split_obj)

train_tbl <- train_tbl %>% select(-row_id)
test_tbl <- test_tbl %>% select(-row_id)

# We can take a quick glimpse of the training data. 
train_tbl %>% glimpse()
# It's 36 observations randomly selected.

# And, we can take a quick glimpse of the testing data. 
test_tbl %>% glimpse()
# It's the remaining 10 observations.

#----> Creating Custom Control Parameters and Cross-validation
# lets use trainControl() function which enables parameter coefficient estimation using resampling approaches such as cross-validation and boosting
library(caret)

set.seed(1234)

cvcontrol <- trainControl(method = "repeatedcv",  # number = 10 means creating 10 fold cross validation which means d training data is broken into 10 parts, den d model is made from 9 parts while d 10th part is used for error estimation. 
                          number = 10,      # Then d model is made again from 9 parts while d 8th part is used for error estimation. Again d model is made again from 9 parts while d 7th part is used for error estimation and so on...thus creating 10 models with diff parts used for error estimation for each model   
                          repeats = 5,     # use 2 or 5 as values
                          )


# 3.8 Implement Machine Learning Algorithms
# We'll implement the new parsnip R package.
# Our strategy will be to implement 3 modeling approaches:
# - Linear Regression - Linear, Explainable (Baseline)
# - Decision Tree
# Pros: Non-Linear, Explainable.
# Cons: Lower Performance
# - XGBoost
# Pros: Non-Linear, High Performance
# Cons: Less Explainable

#---> 3.8.1 Linear Regression (Baseline)
# lets create a Linear Regression model using caret package
model_01_lm <- train(Enrollments ~ ., data = train_tbl,         # using d entire dataset
                  method = "lm", trControl = cvcontrol)

summary(model_01_lm)

model_01_lm$results 

# Next, we can make predictions on the test set using predict(). 
mypred <- model_01_lm %>%
  predict(new_data = test_tbl)   # use d model to predict on test data


head(mypred)    # displays the predicted values for 1st 6 rows
head(test_tbl$Enrollments)    # dis is the test/validate data. so compare the predicted column values(in head(mypred)) to d actual values in Enrollments column in test data 



#--> We can re-run the Linear Model using only the relevant variables
model_01_lm_2 <- train(Enrollments ~ Pageviews + Clicks, 
                     data = train_tbl,         # using d entire dataset
                     method = "lm", trControl = cvcontrol)

summary(model_01_lm_2)

model_01_lm_2$results 

# Next, we can make predictions on the test set using predict(). 
mypred2 <- model_01_lm_2 %>%
  predict(new_data = test_tbl)   # use d model to predict on test data
  

head(mypred2)    # displays the predicted values for 1st 6 rows
head(test_tbl$Enrollments)    # dis is the test/validate data. so compare the predicted column values(in head(mypred)) to d actual values in Enrollments column in test data 



#---> 3.8.3 Decision Trees
# Decision Trees are excellent models that can pick up on non-linearities and often make very informative models that compliment linear models by providing a different way of viewing the problem. 
set.seed(1234)

# Using tuneLength() parameter
mydecision = train(Enrollments ~ ., 
                  data = train_tbl, 
                  method = "rpart",  # for classification tree
                  tuneLength = 5,  # choose up to 5 combinations of tuning parameters (cp). use values 2,5,7 or 10
                  metric = "RMSE",  # evaluate hyperparameter combinations with RMSE
                  trControl = trainControl(
                    method = "cv",  # k-fold cross validation
                    number = 10,  # 10 folds
                    savePredictions = "final"       # save predictions for the optimal tuning parameter
                  )
)

plot(mydecision)

# Visualizing d Decision Tree
rpart.plot(mydecision$finalModel)

# Prediction on Testing data
yrpred <- predict(mydecision, test_tbl, type = "raw")

plot(test_tbl$Enrollments, yrpred, 
     main = "Simple Regression: Predicted vs. Actual",
     xlab = "Actual",
     ylab = "Predicted")

# RMSE value
rmse2 <- RMSE(pred = yrpred,
              obs = test_tbl$Enrollments)
rmse2


# Important Variables to d model
plot(varImp(mydecision), main="Variable Importance with Simple Regression")




#---> 3.8.4 XGBoost
# The final model we'll implement is an xgboost model. 
# The parameters selected for the model were determined using 5-fold cross validation to prevent over-fitting. 
set.seed(123)

# using tuneGrid() parameter
yrboost <- train(Enrollments ~ ., 
                 data = train_tbl,
                 method = "xgbTree",   
                 trControl = cvcontrol,
                 tuneGrid = expand.grid(nrounds = 500,
                                        max_depth = 4,    # max_depth, eta and gamma parameter values can be tweaked to get a better model 
                                        eta = 0.28,     # eta is learning rate. if d value is low dat means d model learns slowly as such u need to increase nrounds parameter value
                                        gamma = 1.8,
                                        colsample_bytree = 1,
                                        min_child_weight = 1,
                                        subsample = 1))
yrboost

varImp(yrboost)     # displays all columns and their respective importance to d model

v <- varImp(yrboost)
plot(v)

plot(v, 5)   # displays only d top 5(or 10) columns important to d model

# Prediction and Misclassification Error
ourpred2 <- predict(yrboost, test_tbl, type = 'raw')


