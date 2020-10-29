# ---------------------------------------------------------------------


 # +----------------------------------------------------------------------------+
 # |                               the R code for the                           |
 # |                                   Big R-book                               |
 # |                           by Philippe J.S. De Brouwer                      |
 # +----------------------------------------------------------------------------+
 # | Copyright: (C) Philippe De Brouwer, 2020                                   |
 # |            Owners of the book can use this code for private and commercial |
 # |            use, except re-publishing.                                      |
 # +----------------------------------------------------------------------------+
 # | This document is a true copy of all the R code that appears in the book.   |
 # | It is provided for you so that when you work with the book, you can copy   | 
 # | and paste any code segment that you want to test without typing everything |
 # | again.                                                                     |
 # | No warranties or promises are given.                                       |
 # | Have an intersting time and learn faster than me!                          |
 # | Philippe J.S. De Brouwer                                                   |
 # +----------------------------------------------------------------------------+
 # | For your convenience, blocks are separated by: '# ----------------------'. |
 # +----------------------------------------------------------------------------+
 #
# ---------------------------------------------------------------------


rm(list=ls())   # reset R at the beginning of a new part
# ---------------------------------------------------------------------


library(MASS)

# Explore the data by plotting it:
plot(survey$Height, survey$Wr.Hnd)
# ---------------------------------------------------------------------


< dependent variable> tilde <sum of independent variables>
# ---------------------------------------------------------------------


# Create the model:
lm1 <- lm (formula = Wr.Hnd ~ Height, data = survey)
summary(lm1)
# ---------------------------------------------------------------------


# Create predictions:
h <- data.frame(Height = 150:200)
Wr.lm <- predict(lm1, h)

# Show the results:
plot(survey$Height, survey$Wr.Hnd,col="red")
lines(t(h),Wr.lm,col="blue",lwd = 3)
# ---------------------------------------------------------------------


# Or use the function abline()
plot(survey$Height, survey$Wr.Hnd,col = "red",
     main = "Hand span in function of Height", 
     abline(lm(survey$Wr.Hnd ~ survey$Height ),
            col='blue',lwd = 3),
     cex = 1.3,pch = 16,
     xlab = "Height",ylab ="Hand span")
# ---------------------------------------------------------------------


# We use mtcars from the library MASS
model <- lm(mpg ~ disp + hp + wt, data = mtcars)
print(model)
# ---------------------------------------------------------------------


# Accessing the coefficients
intercept <- coef(model)[1]
a_disp    <- coef(model)[2]
a_hp      <- coef(model)[3]
a_wt      <- coef(model)[4]

paste('MPG =', intercept, '+', a_disp, 'x disp +', 
     a_hp,'x hp +', a_wt, 'x wt')
# ---------------------------------------------------------------------


# This allows us to manually predict the fuel consumption
# e.g. for the Mazda Rx4
2.23 + a_disp * 160 + a_hp * 110 + a_wt * 2.62
# ---------------------------------------------------------------------


m <- lm(mpg ~ gear + wt + cyl * am, data = mtcars)
summary(m)
# ---------------------------------------------------------------------


f1 <- mpg ~ gear + wt + cyl * am
f2 <- mpg ~ gear + wt + cyl + am + cyl * am
# ---------------------------------------------------------------------


m <- lm(mpg ~ gear + wt + cyl:am, data = mtcars)
summary(m)
# ---------------------------------------------------------------------


# The . will automatically expand as 'all other variables':
m <- lm(mpg ~ ., data = mtcars)
summary(m)
# ---------------------------------------------------------------------


m <- lm(mpg ~ . * ., data = mtcars)
# ---------------------------------------------------------------------


m <- lm(mpg ~ . * ., data = mtcars)
summary(m)
# ---------------------------------------------------------------------


glm(formula, data, family)
# ---------------------------------------------------------------------


m <- glm(cyl ~ hp + wt, data = mtcars, family = "poisson")
summary(m)
# ---------------------------------------------------------------------


  m <- glm(cyl ~ hp, data = mtcars, family = "poisson")
  summary(m)
# ---------------------------------------------------------------------


# Consider observations for dt = d0 + v0 t + 1/2 a t^2
t  <- c(1,2,3,4,5,1.5,2.5,3.5,4.5,1)
dt <- c(8.1,24.9,52,89.2,136.1,15.0,37.0,60.0,111.0,8)

# Plot these values.
plot(t,  dt, xlab = "time", ylab = "distance")

# Take the assumed values and fit into the model.
model <- nls(dt ~ d0 + v0 * t + 1/2 * a * t^2,
             start = list(d0 = 1,v0 = 3,a = 10))

# Plot the model curve
simulation.data <- data.frame(t = seq(min(t),max(t),len = 100))
lines(simulation.data$t,predict(model,
      newdata = simulation.data), col = "red", lwd = 3)
# ---------------------------------------------------------------------


# Learn about the model:
summary(model)                # the summary
print(sum(residuals(model)^2))# squared sum of residuals
print(confint(model))         # confidence intervals
# ---------------------------------------------------------------------


m <- lm(data = mtcars, formula = mpg ~ wt)
summary(m)
summary(m)$r.squared
# ---------------------------------------------------------------------


# Consider the relation between the hours studied and passing
# an exam (1) or failing it (0):

# First prepare the data:
hours <- c(0,0.50, 0.75, 1.00, 1.25, 1.50, 1.75, 
           1.75, 2.00, 2.25, 2.50, 2.75, 3.00, 3.25,
	   3.50, 4.00, 4.25, 4.50, 4.75, 5.00, 5.50)
pass  <- c(0,0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 
           1, 0, 1, 1, 1, 1, 1, 1)
d <- data.frame(cbind(hours,pass))

# Then fit the model:
m <- glm(formula = pass ~ hours, family = binomial, 
         data = d)
# ---------------------------------------------------------------------


# Visualize the results:
plot(hours, pass, col = "red", pch = 23, bg = "grey",
     xlab = 'Hours studied',
     ylab = 'Passed exam (1) or not (0)')
pred <- 1 / (1+ exp(-(coef(m)[1] + hours * coef(m)[2])))
lines(hours, pred, col = "blue", lwd = 4)
# ---------------------------------------------------------------------


set.seed(1837)
# Split mtcars in two parts:
# ---------------------------------------------------------------------


# First build the random indexes:
N <- nrow(mtcars)
train_idx <- sample(1:N, 0.8 * N)
test_idx  <- setdiff(1:N, train_idx)

# Then build the training and test data set:
d_train <- mtcars[train_idx, ]
d_test  <- mtcars[test_idx, ]
# ---------------------------------------------------------------------


# Fit a seriously over-fit model on the training data:
m <- lm( formula = mpg ~ . * ., data = d_train)  
# ---------------------------------------------------------------------


# The output of summary(m) is rather long, but note that it includes:
# Residuals:
# ALL 25 residuals are 0: no residual degrees of freedom!
# ---------------------------------------------------------------------


# Make predictions for the test dataset:
p <- predict(m, d_test)
# Note that the warning above tells us that the model is way over-fit.
# ---------------------------------------------------------------------


# See how good the predictions are:
cbind(observed    = d_test$mpg, 
      predictions = round(p, 1), 
      diff_abs    = round(p - d_test$mpg, 1),
      diff_pct    = round((p - d_test$mpg) / d_test$mpg, 2) * 100
      )
# ---------------------------------------------------------------------


diff_pct    = round((p - d_test$mpg) / d_test$mpg, 2) * 100
diff_pct_min    = min(diff_pct)
diff_pct_max    = max(diff_pct)
# ---------------------------------------------------------------------


# if necessary: install.packages('titanic')
library(titanic)

# This provides a.o. two datasets titanic_train and titanic_test.
# We will work further with the training-dataset.
t <- titanic_train
colnames(t)
# ---------------------------------------------------------------------


# fit provide a simple model
m <- glm(data    = t, 
         formula = Survived ~ Pclass + Sex + Pclass * Sex + Age + SibSp, 
         family  = binomial)
summary(m)
# ---------------------------------------------------------------------


# We build further on the model m.

# Predict scores between 0 and 1 (odds):
t2 <- t[complete.cases(t),]
predicScore <- predict(object=m,type="response", newdat = t2)

# Introduce a cut-off level above which we assume survival:
predic <- ifelse(predicScore > 0.7, 1, 0)

# The confusion matrix is one line, the headings 2:
confusion_matrix <- table(predic, t2$Survived)
rownames(confusion_matrix) <- c("predicted_death",
                                "predicted_survival")
colnames(confusion_matrix) <- c("observed_death",
                                "observed_survival")

# Display the result:                                
print(confusion_matrix)
# ---------------------------------------------------------------------


library(ROCR)
# Re-use the model m and the dataset t2:
pred <- prediction(predict(m, type = "response"), t2$Survived)

# Visualize the ROC curve:
plot(performance(pred, "tpr", "fpr"), col="blue", lwd = 3)
abline(0, 1, lty = 2)
# ---------------------------------------------------------------------


S4_perf <- performance(pred, "tpr", "fpr")
df <- data.frame(
   x = S4_perf@x.values,
   y = S4_perf@y.values,
   a = S4_perf@alpha.values
   )
colnames(df) <- c(S4_perf@x.name, S4_perf@y.name, S4_perf@alpha.name)
head(df)
# ---------------------------------------------------------------------


library(ggplot2)
p <- ggplot(data=df, 
            aes(x = `False positive rate`, y = `True positive rate`)) +
            geom_line(lwd=2, col='blue')  + 
            # The next lines add the shading:
            aes(x = `False positive rate`, ymin = 0, 
                ymax = `True positive rate`) + 
            geom_ribbon(, alpha=.5)
p
# ---------------------------------------------------------------------


# Plotting the accuracy (in function of the cut-off)
plot(performance(pred, "acc"), col="blue", lwd = 3)
# ---------------------------------------------------------------------


# Assuming that we have the predictions in the prediction object:
plot(performance(pred, "tpr", "fpr"), col = "blue", lwd = 4)
abline(0, 1, lty = 2, lwd = 3)
x <- c(0.3, 0.1, 0.8)
y <- c(0.5, 0.9, 0.3)
text(x, y, labels = LETTERS[1:3], font = 2, cex = 3)

# Note: instead you can also call the function text() three times:
# text(x = 0.3, y = 0.5, labels = "A", font = 2, cex = 3)
# text(x = 0.1, y = 0.9, labels = "B", font = 2, cex = 3)
# text(x = 0.8, y = 0.3, labels = "C", font = 2, cex = 3)
# ---------------------------------------------------------------------


AUC <- attr(performance(pred, "auc"), "y.values")[[1]]
AUC
# ---------------------------------------------------------------------


paste("the Gini is:",round(2 * AUC - 1, 2))
# ---------------------------------------------------------------------


# using model m and data frame t2:
predicScore <- predict(object=m,type="response")
d0 <- data.frame(
       score = as.vector(predicScore)[t2$Survived == 0],
       true_result = 'not survived')
d1 <- data.frame(
       score = as.vector(predicScore)[t2$Survived == 1],
       true_result = 'survived')
d <- rbind(d0, d1)
d <- d[complete.cases(d),]

cumDf0 <- ecdf(d0$score)
cumDf1 <- ecdf(d1$score)
x <- sort(d$score)
cumD0 <- cumDf0(x)
cumD1 <- cumDf1(x)
diff <- cumD0 - cumD1
y1  <- gdata::first(cumD0[diff == max(diff)])
y2  <- gdata::first(cumD1[diff == max(diff)])
x1  <- x2 <- quantile(d0$score, probs=y1, na.rm=TRUE)

# plot this with ggplot2
p <- ggplot(d, aes(x = score)) +
     stat_ecdf(geom = "step", aes(col = true_result), lwd=2) +
     ggtitle('Cummulative distributions and KS') +
     geom_segment(aes(x = x1, y = y1, xend = x2, yend = y2), 
                  color='navy', lwd = 3) + 
     ggplot2::annotate("text", 
              label = paste0("KS=",round((y1-y2)*100,2),"%"), 
              x = x1 + 0.15, y = y2+(y1-y2)/2, color = "navy")
p
# ---------------------------------------------------------------------


pred <- prediction(predict(m,type="response"), t2$Survived)
ks.test(attr(pred,"predictions")[[1]], 
        t2$Survived,
        alternative = 'greater')
# ---------------------------------------------------------------------


perf <- performance(pred, "tpr", "fpr")
ks <- max(attr(perf,'y.values')[[1]] - attr(perf,'x.values')[[1]])
ks

# Note: the following line yields the same outcome
ks <- max(perf@y.values[[1]] - perf@x.values[[1]])
ks
# ---------------------------------------------------------------------


pred <- prediction(predict(m,type="response"), t2$Survived)
perf <- performance(pred, "tpr", "fpr")
plot(perf, main = paste0(' KS is',round(ks*100,1),'%'),
     lwd = 4, col = 'red')
lines(x = c(0,1),y=c(0,1), col='blue')

# The KS line:
diff <- perf@y.values[[1]] - perf@x.values[[1]]
xVal <- attr(perf,'x.values')[[1]][diff == max(diff)]
yVal <- attr(perf,'y.values')[[1]][diff == max(diff)]
lines(x = c(xVal, xVal), y = c(xVal, yVal), 
      col = 'khaki4', lwd=8)
# ---------------------------------------------------------------------


# First, we need a logistic regression. 
# We use the same as before (and repeat the code here):
library(titanic)
t  <- titanic_train
t2 <- t[complete.cases(t), ]
m  <- glm(data    = t, 
         formula  = Survived ~ Pclass + Sex + Pclass*Sex + Age + SibSp, 
         family   = binomial)
library(ROCR)
pred <- prediction(predict(m, type = "response", newdat = t2), 
                   t2$Survived)
perf <- performance(pred, "tpr", "fpr")
# ---------------------------------------------------------------------


# get_best_cutoff
# Finds a cutof for the score so that sensitivity and specificity 
# are optimal.
# Arguments
#   fpr    -- numeric vector -- false positive rate
#   tpr    -- numeric vector -- true positive rate
#   cutoff -- numeric vector -- the associated cutoff values
# Returns:
#   the cutoff value (numeric)
get_best_cutoff <- function(fpr, tpr, cutoff){
        cst <- (fpr - 0)^2 + (tpr - 1)^2
        idx = which(cst == min(cst))
        c(sensitivity = tpr[[idx]], 
          specificity = 1 - fpr[[idx]], 
          cutoff = cutoff[[idx]])
    }
    
# opt_cut_off 
# Wrapper for get_best_cutoff. Finds a cutof for the score so that 
# sensitivity and specificity are optimal.
# Arguments:
#    perf -- performance object (ROCR package)
#    pred -- prediction object (ROCR package)
# Returns:
#   The optimal cutoff value (numeric)
opt_cut_off = function(perf, pred){
    mapply(FUN=get_best_cutoff, 
           perf@x.values, 
           perf@y.values, 
           pred@cutoffs)
   }
# ---------------------------------------------------------------------


opt_cut_off(perf, pred)
# ---------------------------------------------------------------------


# We introduce cost.fp to be understood as a the cost of a 
# false positive, expressed as a multiple of the cost of a 
# false negative.

# get_best_cutoff
# Finds a cutof for the score so that sensitivity and specificity 
# are optimal.
# Arguments
#   fpr     -- numeric vector -- false positive rate
#   tpr     -- numeric vector -- true positive rate
#   cutoff  -- numeric vector -- the associated cutoff values
#   cost.fp -- numeric        -- cost of false positive divided 
#                                by the cost of a false negative
#                                (default = 1)
# Returns:
#   the cutoff value (numeric)
get_best_cutoff <- function(fpr, tpr, cutoff, cost.fp = 1){
        cst <- (cost.fp * fpr - 0)^2 + (tpr - 1)^2
        idx = which(cst == min(cst))
        c(sensitivity = tpr[[idx]], 
          specificity = 1 - fpr[[idx]], 
          cutoff = cutoff[[idx]])
    }
# ---------------------------------------------------------------------


# opt_cut_off 
# Wrapper for get_best_cutoff. Finds a cutof for the score so that 
# sensitivity and specificity are optimal.
# Arguments:
#    perf    -- performance object (ROCR package)
#    pred    -- prediction object (ROCR package)
#    cost.fp -- numeric -- cost of false positive divided by the
#                          cost of a false negative (default = 1)
# Returns:
#   The optimal cutoff value (numeric)
opt_cut_off = function(perf, pred, cost.fp = 1){
    mapply(FUN=get_best_cutoff, 
           perf@x.values, 
           perf@y.values, 
           pred@cutoffs,
	   cost.fp)
   }
# ---------------------------------------------------------------------


# Test the function:
opt_cut_off(perf, pred, cost.fp = 5)
# ---------------------------------------------------------------------


# e.g. cost.fp = 1 x cost.fn
perf_cst1 <- performance(pred, "cost", cost.fp = 1)
str(perf_cst1) # the cost is in the y-values

# the optimal cut-off is then the same as in previous code sample
pred@cutoffs[[1]][which.min(perf_cst1@y.values[[1]])]

# e.g. cost.fp = 5 x cost.fn
perf_cst2 <- performance(pred, "cost", cost.fp = 5)

# the optimal cut-off is now:
pred@cutoffs[[1]][which.min(perf_cst2@y.values[[1]])]
# ---------------------------------------------------------------------


par(mfrow=c(2,1))
plot(perf_cst1, lwd=2, col='navy', main='(a) cost(FP) = cost(FN)')
plot(perf_cst2, lwd=2, col='navy', main='(b) cost(FP) = 5 x cost(FN)')
par(mfrow=c(1,1))
# ---------------------------------------------------------------------


# Draw Gini, deviance, and misclassification functions

# Define the functions:
gini <- function(x) 2 * x * (1-x)
entr <- function(x) (-x*log(x) - (1-x)*log(1-x))/log(2) / 2
misc <- function(x) {1 - pmax(x,1-x)}

# Plot the curves:
curve(gini, 0, 1, ylim = c(0,0.5), col = "red", 
      xlab="p", ylab = "Impurity measure", type = "l", lwd = 6)
curve(entr, 0, 1, add = TRUE, col = "black", lwd = 6)
curve(misc, 0, 1, add = TRUE, col = "blue", type = "l", 
      lty = 2, lwd = 6)

# Add the text:
text(0.85, 0.4,  "Gini index",                 col = "red",   font = 2)
text(0.83, 0.49, "Deviance or cross-entropy",  col = "black", font = 2)
text(0.5,  0.3,  "Misclassification index",    col = "blue",  font = 2)
# ---------------------------------------------------------------------


parms = list(prior = c(0.6,0.4))
# ---------------------------------------------------------------------


## example of a regression tree with rpart on the dataset of the Titanic
##
library(rpart)
titanic <- read.csv("data/titanic3.csv")
frm     <- survived ~ pclass + sex + sibsp + parch + embarked + age
t0      <- rpart(frm, data=titanic, na.action = na.rpart, 
  method="class",
  parms = list(prior = c(0.6,0.4)),
  #weights=c(...), # each observation (row) can be weighted
  control = rpart.control(
  minsplit       = 50,  # minimum nbr. of observations required for split
  minbucket      = 20,  # minimum nbr. of observations in a terminal node
  cp             = 0.001,# complexity parameter set to a small value 
                        # this will grow a large (over-fit) tree
  maxcompete     = 4,   # nbr. of competitor splits retained in output
  maxsurrogate   = 5,   # nbr. of surrogate splits retained in output
  usesurrogate   = 2,   # how to use surrogates in the splitting process
  xval           = 7,   # nbr. of cross validations
  surrogatestyle = 0,   # controls the selection of a best surrogate
  maxdepth       = 6)   # maximum depth of any node of the final tree
  )
# ---------------------------------------------------------------------


# Show details about the tree t0:
printcp(t0)             

# Plot the error in function of the complexity parameter
plotcp(t0)              

# print(t0) # to avoid too long output we commented this out
# summary(t0)

# Plot the original decisions tree
plot(t0)


text(t0)

# Prune the tree:
t1 <- prune(t0, cp=0.01)
plot(t1); text(t1)
# ---------------------------------------------------------------------


# plot the tree with rpart.plot
library(rpart.plot)
prp(t0, type = 5, extra = 8, box.palette = "auto",
    yesno = 1, yes.text="survived",no.text="dead"
    )
# ---------------------------------------------------------------------


# Example of a regression tree with rpart on the dataset mtcars

# The libraries should be loaded by now:
library(rpart); library(MASS); library (rpart.plot)

# Fit the tree:
t <- rpart(mpg ~ cyl + disp + hp + drat + wt + qsec + am + gear,
 data=mtcars, na.action = na.rpart, 
 method     = "anova",
 control    = rpart.control(
   minsplit       = 10,  # minimum nbr. of observations required for split
   minbucket      = 20/3,# minimum nbr. of observations in a terminal node
                         # the default = minsplit/3
   cp             = 0.01,# complexity parameter set to a very small value 
                         # his will grow a large (over-fit) tree
   maxcompete     = 4,   # nbr. of competitor splits retained in output
   maxsurrogate   = 5,   # nbr. of surrogate splits retained in output
   usesurrogate   = 2,   # how to use surrogates in the splitting process
   xval           = 7,   # nbr. of cross validations
   surrogatestyle = 0,   # controls the selection of a best surrogate
   maxdepth       = 30   # maximum depth of any node of the final tree
   )
 )
# ---------------------------------------------------------------------


# Print the tree:
print(t)
summary(t)
# ---------------------------------------------------------------------


# plot(t) ; text(t)  # This would produce the standard plot from rpart.
# Instead we use:
prp(t, type = 5, extra = 1, box.palette = "Blues", digits = 4,
    shadow.col = 'darkgray', branch = 0.5)
# ---------------------------------------------------------------------


# Prune the tree:
t1 <- prune(t, cp = 0.05)

# Finally, plot the pruned tree:
prp(t1, type = 5, extra = 1, box.palette = "Reds", digits = 4,
    shadow.col = 'darkgray', branch = 0.5)
# ---------------------------------------------------------------------


# We use the function stats::predict()
predicPerc <- predict(object=t0, newdata=titanic)

# predicPerc is now a matrix with probabilities: in column 1 the 
# probability not to survive, in column 2 the probability to survive:
head(predicPerc)

# This is not what we need. We need to specify that it is a
# classification tree. Here we correct this:
predic <- predict(object=t0, newdata=titanic, type="class")
  # vector with only the fitted class as prediction
head(predic)
# ---------------------------------------------------------------------


# The confusion matrix:
confusion_matrix <- table(predic, titanic$survived)
rownames(confusion_matrix) <- c("predicted_death",
                                "predicted_survival")
colnames(confusion_matrix) <- c("observed_death",
                                "observed_survival")
confusion_matrix

# As a precentage:
confusion_matrixPerc <- sweep(confusion_matrix, 2, 
                       margin.table(confusion_matrix,2),"/")

# Here is the confusion matrix:
round(confusion_matrixPerc,2)
# ---------------------------------------------------------------------


library(ROCR)
pred <- prediction(predict(t0, type = "prob")[,2], 
                                 titanic$survived)

# Visualize the ROC curve:
plot(performance(pred, "tpr", "fpr"), col = "blue", lwd = 3)
abline(0, 1, lty = 2)
# ---------------------------------------------------------------------


plot(performance(pred, "acc"), col = "blue", lwd = 3)
abline(1, 0, lty = 2)
# ---------------------------------------------------------------------


# AUC:
AUC <- attr(performance(pred, "auc"), "y.values")[[1]]
AUC

# GINI:
2 * AUC - 1

# KS:
perf <- performance(pred, "tpr", "fpr")
max(attr(perf,'y.values')[[1]]-attr(perf,'x.values')[[1]])
# ---------------------------------------------------------------------


library(randomForest)
# ---------------------------------------------------------------------


head(mtcars)
mtcars$l <- NULL  # remove our variable
frm      <- mpg ~ cyl + disp + hp + drat + wt + qsec + am + gear
set.seed(1879)

# Fit the random forest:
forestCars   = randomForest(frm, data = mtcars)

# Show an overview:
print(forestCars)

# Plot the random forest overview:
plot(forestCars)

# Show the summary of fit:
summary(forestCars)

# visualization of the RF:
getTree(forestCars, 1, labelVar=TRUE)

# Show the purity of the nodes:
imp <- importance(forestCars)
imp

# This impurity overview can also be plotted:
plot( imp, lty=2, pch=16)
lines(imp)

# Below we print the partial dependence on each variable.
# We group the plots per 3, to save some space.
impvar = rownames(imp)[order(imp[, 1], decreasing=TRUE)]
op     = par(mfrow=c(1, 3))
for (i in seq_along(impvar)) {
    partialPlot(forestCars, mtcars, impvar[i], xlab=impvar[i],
    main=paste("Partial Dependence on", impvar[i]))
  }
# ---------------------------------------------------------------------


#install.packages("neuralnet")
library(neuralnet)
nn0 <- neuralnet(mpg ~ wt + qsec + am + hp + disp + cyl + drat + gear + carb,
                 data = mtcars, hidden = c(0),
                 linear.output = TRUE)
plot(nn0, rep = "best", intercept = TRUE, information = FALSE, 
     show.weights = TRUE);
# ---------------------------------------------------------------------


neuralnet(formula, data, hidden = 1, stepmax = 1e+05
          linear.output = TRUE)
# ---------------------------------------------------------------------


#install.packages("neuralnet") # Do only once.

# Load the library neuralnet:
library(neuralnet)

# Fit the aNN with 2 hidden layers that have resp. 3 and 2 neurons:
# (neuralnet does not accept a formula wit a dot as in 'y ~ .' )
nn1 <- neuralnet(mpg ~ wt + qsec + am + hp + disp + cyl + drat + 
                       gear + carb,
                 data = mtcars, hidden = c(3,2),
                 linear.output = TRUE)
# ---------------------------------------------------------------------


plot(nn1, rep = "best", information = FALSE);
# ---------------------------------------------------------------------


# Get the data about crimes in Boston:
library(MASS)
d <- Boston
# ---------------------------------------------------------------------


# Inspect if there is missing data:
apply(d, 2, function(x) sum(is.na(x)))
# There are no missing values.
# ---------------------------------------------------------------------


set.seed(1877) # set the seed for the random generator
idx.train <- sample(1:nrow(d), round(0.75 * nrow(d)))
d.train   <- d[idx.train,]
d.test    <- d[-idx.train,]
# ---------------------------------------------------------------------


# Fit the linear model, no default for family, so use 'gaussian':
lm.fit <- glm(medv ~ ., data = d.train)
summary(lm.fit)

# Make predictions:
pr.lm  <- predict(lm.fit,d.test)

# Calculate the MSE:
MSE.lm <- sum((pr.lm - d.test$medv)^2)/nrow(d.test)
# ---------------------------------------------------------------------


# Store the maxima and minima:
d.maxs <- apply(d, 2, max) 
d.mins <- apply(d, 2, min)

# Rescale the data:
d.sc <- as.data.frame(scale(d, center = d.mins, 
                               scale  = d.maxs - d.mins))

# Split the data in training and testing set:
d.train.sc <- d.sc[idx.train,]
d.test.sc  <- d.sc[-idx.train,]
# ---------------------------------------------------------------------


library(neuralnet)

# Since the shorthand notation y~. does not work in the 
# neuralnet() function we have to replicate it:
nm  <- names(d.train.sc)
frm <- as.formula(paste("medv ~", paste(nm[!nm %in% "medv"], collapse = " + ")))

nn2 <- neuralnet(frm, data = d.train.sc, hidden = c(7,5,5),
                 linear.output = T)
# ---------------------------------------------------------------------


plot(nn2, rep = "best", information = FALSE, 
     show.weights = FALSE)
# ---------------------------------------------------------------------


# Our independent variable 'medv' is the 14th column, so:
pr.nn2 <- compute(nn2,d.test.sc[,1:13]) 
                                   
# Rescale back to original span:
pr.nn2 <- pr.nn2$net.result*(max(d$medv)-min(d$medv))+min(d$medv)
test.r <- (d.test.sc$medv)*(max(d$medv)-min(d$medv))+min(d$medv)

# Calculate the MSE:
MSE.nn2 <- sum((test.r - pr.nn2)^2)/nrow(d.test.sc)
print(paste(MSE.lm,MSE.nn2))
# ---------------------------------------------------------------------


par(mfrow=c(1,2))

plot(d.test$medv, pr.nn2, col='red',
     main='Observed vs predicted NN',
     pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright', legend='NN', pch=18, col='red', bty='n')

plot(d.test$medv,pr.lm,col='blue',
     main='Observed vs predicted lm',
     pch=18, cex=0.7)
abline(0,1,lwd=2)
legend('bottomright', legend='LM', pch=18,col='blue', bty='n', 
       cex=.95)
# ---------------------------------------------------------------------


plot  (d.test$medv,pr.nn2,col='red',
       main='Observed vs predicted NN',
       pch=18,cex=0.7)
points(d.test$medv,pr.lm,col='blue',pch=18,cex=0.7)
abline(0,1,lwd=2)
legend('bottomright',legend=c('NN','LM'),pch=18,
       col=c('red','blue'))
# ---------------------------------------------------------------------


library(boot)
set.seed(1875)
lm.fit <- glm(medv ~ ., data = d)

# The estimate of prediction error is now here:
cv.glm(d, lm.fit, K = 10)$delta[1] 
# ---------------------------------------------------------------------


# Reminders:
d   <- Boston
nm  <- names(d)
frm <- as.formula(paste("medv ~", paste(nm[!nm %in% "medv"], 
                        collapse = " + ")))
# Store the maxima and minima:
d.maxs <- apply(d, 2, max) 
d.mins <- apply(d, 2, min)

# Rescale the data:
d.sc <- as.data.frame(scale(d, center = d.mins, 
                               scale  = d.maxs - d.mins))

# Set parameters:
set.seed(1873)
cv.error <- NULL  # Initiate to append later
k        <- 10    # The number of repetitions

# This code might be slow, so you can add a progress bar as follows:
#library(plyr) 
#pbar <- create_progress_bar('text')
#pbar$init(k)

# In k-fold cross validation, we must take care to select each
# observation just once in the testing set. This is made easy 
# with modelr:
library(modelr)
kFoldXval <- crossv_kfold(data = d.sc, k = 10, id = '.id')

# Do the k-fold cross validation:
for(i in 1:k){
    # <see digression below>
    train.cv  <- kFoldXval$train[i]
    test.cv   <- kFoldXval$test[i]
    test.cv.df <- as.data.frame(test.cv)
    
    # Rebuild the formula (names are changed each run):
    nmKfold <- paste0('X', i, '.', nm)
    medvKfld <- paste0('X', i, '.medv')
    frmKfold <- as.formula(paste(medvKfld, "~", 
                             paste(nmKfold[!nmKfold %in% medvKfld], 
                             collapse = " + ")
		             )
			  )

    # Fit the NN:
    nn2       <- neuralnet(frmKfold, data = train.cv, 
                           hidden = c(7, 5, 5),
                           linear.output=TRUE
			   )

    # The explaining variables are in the first 13 rows, so:
    pr.nn2   <- compute(nn2, test.cv.df[,1:13])   
          
    pr.nn2   <- pr.nn2$net.result * (max(d$medv) - min(d$medv)) + 
                min(d$medv)
    test.cv.df.r <- test.cv.df[[medvKfld]] * 
                    (max(d$medv) - min(d$medv)) + min(d$medv)
    cv.error[i] <- sum((test.cv.df.r - pr.nn2)^2)/nrow(test.cv.df)    
    #pbar$step()  #uncomment to see the progress bar
}
# ---------------------------------------------------------------------


index    <- sample(1:nrow(d),round(0.9*nrow(d)))
train.cv <- d.sc[index,]
test.cv  <- d.sc[-index,]
# ---------------------------------------------------------------------


# <see digression below>
# ---------------------------------------------------------------------


# Show the mean of the MSE:
mean(cv.error)
cv.error

# Show the boxplot:
boxplot(cv.error,xlab='MSE',col='gray',
        border='blue',names='CV error (MSE)',
        main='Cross Validation error (MSE) for the ANN',
        horizontal=TRUE)
# ---------------------------------------------------------------------


svm(formula, data, subset, na.action = na.omit, scale = TRUE, 
    type = NULL, kernel = 'radial', degree = 3, 
    gamma = if (is.vector(x)) 1 else 1 / ncol(x), coef0 = 0, 
    cost = 1, nu = 0.5,  class.weights = NULL, cachesize = 40, 
    tolerance = 0.001, epsilon = 0.1, shrinking = TRUE, 
    cross = 0, probability = FALSE, fitted = TRUE, ...)
# ---------------------------------------------------------------------


library(e1071)
svmCars1 <- svm(cyl ~ ., data = mtcars)
summary(svmCars1)
# ---------------------------------------------------------------------


# split mtcars in two subsets (not necessary but easier later):
x <- subset(mtcars, select = -cyl)
y <- mtcars$cyl

# fit the model again as a classification model:
svmCars2 <- svm(cyl ~ ., data = mtcars, type = 'C-classification')

# create predictions
pred <- predict(svmCars2, x)

# show the confusion matrix:
table(pred, y)
# ---------------------------------------------------------------------


svmTune <- tune(svm, train.x=x, train.y=y, kernel = "radial",
                ranges =  list(cost = 10^(-1:2), gamma = c(.5, 1, 2)))

print(svmTune)
# ---------------------------------------------------------------------


library(ggplot2)
library(ggrepel)  # provides geom_label_repel()
ggplot(mtcars, aes(wt, mpg, color = factor(cyl))) + 
       geom_point(size = 5) + 
       geom_label_repel(aes(label = rownames(mtcars)),
                  box.padding   = 0.2, 
                  point.padding = 0.25,
                  segment.color = 'grey60')
# ---------------------------------------------------------------------


ggplot(mtcars, aes(wt, mpg, color = factor(cyl))) + 
       geom_point(size = 5) + 
       geom_text(aes(label = rownames(mtcars)), 
                 hjust = -0.2, vjust = -0.2)
# ---------------------------------------------------------------------


# normalize weight and mpg
d <- data.frame(matrix(NA, nrow = nrow(mtcars), ncol = 1))
d <- d[,-1]  # d is an empty data frame with 32 rows

rngMpg <- range(mtcars$mpg, na.rm = TRUE)
rngWt  <- range(mtcars$wt,  na.rm = TRUE)

d$mpg_n <- (mtcars$mpg - rngMpg[1]) / rngMpg[2]
d$wt_n  <- (mtcars$wt  - rngWt[1])  / rngWt[2]

# Here is the k-means clustering itself. 
# Note the nstart parameter (the number of random starting sets)
carCluster <- kmeans(d, 3, nstart = 15)

print(carCluster)
# ---------------------------------------------------------------------


table(carCluster$cluster, mtcars$cyl)
# Note that the rows are the clusters (1, 2, 3) and the number of 
# cylinders are the columns (4, 6, 8).
# ---------------------------------------------------------------------


# Additionally, we customize colours and legend title. 
# First, we need color names:
my_colors <- if_else(carCluster$cluster == 1, "darkolivegreen3",
                if_else(carCluster$cluster == 2, "coral", "cyan3"))

# We already loaded the libraries as follows:
#library(ggplot2); library(ggrepel)

# Now we can create the plot:
ggplot(mtcars, aes(wt, mpg, fill = factor(carCluster$cluster))) + 
       geom_point(size = 5, colour = my_colors) + 
       scale_fill_manual('Clusters', 
                   values = c("darkolivegreen3","coral", "cyan3"))+
       geom_label_repel(aes(label = rownames(mtcars)),
                  box.padding   = 0.2, 
                  point.padding = 0.25,
                  segment.color = 'grey60') 

# ---------------------------------------------------------------------


# Normalize the whole mtcars dataset:
d <- data.frame(matrix(NA, nrow = nrow(mtcars), ncol = 1))
d <- d[,-1]  # d is an empty data frame with 32 rows
for (k in 1:ncol(mtcars)) {
  rng <- range(mtcars[, k], na.rm = TRUE)
  d[, k]  <- (mtcars[, k]  - rng[1])  / rng[2]
  }
colnames(d) <- colnames(mtcars)
rownames(d) <- rownames(mtcars)

# The PCA analysis:
pca1 <- prcomp(d) 
summary(pca1)

# Note also:
class(pca1)
# ---------------------------------------------------------------------


# Plot for the prcomp object shows the variance explained by each PC
plot(pca1, type = 'l')

# biplot shows a projection in the 2D plane (PC1, PC2)
biplot(pca1)
# ---------------------------------------------------------------------


# Same plot with ggplot2:
library(ggplot2)
library(ggfortify)
library(cluster)

autoplot(pca1, data = d, label = TRUE, shape = FALSE, colour = 'mpg',
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = FALSE, loadings.label.size = 3
         )
# ---------------------------------------------------------------------


# Make the clustering:
carCluster <- kmeans(d, 4, nstart = 10)

d_cluster <- cbind(d, cluster = factor(carCluster$cluster))

# Load the graph librarires:
library(ggplot2)
library(ggrepel)

# Plot the results:
autoplot(pca1, data = d_cluster, label = FALSE, shape = 18, size = 5, 
         alpha = 0.6, colour = 'cluster',
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 5) + 
       geom_label_repel(aes(label = rownames(mtcars)),
                  box.padding   = 0.2, 
                  point.padding = 0.25,
                  segment.color = 'grey60')
# ---------------------------------------------------------------------


my_colors <- if_else(carCluster$cluster == 1,"darkolivegreen3",
                     if_else(carCluster$cluster == 2, "coral", 
		     if_else(carCluster$cluster == 3, "cyan3", 
		             "gray80")))

ggplot(pca1$x,
       aes(x=PC1,y=PC2, fill = factor(carCluster$cluster))) +
       geom_point(size = 5, alpha = 0.7, colour = my_colors)+ 
       scale_fill_manual('Clusters', 
                         values =c("darkolivegreen3","coral", 
			 "cyan3", "gray80")) +
       geom_label_repel(aes(label = rownames(mtcars)),
                  box.padding   = 0.2, 
                  point.padding = 0.25,
                  segment.color = 'grey60')
# ---------------------------------------------------------------------


# We use the same data as defined in previous section: d is the normalised 
# dataset mtcars.
pca1 <- prcomp(d)   # unchanged

# Extract the first 3 PCs
PCs <- data.frame(pca1$x[,1:3])

# Then use this in the kmeans function for example.
carCluster <- kmeans(PCs, 4, nstart=10, iter.max=1000)
# ---------------------------------------------------------------------


# Reminder of previous code:
d <- data.frame(matrix(NA, nrow = nrow(mtcars), ncol = 1))
d <- d[,-1]  # d is an empty data frame with 32 rows
for (k in 1:ncol(mtcars)) {
  rng     <- range(mtcars[, k], na.rm = TRUE)
  d[, k]  <- (mtcars[, k]  - rng[1])  / rng[2]
  }
colnames(d) <- colnames(mtcars)
rownames(d) <- rownames(mtcars)
pca1 <- prcomp(d)               # runs the PCA
PCs <- data.frame(pca1$x[,1:3]) # extracts the first 3 PCs
# Now, PCs holds the three major PCs.

# -- New code below:
# Plot those three PCs:
plot(PCs, col = carCluster$clust, pch = 16, cex = 3)
# ---------------------------------------------------------------------


library(plot3D)
scatter3D(x = PCs$PC1, y = PCs$PC2, z = PCs$PC3, 
   phi = 45, theta = 45,
   pch = 16, cex = 1.5, bty = "f",
   clab = "cluster", 
   colvar = as.integer(carCluster$cluster), 
   col = c("darkolivegreen3", "coral", "cyan3", "gray"),
   colkey = list(at = c(1, 2, 3, 4),
          addlines = TRUE, length = 0.5, width = 0.5,
          labels = c("1", "2", "3", "4"))
   )
text3D(x = PCs$PC1, y = PCs$PC2, z = PCs$PC3,  labels = rownames(d),
        add = TRUE, colkey = FALSE, cex = 1.2)
# ---------------------------------------------------------------------


library(plotly)
plot_ly(x = PCs$PC1, y = PCs$PC2, z = PCs$PC3,  
        type="scatter3d", mode="markers", 
	color=factor(carCluster$cluster))
# ---------------------------------------------------------------------


library(tidyverse)  # provides if_else
library(ggplot2)    # 2D plotting 
library(ggfortify)
library(cluster)    # provides fanny (the fuzzy clustering)
library(ggrepel)    # provides geom_label_repel (de-clutter labels)


carCluster <- fanny(d, 4)
my_colors <- if_else(carCluster$cluster == 1, "coral",
               if_else(carCluster$cluster == 2, "darkolivegreen3", 
	       if_else(carCluster$cluster == 3, "cyan3", 
	         "darkorchid1")))

# Autoplot with visualization of 4 clusters
autoplot(carCluster, label=FALSE, frame=TRUE,  frame.type='norm', 
         shape=16,
         loadings=TRUE,  loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 5,
	 loadings.label.vjust = 1.2, loadings.label.hjust = 1.3) + 
       geom_point(size = 5, alpha = 0.7, colour = my_colors) + 
       geom_label_repel(aes(label = rownames(mtcars)),
                  box.padding   = 0.2, 
                  point.padding = 0.25,
                  segment.color = 'grey40') + 
		  theme_classic()
# ---------------------------------------------------------------------


# Compute hierarchical clustering
library(tidyverse)
cars_hc <- mtcars                      %>%
           scale                       %>% # scale the data
           dist(method = "euclidean")  %>% # dissimilarity matrix
           hclust(method = "ward.D2")      # hierachical clustering

plot(cars_hc)
# ---------------------------------------------------------------------


library(class)
knn(train, test, cl, k = 1, l = 0, prob = FALSE, use.all = TRUE)
# ---------------------------------------------------------------------


library(tidyverse)
library(modelr)
# ---------------------------------------------------------------------


d   <- mtcars
lm1 <- lm(mpg ~ wt + cyl, data = d)
# ---------------------------------------------------------------------


add_predictions(data, model, var = "pred", type = NULL)
# ---------------------------------------------------------------------


library(modelr)

# Use the data defined above:
d1 <- d %>% add_predictions(lm1)

# d1 has now an extra column "pred"
head(d1)
# ---------------------------------------------------------------------


add_residuals(data, model, var = "resid")
# ---------------------------------------------------------------------


d2 <- d1 %>% add_residuals(lm1)

# d2 has now an extra column "resid"
head(d2)
# ---------------------------------------------------------------------


bootstrap(data, n, id = ".id")
# ---------------------------------------------------------------------


set.seed(1872)   # make sure that results can be replicated
library(modelr)  # provides bootstrap
library(purrr)   # provides map, map_df, etc.
library(ggplot2) # provides ggplot
d    <- mtcars
boot <- bootstrap(d, 10)

# Now, we can leverage tidyverse functions such as map to create 
# multiple models on the 10 datasets
models <- map(boot$strap, ~ lm(mpg ~ wt + cyl, data = .))

# The function tidy of broom (also tidyverse) allows to create a
# dataset based on the list of models. Broom is not loaded, because
# it also provides a function bootstrap().
tidied <- map_df(models, broom::tidy, .id = "id")
# ---------------------------------------------------------------------


# Visualize the results with ggplot2:
p <- ggplot(tidied, aes(estimate)) + 
     geom_histogram(bins = 5, col = 'red', fill='khaki3', 
                    alpha = 0.5) + 
     ylab('Count') + 
     xlab('Estimate of the coefficient in the plot-title') +
     facet_grid(. ~ term, scales = "free")
p
# ---------------------------------------------------------------------


# load modelr:
library(modelr)

# Fit a model:
lm1 <- lm(mpg ~ wt + qsec + am, data = mtcars)

# MSE (mean square error):
mse(lm1, mtcars)

# RMSE (root mean square error):
rmse(lm1, mtcars)

# MAD (mean absolute error):
mae(lm1, mtcars)

# Quantiles of absolute error:
qae(lm1, mtcars)

# R-square (variance of predictions divided by the variance of the 
# response variable):
rsquare(lm1, mtcars)
# ---------------------------------------------------------------------


set.seed(1871)

# Split the data:
rs  <- mtcars  %>%
       resample_partition(c(train = 0.6, test = 0.4))

# Train the model on the training set:
lm2 <- lm(mpg ~ wt + qsec + am, data = rs$train)

# Compare the RMSE on the training set with the testing set:
rmse(lm2, rs$train); rmse(lm2, rs$test)

# Note that this can alos be done with the pipe operator:
lm2 %>% rmse(rs$train)
lm2 %>% rmse(rs$test)
# ---------------------------------------------------------------------


# Fit the model:
lm1 <- lm(mpg ~ wt + qsec + am, data = mtcars)

# Add the predictions and residuals:
df <- mtcars               %>% 
      add_predictions(lm1) %>%
      add_residuals(lm1)

# The predictions are now available in $pred:
head(df$pred)

# The residuals are now available in $resid:
head(df$resid)

# It is now easy to do something with those predictions and 
# residuals, e.g. the following 3 lines all do the same:
sum((df$pred - df$mpg)^2) / nrow(mtcars)
sum((df$resid)^2) / nrow(mtcars)
mse(lm1, mtcars)  # Check if this yields the same
# ---------------------------------------------------------------------


d <- data_grid(mtcars, wt = seq_range(wt, 10), qsec, am) %>% 
     add_predictions(lm1)
plot(d$wt, d$pred)
# ---------------------------------------------------------------------


# Create the sample:
SP500_sample <- sample(SP500,size=100)

# Change plotting to 4 plots in one output:
par(mfrow=c(2,2))

# The histogram of the complete dataset:
hist(SP500,main="(a) Histogram of all data",fr=FALSE,
     breaks=c(-9:5),ylim=c(0,0.4))

# The histogram of the sample:
hist(SP500_sample,main="(b) Histogram of the sample",
     fr=FALSE,breaks=c(-9:5),ylim=c(0,0.4))

# The boxplot of the complete dataset:
boxplot(SP500,main="(c) Boxplot of all data",ylim=c(-9,5))

# The boxplot of the complete sample:
boxplot(SP500_sample,main="(c) Boxplot of the sample",
        ylim=c(-9,5))

# Reset the plot parameters:
par(mfrow=c(1,1))
# ---------------------------------------------------------------------


mean(SP500)
mean(SP500_sample)
sd(SP500)
sd(SP500_sample)
# ---------------------------------------------------------------------


# Bootstrap generates a number of re-ordered datasets
boot <- bootstrap(mtcars, 3)
# The datasets are now in boot$strap[[n]]
# with n between 1 and 3

# e.g. the 3rd set is addressed as follows:
class(boot$strap[[3]])
nrow(boot$strap[[3]])
mean(as.data.frame(boot$strap[[3]])$mpg)

# It is also possible to coerce the selections into a data-frame:
df <- as.data.frame(boot$strap[[3]])
class(df)
# ---------------------------------------------------------------------


set.seed(1871)
library(purrr)  # to use the function map()
boot <- bootstrap(mtcars, 150)
     
lmodels <- map(boot$strap, ~ lm(mpg ~ wt + hp + am:vs, data = .))

# The function tidy of broom turns a model object in a tibble:
df_mods <- map_df(lmodels, broom::tidy, .id = "id")

# Create the plots of histograms of estimates for the coefficients:
par(mfrow=c(2,2))
hist(subset(df_mods, term == "wt")$estimate, col="khaki3",
     main = '(a) wt', xlab = 'estimate for wt')
hist(subset(df_mods, term == "hp")$estimate, col="khaki3",
     main = '(b) hp', xlab = 'estimate for hp')
hist(subset(df_mods, term == "am:vs")$estimate, col="khaki3",
     main = '(c) am:vs', xlab = 'estimate for am:vs')
hist(subset(df_mods, term == "(Intercept)")$estimate, col="khaki3",
     main = '(d) intercept', xlab = 'estimate for the intercept')
par(mfrow=c(1,1))
# ---------------------------------------------------------------------


d <- mtcars                # get data
set.seed(1871)             # set the seed for the random generator
idx.train <- sample(1:nrow(d),round(0.75*nrow(d)))
d.train <- d[idx.train,]   # positive matches for training set
d.test  <- d[-idx.train,]  # the opposite to the testing set
# ---------------------------------------------------------------------


set.seed(1870)
sample_cars <- mtcars %>%
               resample(sample(1:nrow(mtcars),5)) # random 5 cars

# This is a resample object (indexes shown, not data):
sample_cars  

# Turn it into data:
as.data.frame(sample_cars)

# or into a tibble
as_tibble(sample_cars)

# or use the indices to get to the data:
mtcars[as.integer(sample_cars),]
# ---------------------------------------------------------------------


library(modelr)
rs <- mtcars  %>%
      resample_partition(c(train = 0.6, test = 0.4))

# address the datasets with: as.data.frame(rs$train)
#                            as.data.frame(rs$test)

# Check execution:
lapply(rs, nrow)
# ---------------------------------------------------------------------


# 0. Store training and test dataset for further use (optional):
d_train  <- as.data.frame(rs$train)
d_test   <- as.data.frame(rs$test)

# 1. Fit the model on the training dataset:
lm1      <- lm(mpg ~ wt + hp + am:vs, data = rs$train)

# 2. Calculate the desired performance measure (e.g.
# root mean square error (rmse)):
rmse_trn <- lm1 %>% rmse(rs$train)
rmse_tst <- lm1 %>% rmse(rs$test)
print(rmse_trn)
print(rmse_tst)
# ---------------------------------------------------------------------


# 2. Add predictions and residuals:
x_trn  <- add_predictions(d_train, model = lm1) %>% 
          add_residuals(model = lm1)
x_tst  <- add_predictions(d_test,  model = lm1) %>% 
          add_residuals(model = lm1)


# 3. Calculate the desired risk metrics (via the residuals):
RMSE_trn  <- sqrt(sum(x_trn$resid^2) / nrow(d_train))
RMSE_tst  <- sqrt(sum(x_tst$resid^2) / nrow(d_test))
print(RMSE_trn)
print(RMSE_tst)
# ---------------------------------------------------------------------


# Monte Carlo cross validation
cv_mc <- crossv_mc(data = mtcars, # the dataset to split
           n = 50,      # n random partitions train and test
	   test = 0.25, # validation set is 25%
	   id = ".id")  # unique identifier for each model

# Example of use:

# Access the 2nd test dataset:
d <- data.frame(cv_mc$test[2])

# Access mpg in that data frame:
data.frame(cv_mc$test[2])$mpg

# More cryptic notations are possible to obtain the same:
mtcars[cv_mc[[2]][[2]][2]$idx,1]
# ---------------------------------------------------------------------


set.seed(1868)
library(modelr)     # sample functions
library(purrr)      # to use the function map()

cv_mc <- crossv_mc(mtcars, n = 50, test = 0.40)
mods  <- map(cv_mc$train, ~ lm(mpg ~ wt + hp + am:vs, data = .))
RMSE  <- map2_dbl(mods, cv_mc$test, rmse)
hist(RMSE, col="khaki3")
# ---------------------------------------------------------------------


library(magrittr)  # to access the %T>% pipe 
crossv <- mtcars                                          %>% 
          crossv_mc(n = 50, test = 0.40)
RMSE <- crossv                                            %$%
        map(train, ~ lm(mpg ~ wt + hp + am:vs, data = .)) %>%
        map2_dbl(crossv$test, rmse)                      %T>%
        hist(col = "khaki3", main ="Histogram of RMSE", 
	     xlab = "RMSE")
# ---------------------------------------------------------------------


library(modelr)
# k-fold cross validation
cv_k  <- crossv_kfold(data = mtcars, 
           k = 5,      # number of folds
           id = ".id") # unique identifier for each
# ---------------------------------------------------------------------


cv_k$test
# ---------------------------------------------------------------------


set.seed(1868)
library(modelr)
library(magrittr)  # to access the %T>% pipe
crossv <- mtcars                                           %>% 
          crossv_kfold(k = 5)                             
RMSE <- crossv                                             %$%
        map(train, ~ lm(mpg ~ wt + hp + am:vs, data = .))  %>%
        map2_dbl(crossv$test, rmse)                        %T>%
        hist(col = "khaki3", main ="Histogram of RMSE", 
	     xlab = "RMSE")
# ---------------------------------------------------------------------


# Install quantmod:
if(!any(grepl("quantmod", installed.packages()))){
    install.packages("quantmod")}

# Load the library:
library(quantmod)
# ---------------------------------------------------------------------


options("getSymbols.yahoo.warning"=FALSE)
# ---------------------------------------------------------------------


# Download historic data of the Google share price:
getSymbols("GOOG", src = "yahoo")          # get Google's history
getSymbols(c("GS", "GOOG"), src = "yahoo") # to load more than one
# ---------------------------------------------------------------------


setSymbolLookup(HSBC='yahoo',GOOG='yahoo')
setSymbolLookup(DEXJPUS='FRED')
setSymbolLookup(XPTUSD=list(name="XPT/USD",src="oanda"))

# Save the settings in a file:
saveSymbolLookup(file = "qmdata.rda")  
# Use this in new sessions calling:
loadSymbolLookup(file = "qmdata.rda")

# We can also download a list of symbols as follows:
getSymbols(c("HSBC","GOOG","DEXJPUS","XPTUSD")) 
# ---------------------------------------------------------------------


stockList <- stockSymbols()  # get all symbols
nrow(stockList)     # number of symbols
colnames(stockList) # information in this list
# ---------------------------------------------------------------------


getFX("EUR/PLN", from = "2019-01-01")
# ---------------------------------------------------------------------


getSymbols("HSBC",src="yahoo") #get HSBC's data from Yahoo
# ---------------------------------------------------------------------



# 1. The bar chart:
barChart(HSBC)     

# 2. The line chart:
lineChart(HSBC)
# Note: the lineChart is also the default that yields the same as plot(HSBC)

# 3. The candle chart:
candleChart(HSBC, subset='last 1 years',theme="white",
            multi.col=TRUE)
# ---------------------------------------------------------------------


candleChart(HSBC,subset='2018::2018-01')
candleChart(HSBC,subset='last 1 months')
# ---------------------------------------------------------------------


getSymbols(c("HSBC"))
chartSeries(HSBC, subset='last 4 months')
addBBands(n = 20, sd = 2, maType = "SMA", draw = 'bands', 
          on = -1)
# ---------------------------------------------------------------------


myxtsdata["2008-01-01/2010-12-31"]  # between 2 date-stamps

# All data before or after a certain time-stamp:
xtsdata["/2007"]  # from start of data until end of 2007
xtsdata["2009/"]  # from 2009 until the end of the data

# Select the data between different hours:
xtsdata["T07:15/T09:45"]
# ---------------------------------------------------------------------


HSBC['2017']    #returns HSBC's OHLC data for 2017
HSBC['2017-08'] #returns HSBC's OHLC data for August 2017
HSBC['2017-06::2018-01-15'] # from June 2017 to Jan 15 2018

HSBC['::']     # returns all data
HSBC['2017::'] # returns all data in HSBC, from 2017 onward
my.selection <- c('2017-01','2017-03','2017-11')
HSBC[my.selection]
# ---------------------------------------------------------------------


last(HSBC)               # returns the last quotes
last(HSBC,5)             # returns the last 5 quotes
last(HSBC, '6 weeks')    # the last 6 weeks
last(HSBC, '-1 weeks')   # all but the last week
last(HSBC, '6 months')   # the last 6 months
last(HSBC, '3 years')    # the last 3 years

# these functions can also be combined:
last(first(HSBC, '3 weeks'), '5 days')
# ---------------------------------------------------------------------


periodicity(HSBC)
unclass(periodicity(HSBC))
to.weekly(HSBC)
to.monthly(HSBC)
periodicity(to.monthly(HSBC))
ndays(HSBC); nweeks(HSBC); nyears(HSBC)
# ---------------------------------------------------------------------


getFX("USD/EUR")
periodicity(USDEUR)
to.monthly(USDEUR)
periodicity(to.monthly(USDEUR))
# ---------------------------------------------------------------------


endpoints(HSBC,on="years") 

# Find the maximum closing price each year:
apply.yearly(HSBC,FUN=function(x) {max(Cl(x)) } )

# The same thing - only more general:
subHSBC <- HSBC['2012::']
period.apply(subHSBC,endpoints(subHSBC,on='years'), FUN=function(x) {max(Cl(x))} )

# The following line does the same but is faster:
as.numeric(period.max(Cl(subHSBC),endpoints(subHSBC, on='years')))
# ---------------------------------------------------------------------


seriesHi(HSBC)
has.Cl(HSBC)
tail(Cl(HSBC))
# ---------------------------------------------------------------------


Lag(Cl(HSBC))
Lag(Cl(HSBC), c(1, 5, 10)) # One, five and ten period lags
Next(OpCl(HSBC))

# Open to close one, two and three-day lags:
Delt(Op(HSBC),Cl(HSBC),k=1:3)
# ---------------------------------------------------------------------


dailyReturn(HSBC)   
weeklyReturn(HSBC)  
monthlyReturn(HSBC) 
quarterlyReturn(HSBC)
yearlyReturn(HSBC)
allReturns(HSBC)     # all previous returns
# ---------------------------------------------------------------------


# First, we create a quantmod object. 
# At this point, we do not need to load data.

setSymbolLookup(SPY = 'yahoo', VXN = list(name = '^VIX', src = 'yahoo'))

qmModel <- specifyModel(Next(OpCl(SPY)) ~ OpCl(SPY) + Cl(VIX))
head(modelData(qmModel))
# ---------------------------------------------------------------------


getSymbols('HSBC',src='yahoo') #google doesn't carry the adjusted price
lineChart(HSBC)
# ---------------------------------------------------------------------


HSBC.tmp   <- HSBC["2010/"]          #see: subsetting for xts objects    
# ---------------------------------------------------------------------


# use 70% of the data to train the model:
n          <- floor(nrow(HSBC.tmp) * 0.7)
HSBC.train <- HSBC.tmp[1:n]               # training data
HSBC.test  <- HSBC[(n+1):nrow(HSBC.tmp)]  # test-data
# head(HSBC.train)
# ---------------------------------------------------------------------


# Making sure that whenever we re-run this the latest data is pulled in:
m.qm.tr <- specifyModel(Next(Op(HSBC.train)) ~ Ad(HSBC.train)
          + Hi(HSBC.train) - Lo(HSBC.train) + Vo(HSBC.train))

D <- modelData(m.qm.tr)
# ---------------------------------------------------------------------


# Add the additional column:
D$diff.HSBC <- D$Hi.HSBC.train - D$Lo.HSBC.train 

# Note that the last value is NA:
tail(D, n = 3L)

# Since the last value is NA, let us remove it:
D <- D[-nrow(D),]                           
# ---------------------------------------------------------------------


colnames(D) <- c("Next.Op","Ad","Hi","Lo","Vo","Diff")
# ---------------------------------------------------------------------


m1 <- lm(D$Next.Op ~ D$Ad + D$Diff + D$Vo)
summary(m1)
# ---------------------------------------------------------------------


m2 <- lm(D$Next.Op ~ D$Ad + D$Diff)
summary(m2)
# ---------------------------------------------------------------------


qqnorm(m2$residuals)
qqline(m2$residuals, col = 'blue', lwd = 2)
# ---------------------------------------------------------------------


m.qm.tst <- specifyModel(Next(Op(HSBC.test)) ~  Ad(HSBC.test) 
             + Hi(HSBC.test) - Lo(HSBC.test) + Vo(HSBC.test))

D.tst <- modelData(m.qm.tst)
D.tst$diff.HSBC.test <- D.tst$Hi.HSBC.test-D.tst$Lo.HSBC.test  
#tail(D.tst)                           # the last value is NA
D.tst <- D[-nrow(D.tst),]  # remove the last value that is NA

colnames(D.tst) <- c("Next.Op","Ad","Hi","Lo","Vo","Diff")
# ---------------------------------------------------------------------


a   <- coef(m2)['(Intercept)']
bAd <- coef(m2)['D$Ad']
bD  <- coef(m2)['D$Diff']
est <- a + bAd * D.tst$Ad + bD * D.tst$Diff
# ---------------------------------------------------------------------


# -- Mean squared prediction error (MSPE):
#sqrt(mean(((predict(m2,newdata = D.tst) - D.tst$Next.Op)^2)))
sqrt(mean(((est - D.tst$Next.Op)^2)))

# -- Mean absolute errors (MAE):
mean((abs(est - D.tst$Next.Op)))

# -- Mean absolute percentage error (MAPE):
mean((abs(est - D.tst$Next.Op))/D.tst$Next.Op)

# -- Squared sum of residuals:
print(sum(residuals(m2)^2))  

# -- Confidence intervals for the model:
print(confint(m2)) 
# ---------------------------------------------------------------------


# Compare the coefficients in a refit:
m3 <- lm(D.tst$Next.Op ~ D.tst$Ad + D.tst$Diff)
summary(m3)
# ---------------------------------------------------------------------


M0 <- matrix(c(
 1.6 , -0.83 , 1.4 , 4.7 , 1 , 0.9 , 1.1 ,
 1.8 , -0.83 , 1.0 , 4.7 , 1 , 0.9 , 0.8 ,
 1.8 , -0.83 , 1.2 , 4.7 , 1 , 0.9 , 0.6 ,
 1.6 , -1.24 , 1.4 , 2.8 , 1 , 0.9 , 0.8 ,
 0.9 , -0.83 , 1.4 , 4.7 , 1 , 0.7 , 0.8 ,
 0.9 , -0.83 , 0.8 , 4.7 , 1 , 0.7 , 0.6 ,
 0.7 ,  1.02 , 0.2 , 2.0 , 3 , 1.1 , 1.3 ,
 1.1 ,  0.52 , 1.0 , 1.3 , 3 , 0.6 , 0.9 ,
 1.2 , -0.83 , 1.3 , 4.7 , 1 , 0.8 , 0.5 ,
 0.9,  0.18 , 0.9 , 7.3 ,  1 , 0.8 , 0.6 ),
 byrow = TRUE, ncol = 7)
colnames(M0) <- c("tlnt","stab","cost","infl","trvl","infr","life")
# We use the IATA code of a nearby airport as abbreviation,
# so, instead of:
# rownames(M0) <- c("Bangalore", "Mumbai", "Delhi", "Manilla", "Hyderabad", 
#                   "Sao Polo", "Dublin", "Krakow", "Chennai", "Buenos Aires")
# ... we use this:
rownames(M0) <- c("BLR", "BOM", "DEL", "MNL", "HYD", "GRU", 
                  "DUB", "KRK", "MAA", "EZE")

M0  # inspect the matrix
# ---------------------------------------------------------------------


# Political stability is a number between -2.5 and 2.5
# So, we make it all positive by adding 2.5:
M0[,2] <- M0[,2] + 2.5

# Lower wage inflation is better, so invert the data:
M0[,4] <- 1 / M0[,4]

# Then we define a function:

# mcda_rescale_dm 
# Rescales a decision matrix M
# Arguments:
#    M -- decision matrix
#         criteria in columns and higher numbers are better.
# Returns
#    M -- normalised decision matrix
mcda_rescale_dm <- function (M) {
  colMaxs <- function(M) apply(M, 2, max, na.rm = TRUE)
  colMins <- function(M) apply(M, 2, min, na.rm = TRUE)
  M <- sweep(M, 2, colMins(M), FUN="-")
  M <- sweep(M, 2, colMaxs(M) - colMins(M), FUN="/")
  M
}

# Use this function:
M <- mcda_rescale_dm(M0)
# ---------------------------------------------------------------------


# Show the new decision matrix:
knitr::kable(round(M, 2))
# ---------------------------------------------------------------------


# mcda_get_dominated
# Finds the alternatives that are dominated by others
# Arguments:
#    M -- normalized decision matrix with alternatives in rows,
#         criteria in columns and higher numbers are better.
# Returns
#    Dom -- prefM -- a preference matrix with 1 in position ij 
#                    if alternative i is dominated by alternative j.
mcda_get_dominated <- function(M) {
  Dom  <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  dominatedOnes <- c()
  for (i in 1:nrow(M)) {
    for (j in 1:nrow(M)) {
      isDom <- TRUE
      for (k in 1:ncol(M)) {
        isDom <- isDom && (M[i,k] >= M[j,k])
      }
      if(isDom && (i != j)) {
        Dom[j,i] <- 1
        dominatedOnes <- c(dominatedOnes,j)
      }
    }
  }
  colnames(Dom) <- rownames(Dom) <- rownames(M)
  Dom
}
# ---------------------------------------------------------------------


# mcda_get_dominants
# Finds the alternatives that dominate others
# Arguments:
#    M -- normalized decision matrix with alternatives in rows,
#         criteria in columns and higher numbers are better.
# Returns
#    Dom -- prefM -- a preference matrix with 1 in position ij 
#                    if alternative i dominates alternative j.
mcda_get_dominants <- function (M) {
   M <- t(mcda_get_dominated(M))
   class(M) <- "prefM"
   M
   }
# ---------------------------------------------------------------------


Dom <- mcda_get_dominants(M)
print(Dom)
# ---------------------------------------------------------------------


# mcda_del_dominated
# Removes the dominated alternatives from a decision matrix
# Arguments:
#    M -- normalized decision matrix with alternatives in rows,
#         criteria in columns and higher numbers are better.
# Returns
#    A decision matrix without the dominated alternatives
mcda_del_dominated <- function(M) {
  Dom <- mcda_get_dominated(M)
  M[rowSums(Dom) == 0,]
}
# ---------------------------------------------------------------------


M1 <- mcda_del_dominated(M)
knitr::kable(round(M1,2))
# ---------------------------------------------------------------------


M1 <- mcda_rescale_dm(M1)
# ---------------------------------------------------------------------


# First, we load diagram:
require(diagram)

# plot.prefM
# Specific function to handle objects of class prefM for the
# generic function plot()
# Arguments:
#    PM  -- prefM -- preference matrix
#    ... -- additional arguments passed to plotmat()
#           of the package diagram.
plot.prefM <- function(PM, ...)
{
  X <- t(PM) # We want arrows to mean '... is better than ...'
             # plotmat uses the opposite convention, because it expects flows.
  plotmat(X,  
          box.size    = 0.1, 
	  cex.txt     = 0, 
	  lwd         = 5 * X,  # lwd proportional to preference
	  self.lwd    = 3,
	  lcol        = 'blue',
	  self.shiftx = c(0.06, -0.06, -0.06, 0.06),
	  box.lcol    = 'blue',
	  box.col     = 'khaki3',
	  box.lwd     = 2,
	  relsize     = 0.9,
	  box.prop    = 0.5,
	  endhead     = FALSE,
	  main        = "",
	  ...)
}
# ---------------------------------------------------------------------


# We pass the argument 'curve = 0' to the function plotmat, since otherwise 
# the arrow from BLR to MAA would be hidden after the box of EZE.
plot(Dom, curve = 0)
# ---------------------------------------------------------------------


# mcda_wsm
# Calculated the Weigthed Sum MCDA for a decision matrix M and weights w.
# Arguments:
#    M -- normalized decision matrix with alternatives in rows,
#         criteria in columns and higher numbers are better.
#    w -- numeric vector of weights for the criteria
# Returns
#    a vector with a score for each alternative
mcda_wsm <- function(M, w) {
  X <- M %*% w
  colnames(X) <- 'pref'
  X
}
# ---------------------------------------------------------------------


# The critia: "tlnt" "stab" "cost" "infl" "trvl" "infr" "life"
w <- c(        0.125, 0.2,   0.2,   0.2,  0.175,  0.05,  0.05)
w <- w / sum(w)  # the sum was 1 already, but just to be sure.

# Now we can execute our function mcda_wsm():
mcda_wsm(M1, w)
# ---------------------------------------------------------------------


# mcda_wsm_score
# Returns the scores for each of the alternative for each of 
# the criteria weighted by their weights.
# Arguments:
#    M -- normalized decision matrix with alternatives in rows,
#         criteria in columns and higher numbers are better.
#    w -- numeric vector of weights for the criteria
# Returns
#    a score-matrix of class scoreM
mcda_wsm_score <- function(M, w) {
  X <- sweep(M1, MARGIN = 2, w, `*`)
  class(X) <- 'scoreM'
  X
}
# ---------------------------------------------------------------------


# plot.scoreM
# Specific function for an object of class scoreM for the 
# generic function plot().
# Arguments:
#    M -- scoreM -- score matrix
# Returns:
#    plot
plot.scoreM <- function (M) {
  # 1. order the rows according to rowSums
  M <- M[order(rowSums(M), decreasing = T),]
  
  # 2. use a bar-plot on the transposed matrix
  barplot(t(M), 
     legend = colnames(M),
     xlab   = 'Score',
     col    = rainbow(ncol(M))
     )
}
# ---------------------------------------------------------------------


# Whith the normalised decision matrix M1 and the weights w, we calculate the score matrix:
sM <- mcda_wsm_score(M1, w)

# Then we plot the result:
plot(sM)
# ---------------------------------------------------------------------


# mcda_electre Type 2
# Push the preference matrixes PI.plus, PI.min and 
# PI.indif in the environment that calls this function.
# Arguments:
#    M -- normalized decision matrix with alternatives in rows,
#         criteria in columns and higher numbers are better.
#    w -- numeric vector of weights for the criteria
# Returns nothing but leaves as side effect:
#    PI.plus   -- the matrix of preference 
#    PI.min    -- the matrix of non-preference
#    PI.indif  -- the indifference matrix
mcda_electre <- function(M,  w) {
  # initializations
  PI.plus  <<- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  PI.min   <<- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  PI.indif <<- matrix(data=0, nrow=nrow(M), ncol=nrow(M))

  # calculate the preference matrix
  for (i in 1:nrow(M)){
    for (j in 1:nrow(M)) {
      for (k in 1:ncol(M)) {
        if (M[i,k] > M[j,k]) {
          PI.plus[i,j] <<- PI.plus[i,j] + w[k]
        }
        if (M[j,k] > M[i,k]) {
          PI.min[i,j] <<- PI.min[i,j] + w[k]
        }
        if (M[j,k] == M[i,k]) {
          PI.indif[j,i] <<- PI.indif[j,i] + w[k]
        }
      }
    }
  }
}
# ---------------------------------------------------------------------


# mcda_electre1
# Calculates the preference matrix for the ELECTRE method
# Arguments:
#    M -- decision matrix (colnames are criteria, rownames are alternatives)
#    w -- vector of weights
#    Lambda -- the cutoff for the levels of preference
#    r -- the vector of maximum inverse preferences allowed
#    index -- one of ['C1', 'C2']
# Returns:
#    object of class prefM (preference matrix)
mcda_electre1 <- function(M,  w, Lambda, r, index='C1') {
  # get PI.plus, PI.min and PI.indif 
  mcda_electre(M,w)
  
  # initializations
  CM <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  PM <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  colnames(PM) <- rownames(PM) <- rownames(M)
  
  # calcualte the preference matrix
  if (index == 'C1') {
    # for similarity index C1
    for (i in 1:nrow(M)){
      for (j in 1:nrow(M)) {
        CM[i,j] <- (PI.plus[i,j] + PI.indif[i,j]) / (PI.plus[i,j] + 
	            PI.indif[i,j] + PI.min[i,j])
        if((CM[i,j] > Lambda) && ((M[j,] - M[i,]) <= r) && 
	  (PI.plus[i,j] > PI.min[i,j])) PM[i,j] = 1
      }
    }
  } else {
    # for similarity index C2
    for (i in 1:nrow(M)){
      for (j in 1:nrow(M)) {
        if (PI.min[i,j] != 0) 
        {CM[i,j] <- (PI.plus[i,j]) / (PI.min[i,j])}
        else
        {CM[i,j] = 1000 * PI.plus[i,j]} # to avoid dividing by 0
        if((CM[i,j] > Lambda) && ((M[j,] - M[i,]) <= r) && 
	   (PI.plus[i,j] > PI.min[i,j])) {PM[i,j] = 1}
      }
    }
  }
  for (i in 1:nrow(PM)) PM[i,i] = 0
  class(PM) <- 'prefM'
  PM
}
# ---------------------------------------------------------------------


list(PI.plus = PI.plus, PI.min = PI.min, PI.indif = PI.indif)
# ---------------------------------------------------------------------


# If we did not push the values PI.plus, PI.min, and 
# PI.indif into the environment of this functions, we would write:
X <- mcda_electre(M,w)
# and then address X as in the following code as follows:
X$PI.min[i,j]
# ---------------------------------------------------------------------


# the critia: "tlnt" "stab" "cost" "infl" "trvl" "infr" "life"
w <- c(        0.125, 0.2,   0.2,   0.2,  0.175,  0.05,  0.05)
w <- w / sum(w)  # the sum was 1 already, but just to be sure.
r  <- c(0.3,    0.5,  0.5,   0.5,   1,     0.9,   0.5)

eM <- mcda_electre1(M1, w, Lambda=0.6, r=r)
print(eM)
plot(eM)
# ---------------------------------------------------------------------


# the critia: "tlnt" "stab" "cost" "infl" "trvl" "infr" "life"
w <- c(        0.125, 0.2,   0.2,   0.2,  0.175,  0.05,  0.05)
w <- w / sum(w)  # the sum was 1 already, but just to be sure.
r  <- c(0.3,    0.5,  0.5,   0.5,   1,     0.9,   0.5)

eM <- mcda_electre1(M1, w, Lambda=1.25, r=r, index='C2')
plot(eM)
# ---------------------------------------------------------------------


# The critia: "tlnt" "stab" "cost" "infl" "trvl" "infr" "life"
w <- c(        0.125, 0.2,   0.2,   0.2,  0.175,  0.05,  0.05)
w <- w / sum(w)  # the sum was 1 already, but just to be sure.
r  <- c(1,    1,  1,   1,   1,     1,   1)

eM <- mcda_electre1(M1, w, Lambda = 0.0, r = r)
print(eM)
plot(eM)
# ---------------------------------------------------------------------


mcda_electre2 <- function (M1, w) {
  r <- rep(1L, ncol(M))
  mcda_electre1(M1, w, Lambda = 0.0, r = rep(1, length.out = length(w)))
  }
# ---------------------------------------------------------------------


sum(rowSums(prefM)) == A * A - A
# with prefM the preference matrix and 
#      A the number of alternatives.
# ---------------------------------------------------------------------


library(ggplot2)
library(latex2exp)
d <- seq(from = -3, to = +3, length.out = 100)

## error function family:
erf <- function(x) 2 * pnorm(x * sqrt(2)) - 1
# (see Abramowitz and Stegun 29.2.29)
erfc <- function(x) 2 * pnorm(x * sqrt(2), lower = FALSE)
erfinv <- function (x) qnorm((1 + x)/2)/sqrt(2)
erfcinv <- function (x) qnorm(x/2, lower = FALSE)/sqrt(2)
 
## Gudermannian function
gd <- function(x) asin(tanh(x))

f1 <- function(x) erf( sqrt(pi) / 2 * x)
f2 <- function(x) tanh(x)
f3 <- function(x) 2 / pi * gd(pi / 2 * x)
f4 <- function(x) x / sqrt(1 + x^2)
f5 <- function(x) 2 / pi * atan(pi / 2 * x)
f6 <- function(x) x / (1 + abs(x))

df <- data.frame(d = d, y = f1(d), Function = "erf( sqrt(pi) / 2 * d)")
df <- rbind(df, data.frame(d = d, y = f2(d), Function = "tanh(d)"))
df <- rbind(df, data.frame(d = d, y = f3(d), Function = "2 / pi * gd(pi / 2 * d)"))
df <- rbind(df, data.frame(d = d, y = f4(d), Function = "d / (1 + d^2)"))
df <- rbind(df, data.frame(d = d, y = f5(d), Function = "2 / pi * atan(pi / 2 * d)"))
df <- rbind(df, data.frame(d = d, y = f6(d), Function = "x / (1 + abs(d))"))

fn <- ""
fn[1] <- "erf \\left(\\frac{\\sqrt{\\pi} d}{2}\\right)"
fn[2] <- "tanh(x)"
fn[3] <- "\\frac{2}{\\pi} gd\\left( \\frac{\\pi d}{2} \\right)"
fn[4] <- "\\frac{d}{1 + d^2}"
fn[5] <- "\\frac{2}{\\pi} atan\\left(\\frac{\\pi d}{2}\\right)"
fn[6] <- "\\frac{x}{1+ |x|}"


ggplot(data = df, aes(x = d, y = y, color = Function)) +
   geom_line(aes(col = Function), lwd=2) +
   guides(color=guide_legend(title=NULL)) +
   scale_color_discrete(labels=lapply(sprintf('$\\pi(d) = %s$', fn), TeX)) + 
   theme(legend.justification = c(1, 0), 
        legend.position = c(1, 0),   # south east
        legend.box.margin=ggplot2::margin(rep(20, times=4)),
        legend.key.size = unit(1.5, "cm")  # increase vertical space between legend items
	) +
   ylab(TeX('Preference --- $\\pi$'))
# ---------------------------------------------------------------------


f_curve <- function(f) {
  g <- Vectorize(f)
  s <- deparse(f)[2]
   curve(g, xlab = '', ylab = '', col = 'red', lwd = 3, 
         from = -1, to = +1,
         main = bquote(bold(.(s)))
#          main = s
	  )
   }

gaus <- function(x) exp (-(x-0)^2 / (0.5)^2)
f1 <- function(x) ifelse(x<0, 0, - 3/2 * x^5 + 5/2 * x^3)
f2 <- function(x) ifelse(x<0, 0, sin(pi * x / 2))
f3 <- function(x) min(1, max(1.5*x-0.2, 0))
f4 <- function(x) ifelse(x<0, 0, x)
f5 <- function(x) ifelse(x < 0, 0, 1 - gaus(x))
f6 <- function(x) 1+tanh(6*(x-0.6))

par(mfrow=c(3,2))
f_curve(f1)
f_curve(f2)
f_curve(f3)
f_curve(f4)
f_curve(f5)
f_curve(f6)
par(mfrow=c(1,1))
# ---------------------------------------------------------------------


# mcda_promethee
# delivers the preference flow matrices for the Promethee method
# Arguments:
#    M      -- decision matrix
#    w      -- weights
#    piFUNs -- a list of preference functions, 
#              if not provided min(1,max(0,d)) is assumed.
# Returns (as side effect)
# phi_plus <<- rowSums(PI.plus)
# phi_min  <<- rowSums(PI.min)
# phi_     <<- phi_plus - phi_min
#
mcda_promethee <- function(M, w, piFUNs='x')
{
  if (piFUNs == 'x') {
       # create a factory function:
       makeFUN <- function(x) {x; function(x) max(0,x) }
       P <- list()
       for (k in 1:ncol(M)) P[[k]] <- makeFUN(k)
       } # in all other cases we assume a vector of functions
# initializations
PI.plus  <<- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
PI.min   <<- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
# calculate the preference matrix
for (i in 1:nrow(M)){
  for (j in 1:nrow(M)) {
    for (k in 1:ncol(M)) {
      if (M[i,k] > M[j,k]) {
        PI.plus[i,j] = PI.plus[i,j] + w[k] * P[[k]](M[i,k] - M[j,k])
      }
      if (M[j,k] > M[i,k]) {
        PI.min[i,j] = PI.min[i,j] + w[k] * P[[k]](M[j,k] - M[i,k])
      }
    }
  }
}
# note the <<- which pushes the results to the upwards environment
phi_plus <<- rowSums(PI.plus)
phi_min  <<- rowSums(PI.min)
phi_     <<- phi_plus - phi_min
}
# ---------------------------------------------------------------------


# mcda_promethee1
# Calculates the preference matrix for the Promethee1 method
# Arguments:
#    M      -- decision matrix
#    w      -- weights
#    piFUNs -- a list of preference functions, 
#              if not provided min(1,max(0,d)) is assumed.
# Returns:
#    prefM object -- the preference matrix
#
mcda_promethee1 <- function(M, w, piFUNs='x') {
  # mcda_promethee adds phi_min, phi_plus & phi_ to this environment:
  mcda_promethee(M, w, piFUNs='x') 
  
  # Now, calculate the preference relations:
  pref     <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
    for (i in 1:nrow(M)){
      for (j in 1:nrow(M)) {
        if (phi_plus[i] == phi_plus[j] && phi_min[i]==phi_min[j]) {
	    pref[i,j] <- 0
	  }
	  else if ((phi_plus[i] > phi_plus[j] && 
	            phi_min[i] < phi_min[j] ) || 
	          (phi_plus[i] >= phi_plus[j] && 
		    phi_min[i] < phi_min[j] )) {
	    pref[i,j] <- 1
	  }
	  else {
	    pref[i,j] = NA
	  }
	}
      }
  rownames(pref) <- colnames(pref) <- rownames(M)
  class(pref)    <- 'prefM'
  pref
}
# ---------------------------------------------------------------------


# We reuse the decision matrix M1 and weights w as defined above.
m <- mcda_promethee1(M1, w)
# ---------------------------------------------------------------------


# We reuse the decision matrix M1 and weights w as defined above.
m <- mcda_promethee1(M1, w)
plot(m)
# ---------------------------------------------------------------------


# Make shortcuts for some of the functions that we will use:
gauss_val <- function(d) 1 - exp(-(d - 0.1)^2 / (2 * 0.5^2))
x         <- function(d) max(0,d)
minmax    <- function(d) min(1, max(0,2*(d-0.5)))
step      <- function(d) ifelse(d > 0.5, 1,0)

# Create a list of 7 functions (one per criterion):
f <- list()
f[[1]] <- gauss_val
f[[2]] <- x
f[[3]] <- x
f[[4]] <- gauss_val
f[[5]] <- step
f[[6]] <- x
f[[7]] <- minmax

# Use the functions in mcda_promethee1:
m <- mcda_promethee1(M1, w, f)

# Plot the results:
plot(m)
# ---------------------------------------------------------------------


library(latex2exp)
f_curve <- function(f) {
  g <- Vectorize(f)
   curve(g, xlab = '', ylab = '', col = 'red', lwd = 3, 
         from = -1, to = +1,
         main = TeX(toString(deparse(f)[2])))
   }

gaus <- function(x) exp (-(x-0)^2 / 0.5)
f1 <- function(x) - 3/2 * x^5 + 5/2 * x^3
f2 <- function(x) sin(pi * x / 2)
f3 <- function(x) min(1, max(2*x, -1))
f4 <- function(x) x
f5 <- function(x) ifelse(x < 0 , gaus(x) - 1, 1 - gaus(x))
f6 <- function(x) tanh(3 * x)

par(mfrow=c(3,2))
f_curve(f1)
f_curve(f2)
f_curve(f3)
f_curve(f4)
f_curve(f5)
f_curve(f6)
par(mfrow=c(1,1))
# ---------------------------------------------------------------------


# mcda_promethee2
# Calculates the Promethee2 preference matrix
# Arguments:
#    M      -- decision matrix
#    w      -- weights
#    piFUNs -- a list of preference functions, 
#              if not provided min(1,max(0,d)) is assumed.
# Returns:
#    prefM object -- the preference matrix
#
mcda_promethee2 <- function(M, w, piFUNs='x')
 { # promethee II
  mcda_promethee(M, w, piFUNs='x')
  pref     <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
    for (i in 1:nrow(M)){
      for (j in 1:nrow(M)) {
          pref[i,j] <- max(phi_[i] - phi_[j],0)
        }
      }
rownames(pref) <- colnames(pref) <- rownames(M)
class(pref) <- 'prefM'
pref
}
# ---------------------------------------------------------------------


m <- mcda_promethee2(M1, w)
plot(m)
# ---------------------------------------------------------------------


# We can consider the rowSums as a "score".
rowSums(m)

# So, consider the prefM as a score-matrix (scoreM):
plot.scoreM(m)
# ---------------------------------------------------------------------


pca1 <- prcomp(M1) 
summary(pca1)

# plot for the prcomp object shows the variance explained by each PC
plot(pca1, type = 'l')

# biplot shows a projection in the 2D plane (PC1, PC2)
biplot(pca1)
# ---------------------------------------------------------------------


library(ggplot2)
library(ggfortify)
library(cluster)

# Autoplot with labels colored
autoplot(pca1, data = M1, label = TRUE, shape = FALSE, colour='cost', label.size = 6,
         loadings = TRUE, loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 6
         )

# Autoplot with visualization of 2 clusters
autoplot(fanny(M1,2), label=TRUE, frame=TRUE, shape = FALSE, label.size = 6,
         loadings = TRUE,  loadings.colour = 'blue',
         loadings.label = TRUE, loadings.label.size = 6)
# ---------------------------------------------------------------------


# Use the weights as defined above:
w

# Calculate coordinates
dv1 <- sum( w * pca1$rotation[,1])  # decision vector PC1 component
dv2 <- sum( w * pca1$rotation[,2])  # decision vector PC2 component

p <- autoplot(pam(M1,2), frame=TRUE, frame.type='norm', label=TRUE, 
         shape=FALSE,
         label.colour='blue',label.face='bold', label.size=6,
         loadings=TRUE,  loadings.colour = 'dodgerblue4',
         loadings.label = TRUE, loadings.label.size = 6, 
	 loadings.label.colour='dodgerblue4',
         loadings.label.vjust = 1.2, loadings.label.hjust = 1.3
         )
p <- p + scale_y_continuous(breaks = 
                        round(seq(from = -1, to = +1, by = 0.2), 2))
p <- p + scale_x_continuous(breaks = 
                        round(seq(from = -1, to = +1, by = 0.2), 2))
p <- p + geom_segment(aes(x=0, y=0, xend=dv1, yend=dv2), size = 2,
                      arrow = arrow(length = unit(0.5, "cm")))
p <- p + ggplot2::annotate("text", x = dv1+0.2, y = dv2-0.01, 
                  label = "decision vector",
                  colour = "black", fontface = 2)
p
# ---------------------------------------------------------------------


### Outrank
# M is the decision matrix (formulated for a maximum problem)
# w the weights to be used for each rank
outrank <- function (M, w)
{
  order      <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  order.inv  <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  order.pref <- matrix(data=0, nrow=nrow(M), ncol=nrow(M))
  
  for (i in 1:nrow(M)){
    for (j in 1:nrow(M)) {
      for (k in 1:ncol(M)) {
        if (M[i,k] > M[j,k]) { order[i,j] = order[i,j] + w[k] }
        if (M[j,k] > M[i,k]) { order.inv[i,j] = order.inv[i,j] + w[k] }
      }
    }
  }
  for (i in 1:nrow(M)){
    for (j in 1:nrow(M)) {
      if (order[i,j] > order[j,i]){
        order.pref[i,j] = 1
        order.pref[j,i] = 0
      }
      else if (order[i,j] < order[j,i]) {
        order.pref[i,j] = 0
        order.pref[j,i] = 1
      }
      else {
        order.pref[i,j] = 0
        order.pref[j,i] = 0
      }
    }
  }
 class(order.pref) <- 'prefM'
 order.pref
}
