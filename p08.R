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


# Load the library:
library(parallel)

# The function detectCores finds the number of cores available:
numCores <- detectCores() 
numCores 
# ---------------------------------------------------------------------


# Set the sample size:
N <- 100

# Set the starting points for the k-means algorithm:
starts <- rep(100, N)   # 50 times the value 100

# Load the dataset of the Titanic disaster:
library(titanic)

# Prepare data as numeric only:
t <- as.data.frame(titanic_train)
t <- t[complete.cases(t),]
t <- t[,c(2,3,5,6,7,8)]
t$Sex <- ifelse(t$Sex == 'male', 1, 0)

# Prepare the functions to be executed:
f <- function(nstart) kmeans(t, 4, nstart = nstart)

# Now, we are ready to go, we try two approaches:

# 1. With the standard function base::lapply
system.time(results <- lapply(starts, f))

# 2. With parallel::mclapply
system.time(results <- mclapply(starts, f, mc.cores = numCores))
# ---------------------------------------------------------------------


# 1. The regular for loop:
for (n in 1:4) print(gamma(n))

# 2. The foreach loop:
library(foreach)
foreach (n = 1:4) %do% print(gamma(n))
# ---------------------------------------------------------------------


class(foreach (n = 1:4))
# ---------------------------------------------------------------------


# Once installed, the package must be loaded once per session:
library(doParallel)

# Find the number of available cores (as in previous section):
numCores <- parallel::detectCores() 

# Register doParallel:
registerDoParallel(numCores)  

# Now, we are ready to put it in action:
foreach (n = 1:4) %dopar% print(gamma(n))
# ---------------------------------------------------------------------


# Collapse to wide data.frame:
foreach (n = 1:4, .combine = cbind) %dopar% print(gamma(n))

# Collapse to long data.frame:
foreach (n = 1:4, .combine = rbind) %dopar% print(gamma(n))

# Collapse to vector:
foreach (n = 1:4, .combine = c)     %dopar% print(gamma(n))
# ---------------------------------------------------------------------


# We reuse parameters and data as defined in the previous section:
system.time(
foreach (n = 1:N) %dopar% {
    kmeans(t, 4, nstart=100)
    }
  )
# ---------------------------------------------------------------------


detach("package:doParallel", unload=TRUE)
#detach("package:parallel", unload=TRUE)
# ---------------------------------------------------------------------


library(snow)
# ---------------------------------------------------------------------


detach("package:parallel", unload = TRUE)  
library(snow)
# ---------------------------------------------------------------------


cl <- makeCluster(c("localhost", "localhost"), type = "SOCK")
# ---------------------------------------------------------------------


library(titanic)  # provides the dataset: titanic_train
N <- 50
starts <- rep(100, N)   # 50 times the value 100

# Prepare data as numeric only:
t <- as.data.frame(titanic_train)
t <- t[complete.cases(t),]
t <- t[,c(2,3,5,6,7,8)]
t$Sex <- ifelse(t$Sex == 'male', 1, 0)

# Prepare the functions to be executed:
f <- function(nstart) kmeans(t, 4, nstart=nstart)

# 1. with the standard function 
system.time(results <- lapply(starts, f))

# 2. with the cluster
# First, we must export the object t,  so that it can 
# be used by the cluster:
clusterExport(cl, "t")
#clusterExport(cl, "starts") # Not needed since it is in the function f
system.time(
  result2 <- parLapply(cl, starts, f)
  )
# ---------------------------------------------------------------------


f <- function (x, y) x + y + rnorm(1)
clusterCall(cl, function (x, y) {x + y + rnorm(1)}, 0, pi)
clusterCall(cl, f, 0, pi)

# Both forms are semantically similar to:
clusterCall(cl, eval, f(0,pi))
# However, note that the random numbers are the same on both clusters.
# ---------------------------------------------------------------------


# Note that ...
clusterEvalQ(cl, rnorm(1))

# ... is almost the same as
clusterCall(cl, evalq, rnorm(1))
# ... but that the random numbers on both slaves nodes are the same
# ---------------------------------------------------------------------


clusterApply(cl, c(0, 10), sum, pi)
# ---------------------------------------------------------------------


clusterApplyLB(cl, seq, fun, ...)
# ---------------------------------------------------------------------


stopCluster(cl)
# ---------------------------------------------------------------------


detach("package:snow", unload=TRUE)
# ---------------------------------------------------------------------


# Remove a list of selected variables:
rm (list = c('A','B', 'gpuA', 'gpuB', 'vlcA', 'vlcB')) 

# Alternatively remove *all* variables previously defined (be careful!):
rm(list = ls()) 

# Run the garbage collector:
gc()                        

# If not longer needed, unload gpuR:
detach('package:gpuR', unload = TRUE)
# ---------------------------------------------------------------------


require(gpuR)
require(tidyr)
require(ggplot2)
set.seed(1890)
NN <- seq(from = 500, to = 5500,by = 1000)
t <- data.frame(N = numeric(), `CPU/CPU` = numeric(),
                `CPU/GPU` = numeric(), `GPU/GPU` = numeric())
i <- 1

# Run the experiment:
for(k in NN) {
  A <- matrix(rnorm(k^2), nrow = k, ncol = k)
  # storage in CPU-RAM, calculations in CPU
  gpuA <- gpuMatrix(A) # storage in CPU-RAM, calculations in GPU
  vclA <- vclMatrix(A) # storage in GPU-RAM, calculations in GPU
  t[i,1] <- k
  t[i,2] <- system.time(B <- A %*% A)[3]
  t[i,3] <- system.time(gpuB <- gpuA %*% gpuA)[3]
  t[i,4] <- system.time(vclB <- vclA %*% vclA)[3]
  i <- i + 1
  }
  
# Print the results:
t

# Tidy up the data-frame:
tms <- gather(t, 'RAM/PU', 'time', 2:4)

# Plot the results:
scaleFUN <- function(x) sprintf("%.2f", x)
p <- ggplot(tms, aes(x = N, y = time, colour = `RAM/PU`)) +
     geom_point(size=5) +
     geom_line() +
     scale_y_continuous(trans = 'log2', labels = scaleFUN) + # NEW!!
     xlab('Matrix size (number of rows and columns)') +
     ylab('Time in seconds') +
     theme(axis.text = element_text(size=12),
           axis.title = element_text(size=14)
           )
print(p)
# ---------------------------------------------------------------------


# install from github
devtools::install_github(repo = "rstudio/spark-install", 
                         subdir = "R")
library(sparkinstall)

# lists the versions available to install
spark_available_versions()

# installs an specific version
spark_install(version = "2.4.3")

# uninstalls an specific version
spark_uninstall(version = "2.4.3", hadoop_version = "2.6")
# ---------------------------------------------------------------------


system('start-master.sh')
# ---------------------------------------------------------------------


library(tidyverse)
library(dplyr)
library(SparkR)
# Note that loading SparkR will generate many warning messages,
# because it overrrides many functions such as summary, first,
# last, corr, ceil, rbind, expr, cov, sd and many more.

sc <- sparkR.session(master = "local", appName = 'first test',
                     sparkConfig = list(spark.driver.memory = '2g'))

# Show the session:
sc
# ---------------------------------------------------------------------


# Create a SparkDataFrame:
DF <- as.DataFrame(mtcars)

# The DataFrame is for big data, so the attempt to print all data,
# might surprise us a little:
DF
# R assumes that the data-frame is big data and does not even
# start printing all data.

# head() will collapse the first lines to a data.frame:
head(DF)
# ---------------------------------------------------------------------


# If DDF is a distributed DataFrame defined by SparkR,
# we can add a checkpoint as follows:
DDF <- SparkR::checkpoint(DDF)
# ---------------------------------------------------------------------


library(titanic)
library(tidyverse)

# This provides a.o. two datasets titanic_train and titanic_test.
# We will work further with the training-dataset:
T <- as.DataFrame(titanic_train)
# ---------------------------------------------------------------------


# The SparkDataFrame inherits from data.frame, so most functions 
# work as expected on a DataFrame:
colnames(T)
str(T)
summary(T)
class(T)

# The scheme is a declaration of the structure:
printSchema(T)

# Truncated information collapses to data.frame:
T %>% head %>% class
# ---------------------------------------------------------------------


X <- T %>% SparkR::select(T$Age)         %>% head
Y <- T %>% SparkR::select(column('Age')) %>% head
Z <- T %>% SparkR::select(expr('Age'))   %>% head
cbind(X, Y, Z)
# ---------------------------------------------------------------------


T %>% 
  SparkR::filter("Age < 20 AND Sex == 'male' AND Survived == 1") %>%
  SparkR::select(expr('PassengerId'), expr('Pclass'), expr('Age'), 
                 expr('Survived'), expr('Embarked'))             %>%
  head
      
# The following is another approach. The end-result is the same, however, 
# we bring the data first to the R's working memory and then use dplyr. 
# Note the subtle differences in syntax.
SparkR::collect(T)                                              %>%
      dplyr::filter(Age < 20 & Sex == 'male' & Survived == 1)   %>%
      dplyr::select(PassengerId, Pclass, Age, Survived,
                   Embarked)                                    %>%
      head
# ---------------------------------------------------------------------


# Extract the survival percentage per class for each gender:
TMP <- T                                              %>% 
       SparkR::group_by(expr('Pclass'), expr('Sex'))  %>%
             summarize(countS = sum(expr('Survived')), count = n(expr('PassengerId')))
N   <- nrow(T)

TMP                                                          %>% 
    mutate(PctAll   = expr('count') / N  * 100)              %>% 
    mutate(PctGroup = expr('countS') / expr('count')  * 100) %>% 
    arrange('Pclass', 'Sex')                                 %>% 
    SparkR::collect()
# ---------------------------------------------------------------------


library(tidyverse)
library(SparkR)
library(titanic)
sc <- sparkR.session(master = "local", appName = 'first test',
                     sparkConfig = list(spark.driver.memory = '2g'))
T <- as.DataFrame(titanic_train)
# ---------------------------------------------------------------------


# The data:
T <- as.DataFrame(titanic_train)

# The schema can be a structType:
schema <- SparkR::structType(SparkR::structField("Age", "double"), 
                        SparkR::structField("ageGroup", "string"))

# Or (since Spark 2.3) it can also be a DDL-formatted string
schema <- "age DOUBLE, ageGroup STRING"

# The function to be applied:
f <- function(x) { 
  data.frame(x$Age, if_else(x$Age < 30, "youth", "mature")) 
  }

# Run the function f on the Spark cluster:
T2 <- SparkR::dapply(T, f, schema)

# Inspect the results:
head(SparkR::collect(T2))

##     age           ageGroup
## 1    22              youth
## 2    38             mature
## 3    26              youth
## 4    35             mature
## 5    35             mature
## 6    NA               <NA>
# ---------------------------------------------------------------------


DFcars  <- createDataFrame(mtcars)
DFcars2 <- dapply(DFcars, function(x) {x}, schema(DFcars))
head(collect(DFcars2))
# ---------------------------------------------------------------------


# clean up
rm(T)
gc()
# ---------------------------------------------------------------------


# The data:
T <- as.DataFrame(titanic_train)

# The function to be applied:
f <- function(x) { 
  y <- data.frame(x$Age, ifelse(x$Age < 30, "youth", "mature")) 
  
  # We specify now column names in the data.frame to be returned:
    colnames(y) <- c("age", "ageGroup")
  
  # and we return the data.frame (base R type):
  y
  }

# Run the function f on the Spark cluster:
T2_DF <- dapplyCollect(T, f)

# Inspect the results (T2_DF is now a data.frame, no collect needed):
head(T2_DF)
# ---------------------------------------------------------------------


# define the function to be used:
f <- function (key, x) {
  data.frame(key, min(x$Age,  na.rm = TRUE), 
                  mean(x$Age, na.rm = TRUE), 
		  max(x$Age,  na.rm = TRUE))
  }

# The schema also can be specified via a DDL-formatted string
schema <- "class INT, min_age DOUBLE, avg_age DOUBLE, max_age DOUBLE"

maxAge <- gapply(T, "Pclass", f, schema)

head(collect(arrange(maxAge, "class", decreasing = TRUE)))
# ---------------------------------------------------------------------


# define the function to be used:
f <- function (key, x) {
  y <- data.frame(key, min(x$Age,  na.rm = TRUE), 
                       mean(x$Age, na.rm = TRUE), 
	               max(x$Age,  na.rm = TRUE))
  colnames(y) <- c("class", "min_age", "avg_age", "max_age")
  y
  }

maxAge <- gapplyCollect(T, "Pclass", f)

head(maxAge[order(maxAge$class, decreasing = TRUE), ])
# ---------------------------------------------------------------------


# First a trivial example to show how spark.lapply works:
surfaces <- spark.lapply(1:3, function(x){pi * x^2})
print(surfaces)
# ---------------------------------------------------------------------


mFamilies   <- c("binomial", "gaussian")
trainModels <- function(fam) {
  m <- SparkR::glm(Survived ~ Age + Pclass, 
                   data = T, 
		   family = fam)
  summary(m)
  }
mSummaries <- spark.lapply(mFamilies, trainModels)
# ---------------------------------------------------------------------


# df is a data.frame (R-data.frame)
# DF is a DataFrame (distributed Spark data-frame)

# We already saw how to get data from Spark to R:
df <- collect(DF)

# From R to Spark:
DF <- createDataFrame(df)
# ---------------------------------------------------------------------


loadDF(fileName,
       source = "csv",
       header = "true",
       sep = ",")
# ---------------------------------------------------------------------


T %>% withColumn("AgeGroup", column("Age") / lit(10))                    %>%
      SparkR::select(expr('PassengerId'), expr('Age'), expr('AgeGroup')) %>%
      head
# ---------------------------------------------------------------------


T %>% cube("Pclass", "Sex")  %>%
      agg(avg(T$Age))        %>%
      collect()
# ---------------------------------------------------------------------


# Prepare training and testing data:
T_split <- randomSplit(T, c(8,2), 2)
T_train <- T_split[[1]]
T_test  <- T_split[[2]]

# Fit the model:
M1 <- spark.glm(T_train, Survived ~ Pclass + Sex, family = "binomial")

# Save the model:
path1 <- tempfile(pattern = 'ml', fileext = '.tmp')
write.ml(M1, path1)
# ---------------------------------------------------------------------


# Retrieve the model
M2 <- read.ml(path1)

# Do something with M2:
summary(M2)
# ---------------------------------------------------------------------


# Add predictions to the model for the test data:
PRED1 <- predict(M2, T_test)

# Show the results:
x <- head(collect(PRED1))
head(cbind(x$Survived, x$prediction))
# ---------------------------------------------------------------------


# Close the connection:
unlink(path1)
# ---------------------------------------------------------------------


# We unload the package SparkR.
detach("package:SparkR",    unload = TRUE)
detach("package:tidyverse", unload = TRUE)
detach("package:dplyr",     unload = TRUE)
# ---------------------------------------------------------------------


install.packages('sparklyr')
# ---------------------------------------------------------------------


# First, load the tidyverse packages:
library(tidyverse)
library(dplyr)

# Load the package sparklyr:
library(sparklyr)

# Load the data:
library(titanic)

# Our spark-master is already running, so no need for:
# system('start-master.sh')

# Connect to the local Spark master
sc <- spark_connect(master = "local")
# ---------------------------------------------------------------------


Titanic_tbl <- copy_to(sc, titanic_train)
Titanic_tbl

## # Source: spark<titanic_train> [?? x 12]
##    PassengerId Survived Pclass Name    Sex     Age SibSp Parch ...
##          <int>    <int>  <int> <chr>   <chr> <dbl> <int> <int> ...
##  1           1        0      3 Brau... male     22     1     0 ...
##  2           2        1      1 Cumi... female   38     1     0 ...
##  3           3        1      3 Heik... female   26     0     0 ...
##  4           4        1      1 Futr... female   35     1     0 ...
##  5           5        0      3 Alle... male     35     0     0 ...
##  6           6        0      3 Mora... male    NaN     0     0 ...
##  7           7        0      1 McCa... male     54     0     0 ...
##  8           8        0      3 Pals... male      2     3     1 ...
##  9           9        1      3 John... female   27     0     2 ...
## 10          10        1      2 Nass... female   14     1     0 ...
## # ... with more rows, and 1 more variable: Embarked <chr>


# More datasets can be stored in the same connection:
cars_tbl <- copy_to(sc, mtcars)

# List the available tables:
src_tbls(sc)
## [1] "mtcars"        "titanic_train"
# ---------------------------------------------------------------------


Titanic_tbl %>% summarise(n = n())
## # Source: spark<?> [?? x 1]
##       n
##   <dbl>
## 1   891

# Alternatively:
Titanic_tbl %>% spark_dataframe() %>% invoke("count")
## [1] 891

Titanic_tbl                                   %>% 
  dplyr::group_by(Sex, Embarked)              %>%
  summarise(count = n(), AgeMean = mean(Age)) %>%
  collect
## # A tibble: 7 x 4
##   Sex    Embarked count AgeMean
##   <chr>  <chr>    <dbl>   <dbl>
## 1 male   C           95    33.0
## 2 male   S          441    30.3
## 3 female C           73    28.3
## 4 female ""           2    50  
## 5 female S          203    27.8
## 6 male   Q           41    30.9
## 7 female Q           36    24.3
# ---------------------------------------------------------------------


library(DBI)
sSQL <- "SELECT Name, Age, Sex, Embarked FROM titanic_train 
         WHERE Embarked = 'Q' LIMIT 10"
dbGetQuery(sc, sSQL)
##                            Name Age    Sex Embarked
## 1              Moran, Mr. James NaN   male        Q
## 2          Rice, Master. Eugene   2   male        Q
## 3   McGowan, Miss. Anna "Annie"  15 female        Q
## 4 O'Dwyer, Miss. Ellen "Nellie" NaN female        Q
## 5      Glynn, Miss. Mary Agatha NaN female        Q
# ---------------------------------------------------------------------


# sdf_len creates a DataFrame of a given length (5 in the example)
x <- sdf_len(sc, 5, repartition = 1) %>% 
     spark_apply(function(x) pi * x^2)
print(x)
## # Source: spark<?> [?? x 1]
##      id
##   <dbl>
## 1  3.14
## 2 12.6 
## 3 28.3 
## 4 50.3 
## 5 78.5
# ---------------------------------------------------------------------


# Transform data and create buckets for Age:
t <- Titanic_tbl %>%
  ft_bucketizer(input_col  = "Age",
                output_col = "AgeBucket",
                splits     = c(0, 10, 30, 90)) 

# Split data in training and testing set:
partitions <- t %>%
  sdf_random_split(training = 0.7, test = 0.3, seed = 1942)
t_training <- partitions$training
t_test     <- partitions$test 
  
# Fit a logistic regression model
M <- t_training %>% 
    ml_logistic_regression(Survived ~ AgeBucket + Pclass + Sex)
 
# Add predictions:
pred <- sdf_predict(t_test, M)

# Show details of the quality of the fit of the model:
ml_binary_classification_evaluator(pred)
# ---------------------------------------------------------------------


system('stop-master.sh')
#detach("package:sparklyr", unload=TRUE)
detach("package:SparkR", unload=TRUE)
detach("package:tidyverse", unload=TRUE)
#detach("package:dplyr", unload=TRUE)
# ---------------------------------------------------------------------


x <- 1:1e4

# Tmiming the function mean:
system.time(mean(x))
# ---------------------------------------------------------------------


N <- 2500

# Repeating it to gather longer time:
system.time({for (i in 1:N) mean(x)})
# ---------------------------------------------------------------------


system.time({for (i in 1:N) mean(x)})
system.time({for (i in 1:N) mean(x)})
# ---------------------------------------------------------------------


# Timing a challenger model:
system.time({for (i in 1:N) sum(x) / length(x)})
# ---------------------------------------------------------------------


N <- 1500

# Load microbenchmark:
library(microbenchmark)

# Create a microbenchmark object:
comp <- microbenchmark(mean(x),              # 1st code block
                       {sum(x) / length(x)}, # 2nd code block
		       times = N)            # number times to run
# ---------------------------------------------------------------------


summary(comp)
# ---------------------------------------------------------------------


# Load ggplot2:
library(ggplot2)

# Use autoplot():
autoplot(comp)
# ---------------------------------------------------------------------


x <- 1:1e+4
y <- 0

# Here we recalculate the sum multiple times.
system.time(for(i in x) {y <- y + sum(x)})

# Here we calculate it once and store it.
system.time({sum_x <- sum(x); for(i in x) {y <- y + sum_x}})
# ---------------------------------------------------------------------


# Define some numbers.
x1 <- 3.00e8
x2 <- 663e-34
x3 <- 2.718282
x4 <- 6.64e-11

y1 <- pi
y2 <- 2.718282
y3 <- 9.869604
y4 <- 1.772454

N <- 1e5
# ---------------------------------------------------------------------


# 1. Adding some of them directly:
f1 <- function () {
  for (i in 1:N) {
    x1 + y1
    x2 + y2
    x3 + y3
    x4 + y4
    }
  }
system.time(f1())
# ---------------------------------------------------------------------


# 2. Converting first to a vector and then adding the vectors:
f2 <- function () {
  x <- c(x1, x2, x3, x4)
  y <- c(y1, y2, y3, y4)
  for (i in 1:N) x + y
}
system.time(f2())
# ---------------------------------------------------------------------


# 3. Working with the elements of the vectors:
f3 <- function () {
  x <- c(x1, x2, x3, x4)
  y <- c(y1, y2, y3, y4)
  for (i in 1:N) {
    x[1] + y[1]
    x[2] + y[2]
    x[3] + y[3]
    x[4] + y[4]
  }
  }
system.time(f3())
# ---------------------------------------------------------------------


# 4. Working with the elements of the vectors and code shorter:
f4 <- function () {
  x <- c(x1, x2, x3, x4)
  y <- c(y1, y2, y3, y4)
  for (i in 1:N) for(n in 1:4) x[n] + y[n]
  }
system.time(f4())
# ---------------------------------------------------------------------


# Remind the packages:
library(microbenchmark)
library(ggplot2)

# Compare all four functions:
comp <- microbenchmark(f1(), f2(), f3(), f4(), times = 15)

# Create the violin plots:
autoplot(comp)
# ---------------------------------------------------------------------


N <- 2e4
   
# Method 1: using append():
system.time ({
    lst <- list()
    for(i in 1:N) {lst <- append(lst, pi)}
   })
# ---------------------------------------------------------------------


# Method 2: increasing length while counting with length():
system.time ({
    lst <- list()
    for(i in 1:N) {
        lst[[length(lst) + 1]] <- pi}
   })
# ---------------------------------------------------------------------


# Method 3: increasing length using a counter:
system.time ({
    lst <- list()
    for(i in 1:N) {lst[[i]] <- pi}
   })
# ---------------------------------------------------------------------


# Method 4: pre-allocate memory:
system.time({
    lst <- vector("list", N)
    for(i in 1:N) {lst[[i]] <- pi}
    })
# ---------------------------------------------------------------------


N <- 500
# simple operations on a matrix
M <- matrix(1:36, nrow = 6)
system.time({for (i in 1:N) {x1 <- t(M); x2 <- M + pi}})

# simple operations on a data-frame
D <- as.data.frame(M)
system.time({for (i in 1:N) {x1 <- t(D); x2 <- D + pi}})
# ---------------------------------------------------------------------


x <- 1:1e4
N <- 1000
system.time({for (i in 1:N) mean(x)})
system.time({for (i in 1:N) sum(x) / length(x)})
# ---------------------------------------------------------------------


N <- 732
# Use ts from stats to create the a time series object:
t1 <- stats::ts(rnorm(N), start = c(2000,1), end = c(2060,12), 
                frequency = 12)
t2 <- stats::ts(rnorm(N), start = c(2010,1), end = c(2050,12), 
                frequency = 12)

# Create matching zoo and xts objects:
zoo_t1 <- zoo::zoo(t1)
zoo_t2 <- zoo::zoo(t2)
xts_t1 <- xts::as.xts(t1)
xts_t2 <- xts::as.xts(t2)

# Run a merge on them:
# Note that base::merge() is a dispatcher function.
system.time({zoo_t <- merge(zoo_t1, zoo_t2)})
system.time({xts_t <- merge(xts_t1, xts_t2)})

# Calculate the lags:
system.time({for(i in 1:100) lag(zoo_t1)})
system.time({for(i in 1:100) lag(xts_t1)})
# ---------------------------------------------------------------------


# standard function:
f1 <- function(n, x = pi) for(i in 1:n) x = 1 / (1+x)

# using curly brackets:
f2 <- function(n, x = pi) for(i in 1:n) x = 1 / {1+x}

# adding unnecessary round brackets:
f3 <- function(n, x = pi) for(i in 1:n) x = (1 / (1+x))

# adding unnecessary curly brackets:
f4 <- function(n, x = pi) for(i in 1:n) x = {1 / {1+x}}

# achieving the same result by raising to a power
f5 <- function(n, x = pi) for(i in 1:n) x = (1+x)^(-1)

# performing the power with curly brackets
f6 <- function(n, x = pi) for(i in 1:n) x = {1+x}^{-1}

N <- 1e6
library(microbenchmark)
comp <- microbenchmark(f1(N), f2(N), f3(N), f4(N), f5(N), f6(N), 
                       times = 150)

comp
## Unit: milliseconds
##   expr      min       lq     mean   median       uq      max  ...
##   f1(N) 37.37476 37.49228 37.76950 37.57212 37.79876 39.99120 ...
##   f2(N) 37.29297 37.50435 37.79612 37.63191 37.81497 41.09414 ...
##   f3(N) 37.96886 38.18751 38.59619 38.28713 38.68162 47.66612 ...
##   f4(N) 37.88111 38.06787 38.41134 38.16297 38.36706 42.53103 ...
##   f5(N) 45.12742 45.31632 45.67364 45.45465 45.69882 49.65297 ...
##   f6(N) 45.93406 46.03159 46.51151 46.15287 46.64509 52.95426 ...

# Plot the results:
library(ggplot2)
autoplot(comp)
# ---------------------------------------------------------------------


library(compiler)
N <- as.double(1:1e7)

# Create a *bad* function to calculate mean:
f <- function(x) {
  xx = 0
  l = length(x)
  for(i in 1:l)
    xx = xx + x[i]/l
  xx
}

# Time the function:
system.time(f(N))
##   user  system elapsed 
##  0.61    0.000    0.61 

# Compile the function:
cmp_f <- cmpfun(f)

# Time the compiled version
system.time(cmp_f(N))
##  user  system elapsed 
##  0.596  0.00    0.596 
# ---------------------------------------------------------------------


# The difference is small, so we use microbenchmark
library(microbenchmark)
comp <- microbenchmark(f(N), cmp_f(N), times = 150)

# See the results:
comp
## Unit: milliseconds
##      expr      min       lq     mean   median       uq      ...
##      f(N) 552.2785 553.9911 559.6025 556.1511 562.7207 601.5...
##  cmp_f(N) 552.2152 553.9453 558.1029 555.8457 560.0771 588.4...


# Plot the results.
library(ggplot2)
autoplot(comp)
# ---------------------------------------------------------------------


options(R_COMPILE_PKGS = 1)
# ---------------------------------------------------------------------


options(R_ENABLE_JIT = 0)
# ---------------------------------------------------------------------


# Naive implementation of the Fibonacci numbers in R:
Fib_R <- function (n) {
  if ((n == 0) | (n == 1)) return(1)
  return (Fib_R(n - 1) + Fib_R(n - 2))
  }

# The R-function compiled via cmpfun():
library(compiler)
Fib_R_cmp <- cmpfun(Fib_R)

# Using native C++ via cppFunction():
Rcpp::cppFunction('int Fib_Cpp(int n) {
  if ((n == 0) || (n == 1)) return(1);
  return (Fib_Cpp(n - 1) + Fib_Cpp(n - 2));
}')

library(microbenchmark)
N <- 30
comp <- microbenchmark(Fib_R(N), Fib_R_cmp(N), 
                       Fib_Cpp(N), times = 25)

comp
## Unit: milliseconds
##          expr         min          lq        mean      median          uq
##      Fib_R(N) 1449.755022 1453.320560 1474.679341 1456.202559 1472.447928
##  Fib_R_cmp(N) 1444.145773 1454.127022 1489.742750 1459.170600 1554.450501
##    Fib_Cpp(N)    2.678766    2.694425    2.729571    2.711567    2.749208
##          max neval cld
##  1596.226483    25   b
##  1569.764246    25   b
##     2.858784    25  a 

library(ggplot2)
autoplot(comp)
# ---------------------------------------------------------------------


# Efficient function to calculate the Fibonacci numbers in R:
Fib_R2 <- function (n) {
 x = 1
 x_prev = 1
 for (i in 2:n) {
   x <- x + x_prev
   x_prev = x
   }
 x
 }

# Efficient function to calculate the Fibonacci numbers in C++:
Rcpp::cppFunction('int Fib_Cpp2(int n) {
  int x = 1, x_prev = 1, i;
  for (i = 2; i <= n; i++) {
    x += x_prev;
    x_prev = x;
    }
  return x;
  }')
# ---------------------------------------------------------------------


# Test the performance of all the functions:
N <- 30
comp <- microbenchmark(Fib_R(N), Fib_R2(N),
                       Fib_Cpp(N), Fib_Cpp2(N), 
		       times = 20)

comp
## Unit: microseconds
##         expr         min           lq         mean     median ...
##     Fib_R(N) 1453850.637 1460021.5865 1.495407e+06 1471455.852...
##    Fib_R2(N)       2.057       2.4185 1.508404e+02       4.792...
##   Fib_Cpp(N)    2677.347    2691.5255 2.757781e+03    2697.519...
##  Fib_Cpp2(N)       1.067       1.4405 5.209175e+01       2.622...
##          max neval cld
##  1603991.462    20   b
##     2925.070    20  a 
##     3322.077    20  a 
##      964.378    20  a 

library(ggplot2)
autoplot(comp)
# ---------------------------------------------------------------------


sourceCpp("/path/cppSource.cpp")
# ---------------------------------------------------------------------


Rprof("/path/to/my/logfile")
... code goes here
Rprof(NULL)
# ---------------------------------------------------------------------


f0 <- function() x <- pi^2
f1 <- function() x <- pi^2 + exp(pi)
f2 <- function(n) for (i in 1:n) {f0(); f1()}
f3 <- function(n) for (i in 1:(2*n)) {f0(); f1()}
f4 <- function(n) for (i in 1:n) {f2(n); f3(n)}
# ---------------------------------------------------------------------


# Start the profiling:
Rprof("prof_f4.txt")

# Run our functions:
N <- 500
f4(N)

# Stop the profiling process:
Rprof(NULL)
# ---------------------------------------------------------------------


# show the summary:
summaryRprof("prof_f4.txt")
# ---------------------------------------------------------------------


N <- 1000
require(profr)
pr <- profr({f4(N)}, 0.01)
plot(pr)
# ---------------------------------------------------------------------


library(proftools)

# Read in the existing profile data from Rprof:
pd <- readProfileData("prof_f4.txt")

# Print the hot-path:
hotPaths(pd, total.pct = 10.0)

# A flame-graph (stacked time-bars: first following plot)
flameGraph(pd)

# Callee tree-map (intersecting boxes with area related to time
# spent in a function: see plot below)
calleeTreeMap(pd)
# ---------------------------------------------------------------------


unlink("prof_f4.txt")
