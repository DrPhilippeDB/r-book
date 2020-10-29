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


 # Addition:
 2 + 3
 
 # Product:
 2 * 3
 
 # Power:
 2**3
 2^3
 
 # Logic:
 2 < 3
 x <- c(1,3,4,3)
 x.mean <- mean(x)
 x.mean
 y <- c(2,3,5,1)
 x+y
# ---------------------------------------------------------------------


x <- scan()
# ---------------------------------------------------------------------


edit(x)
# ---------------------------------------------------------------------


my_function <- function(a,b)
{
  a +  b
}
# ---------------------------------------------------------------------


# x.1 is assigned the value 5:
x.1 <- 5

# The result of x.1 + 3 is stored in .x:
x.1 + 3 -> .x

# Show the result:
print(.x)
# ---------------------------------------------------------------------


x.3 = 3.14
x.3
# ---------------------------------------------------------------------


v1 <- c(1, 2, 3, NA)
mean(v1, na.rm = TRUE)
# ---------------------------------------------------------------------


# List all variables:
ls()                   # hidden variable starts with dot
ls(all.names = TRUE)   # shows all

# Remove a variable:
rm(x.1)                # removes the variable x.1
ls()                   # x.1 is not there any more
rm(list = ls())        # removes all variables
ls()
# ---------------------------------------------------------------------


# Booleans can be TRUE or FALSE:
x <- TRUE
class(x)

# Integers use the letter L (for Long integer):
x <- 5L
class(x)

# Decimal numbers, are referred to as 'numeric':
x <- 5.135
class(x)

# Complex numbers use the letter i (without multiplication sign):
x <- 2.2 + 3.2i
class(x)

# Strings are called 'character':
x <- "test"
class(x)
# ---------------------------------------------------------------------


# Avoid this:
x <- 3L      # x defined as integer
x
x <- "test"  # R changes data type
x
# ---------------------------------------------------------------------


<<csv1,echo=FALSE,results='hide'>>
# The function as.Data coerces its argument to a date:
d <- as.Date(c("1852-05-12", "1914-11-5", "2015-05-01"))

# Dates will work as expected:
d_recent <- subset(d, d > as.Date("2005-01-01"))
print(d_recent)
# ---------------------------------------------------------------------


x <- c(2, 2.5, 4, 6)
y <- c("apple", "pear")
class(x)
class(y)
# ---------------------------------------------------------------------


# Create v as a vector of the numbes one to 5:
v <- c(1:5)

# Access elements via indexing:
v[2]
v[c(1,5)]

# Access via TRUE/FALSE:
v[c(TRUE,TRUE,FALSE,FALSE,TRUE)]

# Access elements via names:
v <- c("pear" = "green", "banana" = "yellow", "coconut" = "brown")
v
v["banana"]

# Leave out certain elements:
v[c(-2,-3)]
# ---------------------------------------------------------------------


# Define two vectors:
v1 <- c(1,2,3)
v2 <- c(4,5,6)

# Standard arithmetic:
v1 + v2
v1 - v2
v1 * v2
# ---------------------------------------------------------------------


# Define a short and long vector:
v1 <- c(1, 2, 3, 4, 5)
v2 <- c(1, 2)

# Note that R 'recycles' v2 to match the length of v1:
v1 + v2
# ---------------------------------------------------------------------


# Example 1:
v1 <- c(1, -4, 2, 0, pi)
sort(v1)

# Example 2: To make sorting meaningful, all variables are coerced to 
# the most complex type:
v1 <- c(1:3, 2 + 2i)
sort(v1)

# Sorting is per increasing numerical or alphabetical order:
v3 <- c("January", "February", "March", "April")
sort(v3)

# Sort order can be reversed:
sort(v3, decreasing = TRUE)
# ---------------------------------------------------------------------


# Create a matrix:
M <- matrix( c(1:6), nrow = 2, ncol = 3, byrow = TRUE)

# Show it on the screen:
print(M)

M <- matrix( c(1:6), nrow = 2, ncol = 3, byrow = FALSE)
print(M)
# ---------------------------------------------------------------------


# Unit vector:
matrix (1, 2, 1)

# Zero matrix or vector:
matrix (0, 2, 2)

# Recycling also works for shorter vectors:
matrix (1:2, 4, 4)

# Fortunately, R expects that the vector fits exactly n times in the matrix:
matrix (1:3, 4, 4)
# So, the previous was bound to fail.
# ---------------------------------------------------------------------


# Store the names in a vector:
row_names = c("row1", "row2", "row3", "row4")
col_names = c("col1", "col2", "col3")

# Create the matrix:
M <- matrix(c(10:21), nrow = 4, byrow = TRUE, 
            dimnames = list(row_names, col_names))
            
# Display the result:
print(M)
# ---------------------------------------------------------------------


colnames(M) <- c('C1', 'C2', 'C3')
rownames(M) <- c('R1', 'R2', 'R3', 'R4')
M
# ---------------------------------------------------------------------


M <- matrix(c(10:21), nrow = 4, byrow = TRUE)
M

# Access one element:
M[1,2]

# The second row:
M[2,]

# The second column:
M[,2]

# Row 1 and 3 only:
M[c(1, 3),]

# Row 2 to 3 with column 3 to 1:
M[2:3, 3:1]
# ---------------------------------------------------------------------


M1 <- matrix(c(10:21), nrow = 4, byrow = TRUE)
M2 <- matrix(c(0:11),  nrow = 4, byrow = TRUE)
M1 + M2
M1 * M2
M1 / M2
# ---------------------------------------------------------------------


# Example of the dot-product:
a <- c(1:3)
a %*% a
a %*% t(a)
t(a) %*% a

# Define A:
A <- matrix(0:8, nrow = 3, byrow = TRUE)

# Test products:
A %*% a
A %*% t(a) # this is bound to fail!
A %*% A
# ---------------------------------------------------------------------


A %/% A 
# ---------------------------------------------------------------------


# Note the difference between the normal product:
A * A 

# and the matrix product %*%:
A %*% A
# ---------------------------------------------------------------------


# However, there is -of course- only one sum:
A + A
# ---------------------------------------------------------------------


# Note that the quotients yield almost the same:
A %/% A

A / A
# ---------------------------------------------------------------------


# This is the matrix A:
A

# The exponential of A:
exp(A)
# ---------------------------------------------------------------------


# The natural logarithm
log(A)
sin(A)
# ---------------------------------------------------------------------


# Collapse to a vectore:
colSums(A)
rowSums(A)

# Some functions aggregate the whole matrix to one scalar:
mean(A)
min(A)
# ---------------------------------------------------------------------


M <- matrix(c(1,1,4,1,2,3,3,2,1), 3, 3)
M

# The diagonal of M:
diag(M)

# Inverse:
solve(M)

# Determinant:
det(M)

# The QR composition:
QR_M <- qr(M)
QR_M$rank

# Number of rows and columns:
nrow(M)
ncol(M)

# Sums of rows and columns:
colSums(M)
rowSums(M)

# Means of rows, columns, and matrix:
colMeans(M)
rowMeans(M)
mean(M)

# Horizontal and vertical concatenation:
rbind(M, M)
cbind(M, M)

# ---------------------------------------------------------------------


# Create an array:
a <- array(c('A','B'),dim = c(3,3,2))
print(a)

# Access one element:
a[2,2,2]

# Access one layer:
a[,,2]
# ---------------------------------------------------------------------


# Create two vectors:
v1 <- c(1,1)
v2 <- c(10:13)
col.names <- c("col1","col2", "col3")
row.names <- c("R1","R2")
matrix.names <- c("Matrix1","Matrix2")

# Take these vectors as input to the array.
a <- array(c(v1,v2),dim = c(2,3,2),
     dimnames = list(row.names,col.names, 
     matrix.names))
print(a)

# This allows to address the first row in Matrix 1 as follows:
a['R1',,'Matrix1']
# ---------------------------------------------------------------------


<<arrayChunk,echo=FALSE, results='hide'>>
M1 <- a[,,1]
M2 <- a[,,2]
M2
# ---------------------------------------------------------------------


x <- cbind(x1 = 3, x2 = c(4:1, 2:5))
dimnames(x)[[1]] <- letters[1:8]
apply(x, 2, mean, trim = .2)
col.sums <- apply(x, 2, sum)
row.sums <- apply(x, 1, sum)
rbind(cbind(x, Rtot = row.sums), 
      Ctot = c(col.sums, sum(col.sums)))
# ---------------------------------------------------------------------


# Re-create the array a (shorter code):
col.names <- c("col1","col2", "col3")
row.names <- c("R1","R2")
matrix.names <- c("Matrix1","Matrix2")
a <- array(c(1,1,10:13),dim = c(2,3,2),
     dimnames = list(row.names,col.names, 
     matrix.names))

# Demonstrate apply:
apply(a, 1, sum)
apply(a, 2, sum)
apply(a, 3, sum)
# ---------------------------------------------------------------------


# Create a list with the list() function:
myList <- list("Approximation", pi, 3.14, c)

# Display the result:
print(myList)
# ---------------------------------------------------------------------


# Create the list:
L <- list("Approximation", pi, 3.14, c)

# Assign names to elements:
names(L) <- c("description", "exact", "approx","function")

# Show the result:
print(L)
# ---------------------------------------------------------------------


# Addressing elements of the named list:
print(paste("The difference is", L$exact - L$approx))
print(L[3])
print(L$approx)

# However, "function" was a reserved word, so we need to use
# back-ticks in order to address the element:
a <- L$`function`(2,3,pi,5)  # to access the function c(...)
print(a)
# ---------------------------------------------------------------------


# Start with a vector:
V1 <- c(1,2,3)

# Define two lists:
L2 <- list(V1, c(2:7))
L3 <- list(L2,V1)

# Show the results:
print(L3)
print(L3[[1]][[2]][3])
# ---------------------------------------------------------------------


# The first object of L2 as a list:
L2[1]
class(L[2])

# The first element of L2 is a numeric vector:
L2[[1]]
class(L2[[2]])
# range
L2[1:2]

# Unexpected result (ranges are not to be used with x[[.]]):
L2[[1:2]] <- 'a'
L2

# Is this what you would expect?
L2[1:2] <- 'a'
L2
# ---------------------------------------------------------------------


# Define a simple list:
L <- list(1,2)

# Coerce the fourth position to 4:
L[4] <- 4  # position 3 is NULL

# Show the results:
L
# ---------------------------------------------------------------------


L$pi_value <- pi
L
# ---------------------------------------------------------------------


L[1] <- NULL
L
# ---------------------------------------------------------------------


L <- L[-2]
L
# ---------------------------------------------------------------------


# The list:
L  <- list(c(1:5), c(6:10))

# The vectors obtained from the list:
v1 <- unlist(L[1])
v2 <- unlist(L[2])

# Show the results:
v1
v2
v2-v1
# ---------------------------------------------------------------------


# A list of vectors of integers:
L <- list(1L,c(-10L:-8L))
unlist(L)

# Note the named real-valued extra element:
L <- list(c(1:2),c(-10:-8), "pi_value" = pi)
unlist(L)
# ---------------------------------------------------------------------


# Create a vector containing all your observations:
feedback <- c('Good','Good','Bad','Average','Bad','Good')

# Create a factor object:
factor_feedback <- factor(feedback)

# Print the factor object:
print(factor_feedback)
# ---------------------------------------------------------------------


# Plot the histogram -- note the default order is alphabetic
plot(factor_feedback)
# ---------------------------------------------------------------------


# The nlevels function returns the number of levels:
print(nlevels(factor_feedback))
# ---------------------------------------------------------------------


# Store the survey results:
feedback <- c('Good','Good','Bad','Average','Bad','Good')

# Define the factors while providing the levels in right order:
factor_feedback <- factor(feedback,
                          levels = c("Bad", "Average", "Good"))
                          
# Display results:
plot(factor_feedback)
# ---------------------------------------------------------------------


gl(3,2,,c("bad","average","good"),TRUE)
# ---------------------------------------------------------------------


# Create the data frame.
data_test <- data.frame(
   Name   = c("Piotr", "Pawel","Paula","Lisa","Laura"), 
   Gender = c("Male", "Male","Female", "Female","Female"), 
   Score  = c(78,88,92,89,84), 
   Age    = c(42,38,26,30,35)
   )
print(data_test)

# The standard plot function on a data-frame (Figure 4.3)
# is the same as using the pairs() function:
plot(data_test)
# ---------------------------------------------------------------------


<<frame1,echo=FALSE, results='hide'>>
# Get the structure of the data frame:
str(data_test)
# Note that the names became factors.
# ---------------------------------------------------------------------


# Get the summary of the data frame:
summary(data_test)
# ---------------------------------------------------------------------


# Get the first rows:
head(data_test)
# ---------------------------------------------------------------------


# Get the last rows:
tail(data_test)
# ---------------------------------------------------------------------


# Extract the column 2 and 4 and keep all rows
data_test.1 <- data_test[,c(2,4)]
print(data_test.1)
# ---------------------------------------------------------------------


# Extract columns by name and keep only selected rows
data_test[c(2:4),c(2,4)]
# ---------------------------------------------------------------------


d <- data.frame(
   Name   = c("Piotr", "Pawel","Paula","Lisa","Laura"), 
   Gender = c("Male", "Male","Female", "Female","Female"), 
   Score  = c(78,88,92,89,84), 
   Age    = c(42,38,26,30,35),
   stringsAsFactors = FALSE
   )
d$Gender <- factor(d$Gender)  # manually factorize gender
str(d)
# ---------------------------------------------------------------------


de(x)                  # fails if x is not defined
de(x <- c(NA))         # works
x <- de(x <- c(NA))    # will also save the changes in x
data.entry(x)          # de is short for data.entry
x <- edit(x)           # use the standard editor (vi in *nix)
# ---------------------------------------------------------------------


# The following lines do the same.
data_test$Score[1] <- 80
data_test[3,1]     <- 80
# ---------------------------------------------------------------------


# Expand the data frame, simply define the additional column:
data_test$End_date <-  as.Date(c("2014-03-01", "2017-02-13",
                   "2014-10-10", "2015-05-10","2010-08-25"))
print(data_test)
# ---------------------------------------------------------------------


# Or use the function cbind() to combine data frames along columns:
Start_date <- as.Date(c("2012-03-01", "2013-02-13",
                   "2012-10-10", "2011-05-10","2001-08-25"))

# Use this vector directly:
df <- cbind(data_test, Start_date)
print(df)

# or first convert to a data frame:
df <- data.frame("Start_date" = t(Start_date))
df <- cbind(data_test, Start_date)
print(df)
# ---------------------------------------------------------------------


# To add a row, we need the rbind() function:
data_test.to.add <- data.frame(
   Name = c("Ricardo", "Anna"), 
   Gender = c("Male", "Female"), 
   Score = c(66,80), 
   Age = c(70,36),
   End_date = as.Date(c("2016-05-05","2016-07-07"))
   )
data_test.new <- rbind(data_test,data_test.to.add)
print(data_test.new)
# ---------------------------------------------------------------------


data_test.1 <- data.frame(
   Name = c("Piotr", "Pawel","Paula","Lisa","Laura"), 
   Gender = c("Male", "Male","Female", "Female","Female"), 
   Score = c(78,88,92,89,84), 
   Age = c(42,38,26,30,35)
   )
data_test.2 <- data.frame(
   Name = c("Piotr", "Pawel","notPaula","notLisa","Laura"), 
   Gender = c("Male", "Male","Female", "Female","Female"), 
   Score = c(78,88,92,89,84), 
   Age = c(42,38,26,30,135)
   )
data_test.merged <- merge(x=data_test.1,y=data_test.2,
                          by.x=c("Name","Age"),by.y=c("Name","Age"))

# Only records that match in name and age are in the merged table:
print(data_test.merged)
# ---------------------------------------------------------------------


# Use 'N' to refer to 'Name'
data_test$N
# ---------------------------------------------------------------------


# Get the rownames:
colnames(data_test)

# Access the rownames:
rownames(data_test)
colnames(data_test)[2]
rownames(data_test)[3]

# Assign new names:
colnames(data_test)[1] <- "first_name"
rownames(data_test) <- LETTERS[1:nrow(data_test)]
print(data_test)
# ---------------------------------------------------------------------


a <- "Hello"
b <- "world"
paste(a, b, sep = ", ")
c <- "A 'valid' string"
# ---------------------------------------------------------------------


paste0(12, '%')
# ---------------------------------------------------------------------


a <- format(100000000,big.mark=" ",
                 nsmall=3,
                 width=20,
                 scientific=FALSE,
                 justify="r")
print(a)
# ---------------------------------------------------------------------


v1 <- c(2,4,6,8)
v2 <- c(1,2,3,5)
v1 + v2     # addition
v1 - v2     # subtraction
v1 * v2     # multiplication
v1 / v2     # division
v1 %% v2    # remainder of division
v1 %/% v2   # round(v1/v2 -0.5)
v1 ^ v2     # v1 to the power of v2
# ---------------------------------------------------------------------


v1 %*% v2
# ---------------------------------------------------------------------


v1 <- c(8,6,3,2)
v2 <- c(1,2,3,5)
v1 > v2     # bigger than
v1 < v2     # smaller than
v1 <= v2    # smaller or equal
v1 >= v2    # bigger or equal
v1 == v2    # equal
v1 != v2    # not equal
# ---------------------------------------------------------------------


# The vectors:
v1 <- c(TRUE, TRUE, FALSE, FALSE)
v2 <- c(TRUE, FALSE, FALSE, TRUE)

# The basic logical operations:
v1 & v2     # and
v1 | v2     # or
!v1         # not
v1 && v2    # and applied to the first element
v1 || v2    # or applied to the first element
# ---------------------------------------------------------------------


# More aspects of logical values:
v1 <- c(TRUE, FALSE, TRUE, FALSE, 8, 6+3i, -2, 0, NA)

class(v1)  # v1 is a vector or complex numbers
v2 <- c(TRUE)
as.logical(v1)  # coerce to logical (only 0 is FALSE)
v1 & v2
v1 | v2
# ---------------------------------------------------------------------


0 & TRUE
-1 & pi & 1
0 + 5i & TRUE
# ---------------------------------------------------------------------


FALSE | NA
TRUE  | NA
FALSE & NA
TRUE  & NA
FALSE | NA | TRUE | TRUE
TRUE  & NA & FALSE
# ---------------------------------------------------------------------


# left assignment
x <- 3 
x = 3
x <<- 3

# right assignment
3 -> x
3 ->> x

#chained assignment
x <- y <- 4
# ---------------------------------------------------------------------


mean(v1, na.rm = TRUE)  # works (v1 is defined in previous section)
mean(v1, na.rm <- TRUE) # fails
# ---------------------------------------------------------------------


# f
# Assigns in the current and superior environment 10 to x,
# then prints it, then makes it 0 only in the function environment
# and prints it again.
# arguments:
#   x   -- numeric
f <- function(x) {x <<- 10; print(x); x <- 0; print(x)}

x <- 3
x

# Run the function f():
f(x)

# Only the value assigned with <<- is available now:
x
# ---------------------------------------------------------------------


# +-+
# This function is a new operator
# arguments:
#   x -- numeric
#   y -- numeric
# returns:
#   x - y
`+-+` <- function(x, y) x - y
5 +-+ 5
5 +-+ 1

# Remove the new operator:
rm(`+-+`)
# ---------------------------------------------------------------------


# create a list
x <- c(10:20)
x

# %in% can find an element in a vector
2  %in% x   # FALSE since 2 is not an element of x
11 %in% x   # TRUE since 11 is in x
x[x %in% c(12,13)] # selects elements from x
x[2:4]   # selects the elements with index
            # between 2 and 4
# ---------------------------------------------------------------------


# %*% the matrix multiplication (or crossproduct)
M = matrix(c(1,2,3,7,8,9,4,5,6), nrow = 3,ncol = 3,
           byrow = TRUE)
M %*% t(M)
M %*% M 
exp(M)
# ---------------------------------------------------------------------


if (logical statement) {
  executed if logical statement is true
} else {
  executed if the logical statement if false
}
# ---------------------------------------------------------------------


set.seed(1890)
x <- rnorm(1)
if (x < 0) {
  print('x is negative') 
} else if (x > 0) {
  print('x is positive')
} else {
  print('x is zero')
}
# ---------------------------------------------------------------------


  x <- 122
  if (x < 10) {
    print('less than ten')
    } else if (x < 100) {
    print('between 10 and 100')
    } else if (x < 1000) {
    print('between 100 and 1000')
    } else {
    print('bigger than 1000 (or equal to 1000)')
    }
# ---------------------------------------------------------------------


x <- TRUE
y <- pi
y <- if (x) 1 else 2
y  # y is now 1
# ---------------------------------------------------------------------


z <- 0
y <- if (x) {1; z <- 6} else 2
y  # y is now 6
z  # z is also 6
# ---------------------------------------------------------------------


x <- 1:6
ifelse(x %% 2 == 0, 'even', 'odd')
# ---------------------------------------------------------------------


x <- 1:6
y <- LETTERS[1:3]
ifelse(x %% 2 == 0, 'even', y)
# Note that y gets recycled! 
# ---------------------------------------------------------------------


x <- 'b'
x_info <- switch(x,
    'a' = "One",
    'b' = "Two",
    'c' = "Three",
    stop("Error: invalid `x` value")
  )
# x_info should be 'two' now:
x_info
# ---------------------------------------------------------------------


x <- 'b'
x_info <- if (x == 'a' ) {
    "One"
  } else if (x == 'b') {
    "Two"
  } else if (x == 'c') {
    "Three" 
  } else {
    stop("Error: invalid `x` value")
  }
# x_info should be 'two' now:
x_info
# ---------------------------------------------------------------------


for (value in vector) {
   statements
}
# ---------------------------------------------------------------------


x <- LETTERS[1:5]
for ( j in x) {
   print(j)
}
# ---------------------------------------------------------------------


x <- c(0, 1, -1, 102, 102)
for ( j in x) {
   print(j)
}
# ---------------------------------------------------------------------


repeat { 
   commands 
   if(condition) {break}
}
# ---------------------------------------------------------------------


x <- c(1,2)
c <- 2
repeat {
   print(x+c)
   c <- c+1
   if(c > 4) {break}
}
# ---------------------------------------------------------------------


while (test_expression) {
   statement
}
# ---------------------------------------------------------------------


x <- c(1,2); c <- 2
while (c < 4) {
   print(x+c)
   c <- c + 1
}
# ---------------------------------------------------------------------


v <- c(1:5)
for (j in v) {
   if (j == 3) {
      print("--break--")
      break
   }
   print(j)
}
# ---------------------------------------------------------------------


v <- c(1:5)
for (j in v) {
   if (j == 3) {
      print("--skip--")
      next
   }
   print(j)
}
# ---------------------------------------------------------------------


n <- 10^7
v1 <- 1:n

# -- using vector arithmetic
t0 <- Sys.time()
v2 <- v1 * 2
t1 <- Sys.time() - t0
print(t1)

# -- using a for loop
rm(v2)
v2 <- c()
t0 <- Sys.time()
for (k in v1) v2[k] <- v1[k] * 2
t2 <- Sys.time() - t0
print(t2)

# t1 and t2 are difftime objects and only differences 
# are defined.
# To get the quotient, we need to coerce them to numeric.
T_rel <- as.numeric(t2, units = "secs") / 
         as.numeric(t1, units = "secs")
T_rel
# ---------------------------------------------------------------------


help(c)   # shows help help with the function c
?c        # same result

apropos("cov") # fuzzy search for functions
# ---------------------------------------------------------------------


function_name <- function(arg_1, arg_2, ...) {
   function_body 
   return_value
}
# ---------------------------------------------------------------------


# c_surface
# Calculates the surface of a circle
# Arguments:
#   radius -- numeric, the radius of the circle
# Returns
#   the surface of the cicle
c_surface <- function(radius) {
  x <- radius ^ 2 * pi
  return (x)
  }
  
# Test the function:
c_surface(2) 
# ---------------------------------------------------------------------


# c_surface
# Calculates the surface of a circle
# Arguments:
#   radius -- numeric, the radius of the circle
# Returns
#   the surface of the cicle
c_surface <- function(radius) {
  radius ^ 2 * pi
  }

# Test the function:
c_surface(2) 
# ---------------------------------------------------------------------


# Edit the function with vi:
 fix(c_surface)

# Or us edit:
 c_surface <- edit()
# ---------------------------------------------------------------------


c_surface <- function(radius = 2) {
  radius ^ 2 * pi
  }
c_surface(1)
c_surface()
# ---------------------------------------------------------------------


# Download the package (only once):
install.packages('DiagrammeR')

# Load it before we can use it (once per session):
library(DiagrammeR)
# ---------------------------------------------------------------------


# See the path where libraries are stored:
.libPaths()

# See the list of installed packages:
library()

# See the list of currently loaded packages:
search()
# ---------------------------------------------------------------------


# available.packages() gets a list:
pkgs <- available.packages(filters = "duplicates")
colnames(pkgs)

# We don't need all, just keep the name:
pkgs <- pkgs[,'Package']

# Show the results:
print(paste('Today, there are', nrow(pkgs), 'packages for R.'))
# ---------------------------------------------------------------------


# Get the list (only names):
my_pkgs <- library()$results[,1]

# Show the results:
print(paste('I have', length(my_pkgs), 'packages for R.'))
# ---------------------------------------------------------------------


# See all installed packages:
installed.packages()
# ---------------------------------------------------------------------


# List all out-dated packages:
old.packages()
# ---------------------------------------------------------------------


# Update all available packages:
update.packages()
# ---------------------------------------------------------------------


# Update all packages in batch mode:
update.packages(ask = FALSE)
# ---------------------------------------------------------------------


# Update one package (example with the TTR package):
install.packages("TTR")
# ---------------------------------------------------------------------


t <- readLines(file.choose())
# ---------------------------------------------------------------------


t <- readLines("R.book.txt")
# ---------------------------------------------------------------------


# To read a CSV-file it needs to be in the current directory
# or we need to supply the full path, or go first to the relevant folder.
setwd("./data/") # change working directory
data <- read.csv("eurofxref-hist.csv")
is.data.frame(data)
ncol(data)
nrow(data)
head(data)
hist(data$CAD, col = 'khaki3')
plot(data$USD, data$CAD, col = 'red')
# ---------------------------------------------------------------------


<<inputCSV,echo=FALSE,results='hide'>>
# get the maximum exchange rate
maxCAD <- max(data$CAD)
# use SQL-like selection
d0 <- subset(data, CAD == maxCAD)
d1 <- subset(data, CAD > maxCAD - 0.1)

d1[,1]
# ---------------------------------------------------------------------


d2 <- data.frame(d1$Date,d1$CAD)
d2
hist(d2$d1.CAD, col = 'khaki3')
# ---------------------------------------------------------------------


<<csv1,echo=FALSE,results='hide'>>
write.csv(d2,"output.csv", row.names = FALSE)
new.d2 <- read.csv("output.csv")
print(new.d2)
# ---------------------------------------------------------------------


# install the package xlsx if not yet done
if (!any(grepl("xlsx",installed.packages()))){
  install.packages("xlsx")}
library(xlsx)
data <- read.xlsx("input.xlsx", sheetIndex = 1)
# ---------------------------------------------------------------------


  if(!any(grepl("xls", installed.packages()))){
    install.packages("RMySQL")}
  library(RMySQL)
# ---------------------------------------------------------------------


# The connection is stored in an R object, myConnection, and 
# it needs the database name (db_name), username and password
myConnection = dbConnect(MySQL(), 
   user     = 'root', 
   password = 'xxx', 
   dbname   = 'db_name',
   host     = 'localhost')

# e.g. list the tables available in this database.
dbListTables(myConnection)
# ---------------------------------------------------------------------


# Prepare the query for the database
result <- dbSendQuery(myConnection, 
  "SELECT * from tbl_students WHERE age > 33")

# fetch() will get us the results, it takes a parameter n, which
# is the number of desired records.
# Fetch all the records(with n = -1) and store it in a data frame.
data <- fetch(result, n = -1)
# ---------------------------------------------------------------------


sSQL = ""
sSQL[1] <- "UPDATE tbl_students 
            SET score = 'A' WHERE raw_score > 90;"
sSQL[2] <- "INSERT INTO tbl_students 
            (name, class, score, raw_score) 
	    VALUES ('Robert', 'Grade 0', 88,NULL);"
sSQL[3] <- "DROP TABLE IF EXISTS tbl_students;"
for (k in c(1:3)){
  dbSendQuery(myConnection, sSQL[k])
  }
# ---------------------------------------------------------------------


dbWriteTable(myConnection, "tbl_name", 
             data_frame_name[, ], overwrite = TRUE)
# ---------------------------------------------------------------------


environment()  # get the environment
rm(list = ls())  # clear the environment
ls()           # list all objects
a <- "a"
f <- function (x) print(x)
ls()           # note that x is not part of.GlobalEnv
# ---------------------------------------------------------------------


# f
# Multiple actions and side effects to illustrate environments
# Arguments:
#   x -- single type
f <- function(x){
      # define a local function g() within f()
      g <- function(y){
	      b <- "local variable in g()"
              print(" -- function g() -- ")
              print(environment())
              print(ls())
	      print(paste("b is", b))
	      print(paste("c is", c))
              }
              
     # actions in function f:
     a <<- 'new value for a, also in Global_env'
     x <- 'new value for x'
     b <- d    # d is taken from the environment higher
     c <- "only defined in f(), not in g()"
     g("parameter to g")
     print(" -- function f() -- ")
     print(environment())
     print(ls())
     print(paste("a is", a))
     print(paste("b is", b))
     print(paste("c is", c))
     print(paste("x is", x))
     }

# Test the function f():
b <- 0
a <- c <- d <-  pi
rm(x)

f(a)

# Check the existence and values:
a     # a is also changed in Global_env by f()
b     # b had another value in g() but is not changed in Global_env
c     # the value of b is not influenced by f()
x     # x was only defined in f() and hence is not available anymore
# ---------------------------------------------------------------------


x <- 'Philippe'
rm(x)       # make sure the definition is removed
x           # x is indeed not there (generates an error message)
x <- 2L     # now x is created as a long integer
x <- pi     # coerced to double (real)
x <- c(LETTERS[1:5]) # now it is a vector of strings
# ---------------------------------------------------------------------


# f
# Demonstrates the scope of variables
f <- function() {
  a <- pi     # define local variable
  print(a)    # print the local variable
  print(b)    # b is not in the scope of the function
}

# Define two variables a and b
a <- 1
b <- 2

# Run the function and note that it knows both a and b.
# For b it cannot find a local definition and hence
# uses the definition of the higher level of scope.
f()

# f() did not change the value of a in the environment that called f():
print(a)
# ---------------------------------------------------------------------


# Citation from the R documentation:
# Copyright (C) 1997-8 The R Core Team
open.account <- function(total) {
     list(
        deposit = function(amount) {
            if(amount <= 0)
                stop("Deposits must be positive!\n")
            total <<- total + amount
            cat(amount,"deposited. Your balance is", total, "\n\n")
        },
        withdraw = function(amount) {
            if(amount > total)
                stop("You do not have that much money!\n")
            total <<- total - amount
            cat(amount,"withdrawn.  Your balance is", total, "\n\n")
        },
        balance = function() {
            cat("Your balance is", total, "\n\n")
        }
        )
 }

ross <- open.account(100)

robert <- open.account(200)

ross$withdraw(30)

ross$balance()

robert$balance()

ross$deposit(50)

ross$balance()

try(ross$withdraw(500)) # no way..
# ---------------------------------------------------------------------


rm(list=ls())  # clear the environment
# ---------------------------------------------------------------------


L <- list(matrix(1:16, nrow=4))
L$data_source <- "mainframe 1"
L$get_data_src <- function(){L$data_source}
print(L$get_data_src())
# ---------------------------------------------------------------------


# Define a string:
acc <- "Philippe"

# Force an attribute, balance, on it:
acc$balance <- 100

# Inspect the result:
acc
# ---------------------------------------------------------------------


# a function build in core R
typeof(mean)
is.primitive(mean)
# user defined function are "closures:
add1 <- function(x) {x+1}
typeof(add1)
is.function(add1)
is.object(add1)
# ---------------------------------------------------------------------


# is.S3
# Determines if an object is S3
# Arguments:
#    x -- an object
# Returns:
#   boolean -- TRUE if x is S3, FALSE otherwise
is.S3 <- function(x){is.object(x) & !isS4(x)}

# Create two test objects:
M  <- matrix(1:16, nrow=4)
df <- data.frame(M)

# Test our new function:
is.S3(M)
is.S3(df)
is.S3(1 + 2i)
# ---------------------------------------------------------------------


# Define a function to check if something is S3:
is.S3 <- function(x){is.object(x) & !isS4(x)}

# A string is a base type and not an non-S3 ojbect:
x <- 'x'
is.S3(x)

# A string with a class attribute is a valid S3 object:
class(x) <- 'myclass'
is.S3(x)
# ---------------------------------------------------------------------


# Define the method print for the class myclass:
print.myclass <- function(x) {print(paste0('Hello, ', x, '.'))}

# Test it, first with the existing myclass object:
print(x)

# Test it on a character base type (string):
print('x')
# ---------------------------------------------------------------------


library(pryr)
otype(M)
otype(df)
otype(df$X1)            # a vector is not S3
df$fac <-factor(df$X4)
otype(df$fac)           # a factor is S3
# ---------------------------------------------------------------------


mean
ftype(mean)
sum
ftype(sum)
# ---------------------------------------------------------------------


apropos("print.")
apropos("mean.")
# ---------------------------------------------------------------------


methods(methods)
methods(mean)
# ---------------------------------------------------------------------


getS3method("print","table")
# ---------------------------------------------------------------------


methods(class = "data.frame")
# ---------------------------------------------------------------------


my_curr_acc <- list("name" = "Philippe", "balance" <- 100)
class(my_curr_acc) <- "account"  # set the class attribute
otype(my_curr_acc)               # it is an S3 object
class(my_curr_acc)               # the class type is defined above

# Note that the class attribute is not visible in the structure:
str(my_curr_acc)                 
# ---------------------------------------------------------------------


my_object <- structure(list(), class = "boringClass")
# ---------------------------------------------------------------------


# print.account
# Print an object of type 'account'
# Arguments:
#   x -- an object of type account
print.account <- function(x){
   print(paste("account holder",x[[1]],sep=": "))
   print(paste("balance       ",x[[2]],sep=": "))
   }
print(my_curr_acc)
# ---------------------------------------------------------------------


# account
# Constructor function for an object of type account
# Arguments:
#    x -- character (the name of the account holder)
#    y -- numeric (the initial balance of the account
# Returns:
#    Error message in console in case of failure.
account <- function(x,y) {
  if (!is.numeric(y))    stop("Balance must be numeric!")
  if (!is.atomic(x))     stop("Name must be atomic!!")
  if (!is.character(x))  stop("Name must be a string!")
  structure(list("name" = x, "balance" = y), class = "account")
}

# create a new instance for Paul:
paul_account <- account("Paul", 200)

# print the object with print.account():
paul_account
# ---------------------------------------------------------------------


class(paul_account) <- "data.frame"
print(paul_account)   # R thinks now it is a data.frame
paul_account[[2]]     # the data is still correct
class(paul_account) <- "account"
print(paul_account)   # back to normal: the class is just an attribute
# ---------------------------------------------------------------------


# add_balance
# Dispatcher function to handle the action of adding a given amount 
# to the balance of an account object.
# Arguments:
#   x      -- account -- the account object 
#   amount -- numeric -- the amount to add to the balance
add_balance <- function(x, amount) UseMethod("add_balance")
# ---------------------------------------------------------------------


# add_balance.account
# Object specific function for an account for the dispatcher 
# function add_balance()
# Arguments:
#   x      -- account -- the account object 
#   amount -- numeric -- the amount to add to the balance
add_balance.account <- function(x, amount) {
   x[[2]] <- x[[2]] + amount; 
   # Note that much more testing and logic can go here
   # It is not so easy to pass a pointer to a function so we 
   # return the new balance:
   x[[2]]}
   
# Test the function:
my_curr_acc <- add_balance(my_curr_acc, 225) 
print(my_curr_acc)
# ---------------------------------------------------------------------


# add_balance.default
# The default action for the dispatcher function add_balance
# Arguments:
#   x      -- account -- the account object 
#   amount -- numeric -- the amount to add to the balance
add_balance.default <- function(x, amount) {
  stop("Object provided not of type account.")
}
# ---------------------------------------------------------------------


# probe
# Dispatcher function
# Arguments:
#    x -- account object
# Returns
#    confirmation of object type
probe <- function(x) UseMethod("probe")

# probe.account
# action for account object for dispatcher function probe()
# Arguments:
#    x -- account object
# Returns
#    confirmation of object "account"
probe.account <- function(x) "This is a bank account"

# probe.default
# action if an incorrect object type is provided to probe()
# Arguments:
#    x -- account object
# Returns
#    error message
probe.default <- function(x) "Sorry. Unknown class"

probe (structure(list(), class = "account"))

# No method for class 'customer', fallback to 'account'
probe(structure(list(), class = c("customer", "account")))

# No method for class 'customer', so falls back to default
probe(structure(list(), class = "customer"))

# Fallback to default for data.frame:
probe(df)         

# Force R to use the account method, by omitting the dispatcher function:
probe.account(df) 

# First create an account object and then test:
my_curr_acc <- account("Philippe", 150) 
probe(my_curr_acc)
# ---------------------------------------------------------------------


# Create the object type Acc to hold bank-accounts:
setClass("Acc", 
  representation(holder       = "character", 
                 branch       = "character",
                 opening_date = "Date"))

# Create the object type Bnk (bank):
setClass("Bnk", 
  representation(name = "character", phone = "numeric"))

# Define current account as a child of Acc:
setClass("CurrAcc", 
  representation(interest_rate = "numeric",
                 balance  = "numeric"), 
  contains = "Acc")
  
# Define investment account as a child of Acc
setClass("InvAcc", 
  representation(custodian = "Bnk"), contains = "Acc")
# ---------------------------------------------------------------------


# Create an instance of Bnk:
my_cust_bank <- new("Bnk",
                    name = "HSBC",
                    phone = 123456789)

# Create an instance of Acc:
my_acc <- new("Acc", 
             holder       = "Philippe", 
             branch       = "BXL12",
             opening_date = as.Date("2018-10-02"))
# ---------------------------------------------------------------------


# Check if it is really an S4 object:
isS4(my_cust_bank)

# Change the phone number and check:
my_cust_bank@phone = 987654321  # change the phone number
print(my_cust_bank@phone)       # check if it changed
# ---------------------------------------------------------------------


# This will do the same as my_cust_bank@phone:
attr(my_cust_bank, 'phone')

# The function also allows partial matching:
attr(my_cust_bank, which='ph', exact = FALSE)

# attr can also change the value of an attribute.
attr(my_cust_bank, which='phone') <- '123123123'
# Let us verify:
my_cust_bank@phone

# It is even possible to create a new attribute or remove one.
attr(my_cust_bank, 'something') <- 'Philippe'
attr(my_cust_bank, 'something')
attr(my_cust_bank, 'something') <- NULL
attr(my_cust_bank, 'something')
str(my_cust_bank) # the something attribute is totally gone
# ---------------------------------------------------------------------


x <- 1:9
x        # x is a vector
class(x)

attr(x, "dim") <- c(3,3)
x        # is is now a matrix!
class(x) # but R is not fooled.
# ---------------------------------------------------------------------


slot(my_acc, "holder")
# ---------------------------------------------------------------------


my_curr_acc <- new("CurrAcc", 
                  holder        = "Philippe", 
                  interest_rate = 0.01, 
                  balance       = 0, 
                  branch        = "LDN12", 
                  opening_date  = as.Date("2018-12-01"))

# Note that the following does not work and is bound to fail:
also_an_account <- new("CurrAcc", 
                       holder        = "Philippe", 
                       interest_rate = 0.01, 
                       balance       = 0, 
                       Acc           = my_acc)
# ---------------------------------------------------------------------


my_curr_acc@balance <- 500
# ---------------------------------------------------------------------


my_inv_acc <- new("InvAcc", 
                  custodian    = my_cust_bank, 
                  holder       = "Philippe", 
                  branch       = "DUB01", 
                  opening_date = as.Date("2019-02-21"))

# note that the first slot is another S4 object:
my_inv_acc
# ---------------------------------------------------------------------


my_inv_acc@custodian        # our custodian bank is HSBC
my_inv_acc@custodian@name   # note the cascade of @ signs
my_inv_acc@custodian@name <- "DB"  # change it to DB
my_inv_acc@custodian@name   # yes, it is changed
my_cust_bank@name           # but our original bank isn't
my_cust_bank@name <- "HSBC Custody" # try something different
my_inv_acc@custodian@name   # did not affect the account
my_inv_acc@custodian@name <- my_cust_bank@name # change back
# ---------------------------------------------------------------------


getSlots("Acc")
# ---------------------------------------------------------------------


# Note the mistake in the following code:
my_curr_acc <- new("CurrAcc", 
                  holder        = "Philippe", 
                  interest_rate = 0.01, 
                  balance       = "0",  # Here is the mistake!
                  branch        = "LDN12", 
                  opening_date  = as.Date("2018-12-01"))
# ---------------------------------------------------------------------


x_account <- new("CurrAcc", 
                  holder        = "Philippe", 
                  interest_rate = 0.01, 
                  #no balance provided
                  branch        = "LDN12", 
                  opening_date  = as.Date("2018-12-01"))
                  
# Show what R did with it:                  
x_account@balance  
# ---------------------------------------------------------------------


# Define the S4 class with a default for balance:
setClass("CurrAcc", 
  representation(interest_rate = "numeric",
                 balance       = "numeric"), 
  contains = "Acc",
  prototype(holder       = NA_character_, 
            interst_rate = NA_real_, 
            balance      = 0))

# Create an instance x_account:
x_account <- new("CurrAcc", 
                  # no holder
                  # no interest rate
                  # no balance
                  branch       = "LDN12", 
                  opening_date = as.Date("2018-12-01"))
                  
# Show the details of the object (note the defaults filled in):
x_account         
# ---------------------------------------------------------------------


# This is constructor function for a current account, it does something similar
# to opening and account in a bank branch:
.CurrAcc <- function (holder,
                    interest_rate
                    # branch we know from the user
                    # balance should be 0
                    # opening_date is today
                    ) {

  error_msg = "Invalid input while creating an account\n"
  if (is.atomic(holder) & !is.character(holder)) {
    stop(error_msg, "Invalid holder name.")
    }
  if (!(is.atomic(interest_rate) & is.numeric(interest_rate)
      & (interest_rate >= 0) & (interest_rate < 0.1))) {
    stop(error_msg, "Interest rate invalid.")
    }
  br <- "PAR01"  # pretending to find balance by looking up user
  dt <- as.Date(Sys.Date())
  new("CurrAcc", 
                  holder = holder, 
                  interest_rate = interest_rate, 
                  balance=0, 
                  branch = br, 
                  opening_date= dt)
  }

# Create a new account:
lisa_curr_acc <- .CurrAcc("Lisa", 0.01)
lisa_curr_acc
# ---------------------------------------------------------------------


# Here is the prototype of a dataset that holds some extra
# information in a structured way.
 setClass("myDataFrame",
          contains = "data.frame",
          slots = list(MySQL_DB   = "character",
                       MySQL_tbl  = "character",
                       data_owner = "character"
                       )
          )
xdf <- new("myDataFrame",
    data.frame(matrix(1:9, nrow=3)),
    MySQL_DB = "myCorporateDB@102.12.12.001",
    MySQL_tbl = "tbl_current_accounts",
    data_owner = "customer relationship team")
   
xdf@.Data   
xdf@data_owner
# ---------------------------------------------------------------------


str(my_inv_acc)
isS4(my_inv_acc)
pryr::otype(my_inv_acc)
# ---------------------------------------------------------------------


is(my_inv_acc)
is(my_inv_acc, "Acc")
# ---------------------------------------------------------------------


is.S3
# ---------------------------------------------------------------------


# setGeneric needs a function, so we need to create it first.

# credit
# Dispatcher function to credit the ledger of an object of 
# type 'account'.
# Arguments:
#    x -- account object
#    y -- numeric -- the amount to be credited
credit <- function(x,y){useMethod()}

# transform our function credit() to a generic one:
setGeneric("credit")

# Add the credit function to the object CurrAcc
setMethod("credit",
   c("CurrAcc"),
   function (x, y) {
     new_bal <- x@balance + y
     new_bal
     }
   )
   
# Test the function:
my_curr_acc@balance
my_curr_acc@balance <- credit(my_curr_acc, 100)
my_curr_acc@balance
# ---------------------------------------------------------------------


# debet
# Generic function to debet an account
# Arguments:
#    x -- account object
#    y -- numeric -- the amount to be taken from the account
# Returns
#    confirmation of action or lack thereof
debet <- function(x,y){useMethod()}

# Make it a generic function that verifies the balance
# before the account a debet is booked.
setGeneric("debet")

# Add the debet() function as a method for objects of type CurrAcc
setMethod("debet",
   c("CurrAcc"),
   function (x, y) {
     if(x@balance >= y) {
       new_bal <- x@balance + y} else {
       stop("Not enough balance.")
       }
     new_bal
     }
   )
   
# Test the construction:
my_curr_acc@balance  # for reference
my_curr_acc@balance <- debet(my_curr_acc, 50)
my_curr_acc@balance  # the balance is changed

# We do not have enough balance to debet 5000, so the 
# following should fail:
my_curr_acc@balance <- debet(my_curr_acc, 5000)
my_curr_acc@balance  # the balance is indeed unchanged:
# ---------------------------------------------------------------------


selectMethod("credit", list("CurrAcc"))
# ---------------------------------------------------------------------


# Note that we capture the returned value of the setRefClass in this generator function.
# Give this always the same name as the class.
account <- setRefClass("account",
            fields = list(ref_number   = "numeric",
                          holder       = "character",
                          branch       = "character",
                          opening_date = "Date",
                          account_type = "character"
                          ),
            # no method yet.
            )
      
# Create an instance:
x_acc <- account$new(ref_number   = 321654987,
                    holder        = "Philippe",
                    branch        = "LDN05",
                    opening_date  = as.Date(Sys.Date()),
                    account_type  = "current"
                    )
                    
# Show the instance:
x_acc
# ---------------------------------------------------------------------


setRefClass("account", fields = c("ref_number", 
                                  "holder",
                                  "branch",
                                  "opening_date",
                                  "account_type"
                                  )
           )
# ---------------------------------------------------------------------


setRefClass("account", 
    fields = list(holder,   # accepts all types
             branch,        # accepts all types
             opening_date = "Date" # dates only
            )
    )
# ---------------------------------------------------------------------


isS4(account)
# account is S4 and it has a lot more than we have defined:
account
# ---------------------------------------------------------------------


account$fields()
account$help()
# ---------------------------------------------------------------------


custBank    <- setRefClass("custBank",
                   fields = list(name =  "character",
                                 phone = "character"
                                 )
                   )
invAccount  <- setRefClass("invAccount",
                   fields = list(custodian = "custBank"),
                   contains = c("account")
                   # methods go here
                   )
# ---------------------------------------------------------------------


# Definition of RC object currentAccount
currAccount <- setRefClass("currentAccount",
                   fields = list(interest_rate = "numeric",
                                 balance       = "numeric"),
                   contains = c("account"),
                   methods = list(
                      credit = function(amnt) {
                            balance <<- balance + amnt
                            },
                      debet =  function(amnt) {
                            if (amnt <= balance) {
                               balance <<- balance - amnt
                               } else {
                               stop("Not rich enough!")
                               }
                         }
                      )
                   )
# note how the class reports on itself:
currAccount
# ---------------------------------------------------------------------


ph_acc <- currAccount$new(ref_number    = 321654987,
                          holder        = "Philippe",
                          branch        = "LDN05",
                          opening_date  = as.Date(Sys.Date()), 
                          account_type  = "current",
                          interest_rate = 0.01,
                          balance       = 0  
                          )
# ---------------------------------------------------------------------


ph_acc$balance     # after creating balance is 0:
ph_acc$debet(200)  # impossible (not enough balance)
ph_acc$credit(200) # add 200 to the acount
ph_acc$balance     # the money arrived in our account
ph_acc$debet(100)  # this is possible
ph_acc$balance     # the money is indeed gone
# ---------------------------------------------------------------------


# Create the class without methods:
alsoCurrAccount <- setRefClass("currentAccount",
                   fields = list(
                             interest_rate = "numeric",
                             balance       = "numeric"),
                   contains = c("account")
                   )
                   
# Add the methods:
alsoCurrAccount$methods(list(
                      credit = function(amnt) {
                          balance <<- balance + amnt
                          },
                      debet = function(amnt) {
                          if (amnt <= balance) {
                             balance <<- balance - amnt
                             } else {
                             stop("Not rich enough!")
                             }
                            }
                      ))
# ---------------------------------------------------------------------


# we assume that you installed the package before:
# install.packages("tidyverse")
# so load it:
library(tidyverse)
# ---------------------------------------------------------------------


x <- seq(from = 0, to = 2 * pi, length.out = 100)
s <- sin(x)
c <- cos(x)
z <- s + c
plot(x, z, type = "l",col="red", lwd=7)
lines(x, c, col = "blue",  lwd = 1.5)
lines(x, s, col = "darkolivegreen", lwd = 1.5)
# ---------------------------------------------------------------------


x <- seq(from = 0, to = 2 * pi, length.out = 100)
#df <- as.data.frame((x))
df <- rbind(as.data.frame((x)),cos(x),sin(x), cos(x) + sin(x))
# plot etc.
# ---------------------------------------------------------------------


library(tidyverse)
x <- seq(from = 0, to = 2 * pi, length.out = 100)
tb <- tibble(x, sin(x), cos(x), cos(x) + sin(x))
# ---------------------------------------------------------------------


# Note how concise and relevant the output is:
print(tb)  

# This does the same as for a data-frame:
plot(tb)

# Actually a tibble will still behave as a data frame:
is.data.frame(tb)
# ---------------------------------------------------------------------


# The first column can be referred to directly: 
tb$x[1]

# To address the sin(x) column, we need back-tics:
tb$`sin(x)`[1]

# Or refer to the column by its index:
tb[,2]
# ---------------------------------------------------------------------


tb <- tibble(`1` = 1:3, `2` = sin(`1`), `1`*pi, 1*pi)
tb
# ---------------------------------------------------------------------


# -1- data frame
df <- data.frame("value" = pi, "name" = "pi")
df$na        # partial matching of column names

# automatic conversion to factor, plus data frame
# accepts strings:
df[,"name"]   

df[,c("name", "value")] 

# -2- tibble
df <- tibble("value" = pi, "name" = "pi")
df$name       # column name
df$nam        # no partial matching but error msg.
df[,"name"]   # this returns a tibble (no simplification)
df[,c("name", "value")] # no conversion to factor
# ---------------------------------------------------------------------


tb <- tibble(c("a", "b", "c"), c(1,2,3), 9L,9)
is.data.frame(tb)

# Note also that tibble did no conversion to factors, and
# note that the tibble also recycles the scalars:
tb

# Coerce the tibble to data-frame:
as.data.frame(tb)  

# A tibble does not recycle shorter vectors, so this fails:
fail <- tibble(c("a", "b", "c"), c(1,2))
# That is a major advantage and will save many programming errors.
# ---------------------------------------------------------------------


t <- tibble("x" = runif(10))                     
t <- within(t, y <- 2 * x + 4 + rnorm(10, mean = 0, sd = 0.5))
# ---------------------------------------------------------------------


t <- tibble("x" = runif(10))  %>%
     within(y <- 2 * x + 4 + rnorm(10, mean = 0,sd = 0.5))
# ---------------------------------------------------------------------


# 1. pipe: 
a %>% f()
# 2. pipe with shortened function: 
a %>% f
# 3. is equivalent with:
f(a)
# ---------------------------------------------------------------------


a <- c(1:10)
a %>% mean()
a %>% mean
mean(a)
# ---------------------------------------------------------------------


# The following line 
c <- a    %>% 
     f()
# is equivalent with:
c <- f(a)

# Also, it is easy to see that 
x <- a %>% f(y) %>% g(z)
# is the same as:
x <- g(f(a, y), z)
# ---------------------------------------------------------------------


# f1
# Dummy function that from which only the error throwing part 
# is shown.
f1 <- function() {
    # Here goes the long code that might be doing something risky 
    # (e.g. connecting to a database, uploading file, etc.)
    # and finally, if it goes wrong:
    stop("Early exit from f1!")  # throw error
    }
    
tryCatch(f1(),    # the function to try 
         error   = function(e) {paste("_ERROR_:",e)},
         warning = function(w) {paste("_WARNING_:",w)},
         message = function(m) {paste("_MESSSAGE_:",m)},
         finally="Last command"    # do at the end
         )
# ---------------------------------------------------------------------


# f1
# Dummy function that from which only the error throwing part 
# is shown.
f1 <- function() {
    # Here goes the long code that might be doing something risky 
    # (e.g. connecting to a database, uploading file, etc.)
    # and finally, if it goes wrong:
    stop("Early exit from f1!")  # something went wrong
    }   %>%
tryCatch(
         error   = function(e) {paste("_ERROR_:",e)},
         warning = function(w) {paste("_WARNING_:",w)},
         message = function(m) {paste("_MESSSAGE_:",m)},
         finally="Last command"    # do at the end
         )
# Note that it fails in silence.
# ---------------------------------------------------------------------


# This will not work, because lm() is not designed for the pipe.
lm1 <- tibble("x" = runif(10))                            %>%
       within(y <- 2 * x + 4 + rnorm(10, mean=0, sd=0.5)) %>%
       lm(y ~ x)
# ---------------------------------------------------------------------


# The Tidyverse only makes the %>% pipe available. So, to use the
# special pipes, we need to load magrittr 
library(magrittr) 
lm2 <- tibble("x" = runif(10))                           %>%
       within(y <- 2 * x + 4 + rnorm(10, mean=0,sd=0.5)) %$%
       lm(y ~ x)
summary(lm2)
# ---------------------------------------------------------------------


coeff <- tibble("x" = runif(10))                           %>%
         within(y <- 2 * x + 4 + rnorm(10, mean=0,sd=0.5)) %$%
         lm(y ~ x)                                         %>%
         summary                                           %>% 
         coefficients
coeff
# ---------------------------------------------------------------------


library(magrittr)
t <- tibble("x" = runif(100))                           %>%
     within(y <- 2 * x + 4 + rnorm(10, mean=0, sd=0.5)) %T>%
     plot(col="red")   # The function plot does not return anything
                       # so we used the %T>% pipe. Hence the result of 
                       # within() is passed to t.
		       
lm3 <-   t                  %$%
         lm(y ~ x)          %T>% # pass on the linear model for assignment
         summary            %T>% # further pass on the linear model
         coefficients

tcoef <- lm3 %>% coefficients  # we anyhow need the coefficients

# Add the model (the solid line) to the previous plot:
abline(a = tcoef[1], b=tcoef[2], col="blue", lwd = 3)
# ---------------------------------------------------------------------


x <- c(1,2,3) 

# The following line:
x <- x %>% mean  

# is equivalent with the following:
x %<>% mean

# Show x:
x
# ---------------------------------------------------------------------


library(pryr)
x <- runif(100)
object_size(x)
y <- x

# x and y together do not take more memory than only x.
object_size(x,y)   

y <- y * 2

# Now, they are different and are stored separately in memory.
object_size(x,y)   
# ---------------------------------------------------------------------


# The mean of a vector:
x <- c(1,2,3,4,5,60)
mean(x)

# Missing values will block the override the result:
x <- c(1,2,3,4,5,60,NA)
mean(x)
# Missing values can be ignored with na.rm = TRUE:
mean(x, na.rm = TRUE)

# This works also for a matrix:
M <- matrix(c(1,2,3,4,5,60), nrow=3)
mean(M)
# ---------------------------------------------------------------------


v <- c(1,2,3,4,5,6000)
mean(v)
mean(v, trim = 0.2)
# ---------------------------------------------------------------------


returns <- c(0.5,-0.5,0.5,-0.5)

# Arithmetic mean:
aritmean <- mean(returns)

# The ln-mean:
log_returns <- returns
for(k in 1:length(returns)) {
  log_returns[k] <- log( returns[k] + 1)
  }
logmean <- mean(log_returns)
exp(logmean) - 1

# What is the value of the investment after these returns:
V_0 <- 1
V_T <- V_0
for(k in 1:length(returns)) {
  V_T <- V_T * (returns[k] + 1)
  }
V_T

# Compare this to our predictions:
## mean of log-returns
V_0 * (exp(logmean) - 1)
## mean of returns
V_0 * (aritmean + 1)
# ---------------------------------------------------------------------


x <- c(1:5,5e10,NA)
x
median(x)               # no meaningful result with NAs
median(x,na.rm = TRUE)  # ignore the NA
# Note how the median is not impacted by the outlier,
# but the outlier dominates the mean:
mean(x, na.rm = TRUE)
# ---------------------------------------------------------------------


# my_mode 
# Finds the first mode (only one)
# Arguments:
#   v -- numeric vector or factor
# Returns:
#   the first mode
my_mode <- function(v) {
   uniqv <- unique(v)
   tabv  <- tabulate(match(v, uniqv))
   uniqv[which.max(tabv)]
   }

# now test this function
x <- c(1,2,3,3,4,5,60,NA)
my_mode(x)
x1 <- c("relevant", "N/A", "undesired", "great", "N/A", 
        "undesired", "great", "great")
my_mode(x1)

# text from https://www.r-project.org/about.html
t <- "R is available as Free Software under the terms of the 
Free Software Foundation's GNU General Public License in 
source code form. It compiles and runs on a wide variety of
UNIX platforms and similar systems (including FreeBSD and 
Linux), Windows and MacOS."
v <- unlist(strsplit(t,split=" "))
my_mode(v)
# ---------------------------------------------------------------------


# my_mode 
# Finds the mode(s) of a vector v
# Arguments:
#   v -- numeric vector or factor
#   return.all -- boolean -- set to true to return all modes
# Returns:
#   the modal elements
my_mode <- function(v, return.all = FALSE) {
  uniqv  <- unique(v)
  tabv   <- tabulate(match(v, uniqv))
  if (return.all) {
    uniqv[tabv == max(tabv)]
  } else {
    uniqv[which.max(tabv)]
  }
}

# example:
x <- c(1,2,2,3,3,4,5)
my_mode(x)
my_mode(x, return.all = TRUE)
# ---------------------------------------------------------------------


t <- rnorm(100, mean=0, sd=20)
var(t)
sd(t)
sqrt(var(t))
sqrt(sum((t - mean(t))^2)/(length(t) - 1))
# ---------------------------------------------------------------------


mad(t)
mad(t,constant=1)
# ---------------------------------------------------------------------


cor(mtcars$hp,mtcars$wt)
# ---------------------------------------------------------------------


d <- data.frame(mpg = mtcars$mpg, wt = mtcars$wt, hp = mtcars$hp)
# Note that we can feed a whole data-frame in the functions.
var(d)
cov(d)
cor(d)
cov2cor(cov(d))
# ---------------------------------------------------------------------


x  <- c(-10:10)
df <- data.frame(x=x, x_sq=x^2, x_abs=abs(x), x_exp=exp(x))
cor(df)
# ---------------------------------------------------------------------


cor(rank(df$x), rank(df$x_sq))
cor(rank(df$x), rank(df$x_abs))
cor(rank(df$x), rank(df$x_exp))
# ---------------------------------------------------------------------


x  <- c(-10:10)
cor(rank(x), rank(x^2))
# ---------------------------------------------------------------------


# we use the dataset mtcars from MASS
df <- data.frame(mtcars$cyl,mtcars$am)
chisq.test(df)
# ---------------------------------------------------------------------


 obs <- rnorm(600,10,3)
 hist(obs,col="khaki3",freq=FALSE)
 x <- seq(from=0,to=20,by=0.001)
 lines(x, dnorm(x,10,3),col="blue",lwd = 4)
# ---------------------------------------------------------------------


 library(MASS)
 hist(SP500,col="khaki3",freq=FALSE,border="khaki3")
 x <- seq(from=-5,to=5,by=0.001)
 lines(x, dnorm(x,mean(SP500),sd(SP500)),col="blue",lwd=2)
# ---------------------------------------------------------------------


 library(MASS)
 qqnorm(SP500,col="red"); qqline(SP500,col="blue")
# ---------------------------------------------------------------------


# Probability of getting 5 or less heads from 10 tosses of 
# a coin.
pbinom(5,10,0.5)

# visualize this for one to 10 numbers of tosses
x <- 1:10
y <- pbinom(x,10,0.5)
plot(x,y,type="b",col="blue", lwd = 3,
    xlab="Number of tails",
    ylab="prob of maxium x tails",
    main="Ten tosses of a coin")
# ---------------------------------------------------------------------


# How many heads should we at least expect (with a probability 
# of 0.25) when a coin is tossed 10 times.
qbinom(0.25,10,1/2)
# ---------------------------------------------------------------------


# Find 20 random numbers of tails from and event of 10 tosses 
# of a coin
rbinom(20,10,.5)
# ---------------------------------------------------------------------


N <- 100
t <- data.frame(id = 1:N, result = rnorm(N))
summary(t)
# ---------------------------------------------------------------------


library(tidyverse) # provides %>%, group_by, etc.
# In mtcars the type of the car is only in the column names,
# so we need to extract it to add it to the data
n <- rownames(mtcars)  

# Now, add a column brand (use the first letters of the type)
t <- mtcars  %>%
     mutate(brand = str_sub(n, 1, 4))    # add column
# ---------------------------------------------------------------------


# First, we need to find out which are the most abundant brands
# in our dataset (set cutoff at 2: at least 2 cars in database)
top_brands <- count(t, brand) %>% filter(n >= 2)

# top_brands is not simplified to a vector in the tidyverse
print(top_brands)


grouped_cars <- t                      %>% # start with cars
   filter(brand %in% top_brands$brand) %>% # only top-brands
   group_by(brand)                     %>%
   summarise(
       avgDSP = round(mean(disp), 1),
       avgCYL = round(mean(cyl),  1),
       minMPG = min(mpg),
       medMPG = median(mpg),
       avgMPG = round(mean(mpg),2),
       maxMPG = max(mpg),
     )
print(grouped_cars)
# ---------------------------------------------------------------------


# Each call to summarise() removes a layer of grouping:
by_vs_am <- mtcars %>% group_by(vs, am)
by_vs <- by_vs_am %>% summarise(n = n())
by_vs
by_vs %>% summarise(n = sum(n))

# To removing grouping, use ungroup:
by_vs %>%
  ungroup() %>%
  summarise(n = sum(n))

# You can group by expressions: this is just short-hand for
# a mutate/rename followed by a simple group_by:
mtcars %>% group_by(vsam = vs + am)

# By default, group_by overrides existing grouping:
mtcars             %>%
  group_by(cyl)    %>%
  group_by(vs, am) %>%
  group_vars()

# Use add = TRUE to append grouping levels:
mtcars                         %>%
  group_by(cyl)                %>%
  group_by(vs, am, add = TRUE) %>%
  group_vars()
     
# ---------------------------------------------------------------------


# Plot a vector:
x <- c(1:20)^2
plot(x)

# Plot a data frame:
df <- data.frame('a' = x, 'b' = 1/x, 'c' = log(x), 'd' = sqrt(x))
plot(df)
# ---------------------------------------------------------------------


# This sets up an empty plotting field
plot(x = c(0, 4.5),
     y = c(0, 5),
     main = "Some pch arguments",
     xaxt = "n",
     yaxt = "n",
     xlab = "",
     ylab = "",
     cex.main = 2.6, 
     col = "white"
)

# This will plot all of the standard pch arguments
y = rep(5:0, each=5)
for (i in 0:25) { 
  points(x = i %% 5, y = y[i+1], pch = i,cex = 2, col="blue", bg="khaki3")
  text(0.3 + i %% 5, y = y[i+1], i, cex = 2)
}
for (i in 1:2) {
  ch <- LETTERS[i]
  points(x = i, y = 0, pch = ch,cex = 2, col="red")
  text(0.3 + i, y = 0, ch, cex = 2)
}
for (i in 1:2) {
  ch <- letters[i]
  points(x = i + 2, y = 0, pch = ch,cex = 2, col="red")
  text(0.3 + i + 2, y = 0, ch, cex = 2)
}
# ---------------------------------------------------------------------


# Import the data:
library(MASS)

# To make this example more interesting, we convert mpg to l/100km

# mpg2l
# Converts miles per gallon into litres per 100 km
# Arguments:
#    mpg -- numeric -- fuel consumption in MPG
# Returns:
#    Numeric -- fuel consumption in litres per 100 km
mpg2l <- function(mpg = 0) {
  100 * 3.785411784 / 1.609344 / mpg}

mtcars$l <- mpg2l(mtcars$mpg)
plot(x = mtcars$hp,y = mtcars$l, xlab = "Horse Power", 
   ylab = "L per 100km", main = "Horse Power vs Milage",
   pch = 22, col="red", bg="yellow")
# ---------------------------------------------------------------------


# Prepare the data:
years <- c(2000,2001,2002,2003,2004,2005)
sales <- c(2000,2101,3002,2803,3500,3450)
plot(x = years,y = sales, type = 'b', 
     xlab = "Years", ylab = "Sales in USD",
     main = "The evolution of our sales")
points(2004,3500,col="red",pch=16) # highlight one point
text(2004,3400,"top sales")        # annotate the highlight
# ---------------------------------------------------------------------


x <- c(10, 20, 12)            # Create data for the graph
labels <- c("good", "average", "bad")
pie(x,labels)                 # Show in the R Graphics screen
# ---------------------------------------------------------------------


# Illustrating how to saving a plot to a file (assuming x exists):
png(file = "feedback.jpg")  # Give the chart file a name
pie(x,labels)               # Plot the chart
dev.off()                   # Save the file
# ---------------------------------------------------------------------


sales <- c(100,200,150,50,125)
regions <- c("France", "Poland", "UK", "Spain", "Belgium")
barplot(sales, width=1, 
        xlab="Regions", ylab="Sales in EUR", 
	main="Sales 2016", names.arg=regions,
	border="blue", col="brown")
# ---------------------------------------------------------------------


# Create the input vectors:
colours <- c("orange","green","brown")
regions <- c("Mar","Apr","May","Jun","Jul")
product <- c("License","Maintenance","Consulting")

# Create the matrix of the values.
values <- matrix(c(20,80,0,50,140,10,50,80,20,10,30,
       10,25,60,50), nrow = 3, ncol = 5, byrow = FALSE)

# Create the bar chart:
barplot(values, main = "Sales 2016",
   names.arg = regions, xlab = "Region",
   ylab = "Sales in EUR", col = colours)

# Add the legend to the chart:
legend("topright", product, cex = 1.3, fill = colours)
# ---------------------------------------------------------------------


# We reuse the matrix 'values' from previous example.

# Add extra space to right of plot area by changeing clipping to figure:
par(mar = c(5, 4, 4, 8) + 0.1, # default margin was c(5, 4, 4, 2) + 0.1
    xpd = TRUE)      # TRUE to restrict all plotting to the plot region

# Create the plot with all totals coerced to 1.0 with prop.table():
barplot(prop.table(values, 2), main = "Sales 2016",
   names.arg = regions, xlab = "Region",
   ylab = "Sales in EUR", col = colours)

# Add the legend, but move it to the right with inset:
legend("topright", product, cex = 1.0, inset=c(-0.3,0), fill = colours,
       title="Business line")
# ---------------------------------------------------------------------


library(MASS)
boxplot(mpg ~ cyl,data=mtcars,col="khaki3", 
        main="MPG by number of cylinders")
# ---------------------------------------------------------------------


# install.packages('vioplot') # only do once
library(vioplot)              # load the vioplot library
with(mtcars , vioplot(mpg[cyl==4] , mpg[cyl==6], mpg[cyl==8],
                      col=rgb(0.1,0.4,0.7,0.7) , 
		      names=c("4","6","8") 
		      )
    ) 
# ---------------------------------------------------------------------


# Library
library(ggplot2)
 
# First, type of color (color coded in function of number of cylinders as numeric):
ggplot(mtcars, aes(factor(cyl), mpg)) + 
  geom_violin(aes(fill = cyl))
 
# Second type (color coded in function of number of cylinders as factors):
ggplot(mtcars, aes(factor(cyl), mpg)) +
  geom_violin(aes(fill = factor(cyl)))
# ---------------------------------------------------------------------


library(MASS)
incidents <- ships$incidents
# figure 1: with a rug and fixed breaks
hist(incidents, 
     col=c("red","orange","yellow","green","blue","purple"))
rug(jitter(incidents))  # add the tick-marks

# figure 2: user-defined breaks for the buckets
hist(incidents, 
    col=c("red","orange","yellow","green","blue","purple"),
    ylim=c(0,0.3), breaks=c(0,2,5,10,20,40,80),freq=FALSE)
# ---------------------------------------------------------------------


fn1 <- function(x) sqrt(1-(abs(x)-1)^2)
fn2 <- function(x) -3*sqrt(1-sqrt(abs(x))/sqrt(2))
curve(fn1,-2,2,ylim=c(-3,1),col="red",lwd = 4, 
      ylab = expression(sqrt(1-(abs(x)-1)^2) +++ fn_2))
curve(fn2,-2,2,add=TRUE,lw=4,col="red")
text(0,-1,expression(sqrt(1-(abs(x)-1)^2)))
text(0,-1.25,"++++")
text(0,-1.5,expression(-3*sqrt(1-sqrt(abs(x))/sqrt(2))))
# ---------------------------------------------------------------------


# A basic plot is possible with the function image()
# image(volcano)

# We present an improved plot as per documentation of image().

# First we need the size of the image to plot:
x <- 1:nrow(volcano)
y <- 1:ncol(volcano)

# the mapping of colours.
image(x, y, volcano, col = terrain.colors(100), 
      axes = FALSE, xlab='', ylab='')

# add the contour plot
contour(x, y, volcano, levels = seq(90, 200, by = 5),
        add = TRUE, col = "brown")
	
# add axis, a box and a title:
axis(1, at = seq(10, 80, by = 10))
axis(2, at = seq(10, 60, by = 10))
box()
title(main = "Auckland's Maunga Whau Volcano", font.main = 4)
# ---------------------------------------------------------------------


d = as.matrix(mtcars, scale = "none")
heatmap(d)
# ---------------------------------------------------------------------


heatmap(d,scale="column")
# ---------------------------------------------------------------------


heatmap(x, Rowv = NULL, Colv = if(symm) "Rowv" else NULL,
        distfun = dist, hclustfun = hclust,
        reorderfun = function(d, w) reorder(d, w),
        add.expr, symm = FALSE, revC = identical(Colv, "Rowv"),
        scale = c("row", "column", "none"), na.rm = TRUE,
        margins = c(5, 5), ColSideColors, RowSideColors,
        cexRow = 0.2 + 1/log10(nr), cexCol = 0.2 + 1/log10(nc),
        labRow = NULL, labCol = NULL, main = NULL,
        xlab = NULL, ylab = NULL,
        keep.dendro = FALSE, verbose = getOption("verbose"), 
	...)
# ---------------------------------------------------------------------


# If neccesary first download the packages:
install.packages("tm")           # text mining
install.packages("SnowballC")    # text stemming
install.packages("RColorBrewer") # colour palettes
install.packages("wordcloud")    # word-cloud generator 
# ---------------------------------------------------------------------


# Then load the packages:
library("tm")
library("SnowballC")
library("RColorBrewer")
library("wordcloud")
# ---------------------------------------------------------------------


# In this example we use a text version of this very book.
# You will need to use your own text file in the line below:
t <- readLines("data/r-book.txt")

# Then create a corpus of text 
doc <- Corpus(VectorSource(t))
# ---------------------------------------------------------------------


# The file has still a lot of special characters
# e.g. the following replaces '\', '#', and '|' with space:
toSpace <- content_transformer(function (x , pattern ) 
                               gsub(pattern, " ", x))
doc <- tm_map(doc, toSpace, "\\\\")
doc <- tm_map(doc, toSpace, "#")
doc <- tm_map(doc, toSpace, "\\|")
# Note that the backslash needs to be escaped in R
# ---------------------------------------------------------------------


# Convert the text to lower case
doc <- tm_map(doc, content_transformer(tolower))

# Remove numbers
doc <- tm_map(doc, removeNumbers)

# Remove english common stopwords
doc <- tm_map(doc, removeWords, stopwords("english"))

# Remove your own stop words
# specify your stopwords as a character vector:
doc <- tm_map(doc, removeWords, c("can","value","also","price",
              "cost","option","call","need","possible","might",
	      "will","first","etc","one","portfolio", "however",
	      "hence", "want", "simple", "therefore")) 

# Remove punctuations
doc <- tm_map(doc, removePunctuation)

# Eliminate extra white spaces
doc <- tm_map(doc, stripWhitespace)

# Text stemming
#doc <- tm_map(doc, stemDocument)
# ---------------------------------------------------------------------


library(stringi)

wordToReplace <- c('functions', 'packages')
ReplaceWith   <- c('function',  'package')

doc <- tm_map(doc,  function(x) stri_replace_all_fixed(x, 
           wordToReplace, ReplaceWith, vectorize_all = FALSE))
# ---------------------------------------------------------------------


dtm <- TermDocumentMatrix(doc)
m   <- as.matrix(dtm)
v   <- sort(rowSums(m),decreasing=TRUE)
d   <- data.frame(word = names(v),freq=v)
head(d, 10)
# ---------------------------------------------------------------------


barplot(d[1:10,]$freq, las = 2, names.arg = d[1:10,]$word,
        col ="khaki3", main ="Most frequent words",
        ylab = "Word frequencies")
# ---------------------------------------------------------------------


set.seed(1879)
wordcloud(words = d$word, freq = d$freq, min.freq = 10,
          max.words=200, random.order=FALSE, rot.per=0.35, 
          colors=brewer.pal(8, "Dark2"))
# ---------------------------------------------------------------------


  wordcloud(words, freq, scale = c(4,.5), min.freq = 3,
            max.words=Inf, random.order = TRUE, 
	    random.color=FALSE, rot.per=.1,
            colors = "black", ordered.colors = FALSE,
	    use.r.layout=FALSE, fixed.asp=TRUE, ...)
# ---------------------------------------------------------------------


findFreqTerms(dtm, lowfreq = 150)
# ---------------------------------------------------------------------


# e.g. for the word "function"
findAssocs(dtm, terms = "function", corlimit = 0.15)
# ---------------------------------------------------------------------


# find colour numbers that contain the word 'khaki'
grep("khaki",colours())

# find the names of those colours
colors()[grep("khaki",colours())]
# ---------------------------------------------------------------------


# extract the rgb value of a named colour
col2rgb("khaki3")
# ---------------------------------------------------------------------


library(ggplot2)
library(gridExtra)
p <- ggplot(mtcars, aes(x = hp, y = qsec, color = mpg)) + 
      geom_point(size=5) 

# no manual colour specification:
p0 <- p + ggtitle('Default colour range')
      
# using named colours:
p1 <- p + scale_color_gradient(low = "red", high = "green") + 
      ggtitle('Named colour')

# using hexadecimal representation
p2 <- p + scale_color_gradient(low = "#0011ee", high = "#ff55a0") + 
      ggtitle('Hexadecimal colour')

# RGB definition of colour
p3 <- p + scale_color_gradient(low = rgb(0,0,1), high = rgb(0.8,0.8,0)) + 
      ggtitle('RGB colour')

# rainbow colours
p4 <- p + scale_color_gradientn(colours = rainbow(5)) + 
      ggtitle('Rainbow colours')

# using another colour set
p5 <- p + scale_color_gradientn(colours = terrain.colors(5)) + 
      ggtitle('Terrain colours') + 
      theme_light()  # turn off the grey background

grid.arrange(p0, p1, p2, p3, p4, p5, ncol = 2)
# ---------------------------------------------------------------------


N   <- length(colours())  # this is 657
df  <- data.frame(matrix(1:N, nrow=73, byrow = TRUE))
image(1:(ncol(df)), 1:(nrow(df)), as.matrix(t(df)), 
      col = colours(),
      xlab = "X", ylab = "Y")
# ---------------------------------------------------------------------


colours()[(3 - 1)  * 9 + 8]  
colours()[(50 - 1) * 9 + 1]
# ---------------------------------------------------------------------


p <- ggplot(mtcars) +
     geom_histogram(aes(cyl, fill=factor(cyl)), bins=3)

# no manual colour specification:
p0 <- p + ggtitle('default colour range')


# using a built-in colour scale
p1 <- p + scale_fill_grey() + 
      ggtitle('Shades of grey')


library(RColorBrewer)
p2 <- p + scale_fill_brewer() + 
      ggtitle('RColorBrewer')
      
p3 <- p + scale_fill_brewer(palette='Set2') + 
      ggtitle('RColorBrewer Set2')
      
p4 <- p + scale_fill_brewer(palette='Accent') + 
      ggtitle('RColorBrewer Accent')
      
grid.arrange(p0, p1, p2, p3, p4, p5, ncol = 2)
# ---------------------------------------------------------------------


ts(data = NA, start = 1, end = numeric(), frequency = 1,
        deltat = 1, ts.eps = getOption("ts.eps"), class = , 
	names = )
# ---------------------------------------------------------------------


library(MASS)
# The SP500 is available as a numeric vector:
str(SP500)
# ---------------------------------------------------------------------


# Convert it to a time series object.
SP500_ts <- ts(SP500,start = c(1990,1),frequency = 260)
# ---------------------------------------------------------------------


# Compare the original:
class(SP500)

# with:
class(SP500_ts)
# ---------------------------------------------------------------------


plot(SP500_ts)
# ---------------------------------------------------------------------


g <- read.csv('data/gdp/gdp_pol_sel.csv') # get the data
attach(g) # the names of the data are now always available
plot(year, GDP.per.capitia.in.current.USD, type='b', 
     lwd = 3, xlab = 'Year', ylab = 'Polish GDP per Capita in USD')
# ---------------------------------------------------------------------


library(WDI)
# The library WDI allows to search directly via commands like WDIsearch('gdp')

# We want the GDP per capita (in constant 2010 US$) that is the series NY.GDP.PCAP.CD
g = WDI(indicator='NY.GDP.PCAP.CD', country=c('PL')) # we can load more than one country
g <- d[complete.cases(g),]
# ---------------------------------------------------------------------


# Testing accuracy of the model by sampling:
g.ts.tst <- ts(g.data[1:20],start=c(1990))
g.movav.tst <- forecast(ma(g.ts.tst,order=3),h=5)
accuracy(g.movav.tst, g.data[22:26])
# ---------------------------------------------------------------------


plot(g.movav.tst,col="blue",lw=4, 
     main="Forecast of GDP per capita of Poland",
     ylab="Income in current USD")
lines(year, GDP.per.capitia.in.current.USD, col="red",type='b')
# ---------------------------------------------------------------------


train = ts(g.data[1:20],start=c(1990))
test  = ts(g.data[21:26],start=c(2010))
arma_fit <- auto.arima(train)
arma_forecast <- forecast(arma_fit, h = 6)
arma_fit_accuracy <- accuracy(arma_forecast, test)
arma_fit; arma_forecast; arma_fit_accuracy
# ---------------------------------------------------------------------


plot(arma_forecast, col="blue",lw = 4, 
     main = "Forecast of GDP per capita of Poland",
     ylab = "income in current USD")
lines(year,GDP.per.capitia.in.current.USD, col = "red", type = 'b')
# ---------------------------------------------------------------------


g.exp <- ses(g.data,5,initial="simple")
g.exp  # simple exponential smoothing uses the last value as
       # the forecast and finds confidence intervals around it
# ---------------------------------------------------------------------


plot(g.exp,col="blue",lw=4, 
     main="Forecast of GDP per capita of Poland",
     ylab="income in current USD")
lines(year,GDP.per.capitia.in.current.USD,col="red",type='b')
# ---------------------------------------------------------------------


g.exp <- holt(g.data,5,initial="simple")
g.exp  # Holt exponential smoothing
# ---------------------------------------------------------------------


plot(g.exp,col="blue",lw=4, 
     main="Forecast of GDP per capita of Poland",
     ylab="income in current USD")
lines(year,GDP.per.capitia.in.current.USD,col="red",type='b')
# ---------------------------------------------------------------------


# we use the data nottem
# Average Monthly Temperatures at Nottingham, 1920-1939
nottem.stl = stl(nottem, s.window="periodic")
plot(nottem.stl)
# ---------------------------------------------------------------------


add_ts <- log(exp_ts)
# ---------------------------------------------------------------------


# Simple exponential: models level
fit <- HoltWinters(g.data, beta=FALSE, gamma=FALSE)

# Double exponential: models level and trend
fit <- HoltWinters(g.data, gamma=FALSE)

# Triple exponential: models level, trend, and seasonal 
# components. This fails on the example, as there is no 
# seasonal trend:
#fit <- HoltWinters(g.data)  

# Predictive accuracy
library(forecast)
accuracy(forecast(fit,5))
# ---------------------------------------------------------------------


# predict next 5 future values
forecast(fit, 5)
# ---------------------------------------------------------------------


plot(forecast(fit, 5),col="blue",lw=4, 
     main="Forecast of GDP per capita of Poland",
     ylab="income in current USD")
lines(year,GDP.per.capitia.in.current.USD,col="red",type='b')
# ---------------------------------------------------------------------


# Use the Holt-Winters method for the temperatures
n.hw <- HoltWinters(nottem)
n.hw.fc <- forecast(n.hw,50)
plot(n.hw.fc)
