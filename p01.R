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


# This is code
1+pi
Sys.getenv(c("EDITOR","USER","SHELL", "LC_NUMERIC"))
# ---------------------------------------------------------------------


# generate 1000 random numbers between 0 and 100:
x <- rnorm(1000, mean = 100, sd = 2)

# to illustrate previous, we show the histogram:
hist(x, col = "khaki3")

# This part code is after the histogram. It could be 
# following the previous part and form a unit that is
# interupted by the outpuf of the 'hist(..)'-command.
# In rare cases the plot will be on the this page
# alone and this code that follows appears on the 
# previous page.
# ---------------------------------------------------------------------


# First, generate some data:
x <- c(1,2,3)

# Then calculate the mean:
mean(x)
# ---------------------------------------------------------------------


mean(1:100)
# ---------------------------------------------------------------------


# Code and especially the comments in it are part of 
# the normal flow of the text!
