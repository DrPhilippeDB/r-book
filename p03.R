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


x <- 4; y <- x + 1
# ---------------------------------------------------------------------


# install.packages('RMySQL')
library(RMySQL)
# connect to the library
con <- dbConnect(MySQL(), 
                 user     = "librarian", 
                 password = "librarianPWD", 
                 dbname   = "library", 
                 host     = "localhost"
                 )
                 
# in case we would forget to disconnect:
on.exit(dbDisconnect(con))
# ---------------------------------------------------------------------


# Show some information:

show(con)
summary(con, verbose = TRUE)  
# dbGetInfo(con)  # similar as above but in list format
dbListResults(con)
dbListTables(con) # check: this might generate too much output

# Get data:
df_books <- dbGetQuery(con, "SELECT COUNT(*) AS nbrBooks 
                     FROM tbl_author_book GROUP BY author;")

# Now, df_books is a data frame that can be used as usual.

# close the connection:
dbDisconnect(con)
# ---------------------------------------------------------------------


# Load the package: 
library(RMySQL)

# db_get_data
# Get data from a MySQL database
# Arguments:
#    con_info -- MySQLConnection object -- the connection info to 
#                                          the MySQL database
#    sSQL     -- character string       -- the SQL statement that 
#                                          selects the records
# Returns
#    data.frame, containing the selected records
db_get_data <- function(con_info, sSQL){
  con <- dbConnect(MySQL(), 
                 user     = con_info$user, 
                 password = con_info$password, 
                 dbname   = con_info$dbname, 
                 host     = con_info$host
                 )
  df <- dbGetQuery(con, sSQL)
  dbDisconnect(con)
  df
}
# ---------------------------------------------------------------------


# db_run_sql
# Run a query that returns no data in an MySQL database
# Arguments:
#    con_info -- MySQLConnection object -- open connection
#    sSQL     -- character string       -- the SQL statement to run
db_run_sql <-function(con_info, sSQL)
{
  con <- dbConnect(MySQL(), 
                 user     = con_info$user, 
                 password = con_info$password, 
                 dbname   = con_info$dbname, 
                 host     = con_info$host
                 )
  rs <- dbSendQuery(con,sSQL)
  dbDisconnect(con)
}
# ---------------------------------------------------------------------


# use the wrapper functions to get data.

# step 1: define the connection info
my_con_info <- list()
my_con_info$user     <- "librarian"
my_con_info$password <- "librarianPWD"
my_con_info$dbname   <- "library"
my_con_info$host     <- "localhost"


# step 2: get the data
my_query <- "SELECT COUNT(*) AS nbrBooks 
                     FROM tbl_author_book GROUP BY author;"
df <- db_get_data(my_con_info, my_query)

# step 3: use this data to produce the histogram:
hist(df$nbrBooks, col='khaki3')
# ---------------------------------------------------------------------


# -- reset query cache 
sql_reset_query_cache <- function (con_info)
{
  con <- dbConnect(MySQL(), 
                 user     = con_info$user, 
                 password = con_info$password, 
                 dbname   = con_info$dbname, 
                 host     = con_info$host
                 )
# remove all cache:
system("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches")
# clear MySQL cache cache and disconnect:
rc <- dbSendQuery(con, "RESET QUERY CACHE;")
dbDisconnect(con)
# once more remove all cache:
system("sync && echo 3 | sudo tee /proc/sys/vm/drop_caches")
}
