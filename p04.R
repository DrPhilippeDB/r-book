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


# install.packages('sodium') # do only once
                           # fails if you do not have libsodium-dev
library(sodium)

# Create the SHA256 key based on a secret password:
key <- sha256(charToRaw("My sweet secret"))

# Serialize the data to be encrypted:
msg <- serialize("Philippe J.S. De Brouwer", NULL)

# Encrypt:
msg_encr <- data_encrypt(msg, key)


orig <- data_decrypt(msg_encr, key)
stopifnot(identical(msg, orig))

# Tag the message with your key (HMAC):
tag <- data_tag(msg, key)
# ---------------------------------------------------------------------


# -- 
library(RMySQL)

# -- The functions as mentioned earlier:
# db_get_data
# Get data from a MySQL database
# Arguments:
#    con_info -- MySQLConnection object -- containing the connection 
#                                          info to the MySQL database
#    sSQL     -- character string       -- the SQL statement that selects 
#                                          the records
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
#    con_info -- MySQLConnection object -- containing the connection 
#                                          info to the MySQL database
#    sSQL     -- character string       -- the SQL statement to be run
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


# Load dplyr via tidyverse:
library(tidyverse)

# Define the wrapper functions:

# Step 1: define the connection info.
my_con_info <- list()
my_con_info$user     <- "librarian"
my_con_info$password <- "librarianPWD"
my_con_info$dbname   <- "library"
my_con_info$host     <- "localhost"


# -- The data import was similar to what we had done previously.
# -- However, now we import all tables separately
# Step 2: get the data
my_tables <- c("tbl_authors", "tbl_author_book",
               "tbl_books", "tbl_genres")
my_db_names <- c("authors", "author_book", 
               "books", "genres")

# Loop over the four tables and download their data:
for (n in 1:length(my_tables)) {
  my_sql <- paste("SELECT * FROM `",my_tables[n],"`;", sep="")
  df <- db_get_data(my_con_info, my_sql)
  # the next line uses tibbles are from the tidyverse
  as_tibble(assign(my_db_names[n],df))  
 }

# Step 3: do something with the data
# -- This will follow in the remainder of the section
# ---------------------------------------------------------------------


str(authors)
# ---------------------------------------------------------------------


# example to illustrate the parser functions

v       <- c("1.0", "2.3", "2.7", ".")
nbrs    <- c("$100,00.00", "12.4%")
s_dte   <- "2018-05-03"

# The parser functions can generate from these strings 
# more specialized types.

# The following will generate an error:
parse_double(v)         # reason: "." is not a number
parse_double(v, na=".") # Tell R what the encoding of NA is
parse_number(nbrs)
parse_date(s_dte)
# ---------------------------------------------------------------------


parse_guess(v)
parse_guess(v, na = ".")
parse_guess(s_dte)
guess_parser(v)
guess_parser(v[1:3])
guess_parser(s_dte)
guess_parser(nbrs)
# ---------------------------------------------------------------------


library(tidyverse)
s_csv = "'a','b','c'\n001,2.34,.\n2,3.14,55\n3,.,43"
read_csv(s_csv)
read_csv(s_csv, na = '.')  # Tell R how to understand the '.'
read_csv(s_csv, na = '.',  quote = "'") # Tell how a string is quoted
# ---------------------------------------------------------------------


# Method 1: before the actual import
spec_csv(s_csv, na = '.', quote = "'")

# Method 2: check after facts:
t <- read_csv(s_csv, na = '.', quote = "'")
spec(t)
# ---------------------------------------------------------------------


read_csv(s_csv, na = '.', quote = "'",
  col_names = TRUE,
  cols(
    a = col_character(),
    b = col_double(),
    c = col_double()      # coerce to double
    )
  )
# ---------------------------------------------------------------------


# Start with:
t <- read_csv(readr_example("challenge.csv"))

# Then, to see the issues, do:
problems(t)

# Notice that the problems start in row 1001, so
# the first 1000 rows are special cases. The first improvement
# can be obtained by increase the guesses
## compare 
spec_csv(readr_example("challenge.csv"))
## with 
spec_csv(readr_example("challenge.csv"), guess_max = 1001)
# ---------------------------------------------------------------------


# load readr
library(readr)
# Or load the tidyverse with library(tidyverse), it includes readr.
# ---------------------------------------------------------------------


# Make a string that looks like a fixed-width table (shortened):
txt <- "book_id  year  title                                           genre                                                                                                                                                                   
       1  1896  Les plaisirs et les jour                        LITmod 
       2  1927  Albertine disparue                              LITmod 
       3  1954  Contre Sainte-Beuve                             LITmod 
       8  1687  Philosophiæ Naturalis Principia Mathematica      SCIphy 
       9  -300  Elements (translated )                          SCImat 
      10  2014  Big Data World                                  SCIdat 
      11  2016  Key Business Analytics                          SCIdat 
      12  2011  Maslowian Portfolio Theory                      FINinv 
      13  2016  R for Data Science                              SCIdat"
# ---------------------------------------------------------------------


fileConn <- file("books.txt")
writeLines(txt, fileConn)
close(fileConn)

my_headers <- c("book_id","year","title","genre")
# ---------------------------------------------------------------------


# Reading the fixed-width file
# -- > by indicating the widths of the columns
t <- read_fwf(
  file = "./books.txt",   
  skip = 1,               # skip one line with headers
  fwf_widths(c(8, 6, 48, 8), my_headers)
  )

# Inspect the input:
print(t)
# ---------------------------------------------------------------------


print(t3)
# ---------------------------------------------------------------------


head(mtcars)
mtcars[1,1]          # mpg is in the first column
rownames(mtcars[1,]) # the name of the car is not a column
# ---------------------------------------------------------------------


## -- 
## -- Load dplyr via tidyverse
library(tidyverse)
library(RMySQL)

## -- The functions as mentioned earlier:

# db_get_data
# Get data from a MySQL database
# Arguments:
#    con_info -- MySQLConnection object -- the connection info
#                                          to the MySQL database
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

# db_run_sql
# Run a query that returns no data in an MySQL database
# Arguments:
#    con_info -- MySQLConnection object -- the connection info
#                                          to the MySQL database
#    sSQL     -- character string       -- the SQL statement 
#                                          to be run
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


## -- Import 2 tables combined
# step 2: get the data
my_sql <- "SELECT * FROM tbl_authors 
   JOIN tbl_author_book ON author_id = author
   JOIN tbl_books       ON book      = book_id
   JOIN tbl_genres      ON genre     = genre_id;"
t_mix <- db_get_data(my_con_info, my_sql)
t_mix <- as.tibble(t_mix)  

# Show the result:
head(t_mix)
# ---------------------------------------------------------------------


# Make a table of how much each author_id occurs:
nbr_auth <- t_mix  %>% count(author_id)

# Do the same and include all fields that are assumed to
# be part of the table authors.
nbr_auth2 <- t_mix    %>% 
  count(author_id, pen_name, full_name, birth_date, death_date, book)

nbr_auth$n - nbr_auth2$n
# ---------------------------------------------------------------------


# Try without book:
nbr_auth2 <- t_mix    %>% 
  count(author_id, pen_name, full_name, birth_date, death_date)
  
# Now these occurrences are the same:
nbr_auth$n - nbr_auth2$n
# ---------------------------------------------------------------------


my_authors <- tibble(author_id = t_mix$author_id, 
                    pen_name   = t_mix$pen_name,
                    full_name  = t_mix$full_name,
                    birth_date = t_mix$birth_date,
                    death_date = t_mix$death_date
                    )      %>%
              unique       %>%
              print
# ---------------------------------------------------------------------


auth <- tibble(
            author_id = as.integer(my_authors$author_id), 
            pen_name   = my_authors$pen_name,
            full_name  = my_authors$full_name,
            birth_date = as.Date(my_authors$birth_date),
            death_date = as.Date(my_authors$death_date)
               )       %>%
        unique         %>%
        print
# ---------------------------------------------------------------------


auth$full_name <- str_replace(auth$full_name, "\n", "")     %>%
   print
# ---------------------------------------------------------------------


# First read in some data (using a flat file to remind
# how this works):
 x <- " January   100       102       108
 February  106       105       105
 March     104       104       106
 April     120       122       118
 May       130       100       133
 June      141       139       135
 July      175       176       180
 August    170       188       187
 September 142       148       155
 October   133       137       145
 November  122       128       131
 December  102       108       110"

# Read in the flat file via read_fwf from readr:
t <- read_fwf(x,  fwf_empty(x, col_names = my_headers))

# Set the column names:
colnames(t) <-  c("month", "Sales2017", "Sales2018", "Sales2019")

# Finally, we can show the data as it appeared in the spreadsheet
# from the sales department:
print(t)
# ---------------------------------------------------------------------


t2 <- gather(t, "year", "sales", 2:4)
t2$year <- str_sub(t2$year,6,9)  # delete the sales word
t2$year <- as.integer(t2$year)   # convert to integer

# Show the result:
t2
# ---------------------------------------------------------------------


library(lubridate)
t2$date <- parse_date_time(paste(t2$year,t$month), orders = "ym")
plot(x = t2$date, y = t2$sales, col = "red")
lines(t2$date, t2$sales, col = "blue")
# ---------------------------------------------------------------------


library(dplyr)
sales_info <- data.frame(
       time = as.Date('2016-01-01') + 0:9 + rep(c(0,-1), times=5),
       type  = rep(c("bought","sold"),5),
       value = round(runif(10, min = 0, max = 10001))
       )

# Show the data frame:
sales_info

# Use the function spread():
spread(sales_info, type, value)
# ---------------------------------------------------------------------


sales_info                %>%
  spread(type, value)     %>%
  gather(type, value, 2:3)
# ---------------------------------------------------------------------


library(tidyr)
# The original data frame:
turnover <- data.frame(
       what = paste(as.Date('2016-01-01') + 0:9 + rep(c(0,-1), times=5),
                    rep(c("HSBC","JPM"),5), sep="/"),
       value = round(runif(10, min = 0, max = 50))
       )
turnover

# Use the function separate():
separate(turnover, what, into=c("date","counterpart"), sep="/")
# ---------------------------------------------------------------------


# Use as separator a digit followed by a forward slash 
# and then a capital letter.
separate(turnover, what, into=c("date","counterpart"), 
         sep="[0-9]/[A-Z]")
# ---------------------------------------------------------------------


library(tidyr)

# Define a data frame:
df <- data.frame(year = 2018, month = 0 + 1:12, day = 5)
print(df)

# Merge the columns to one variable:
unite(df, 'date', 'year', 'month', 'day', sep = '-')
# ---------------------------------------------------------------------


library(dplyr)
# ---------------------------------------------------------------------


# Using the example of the library:
dplyr::select(genres,      # the first argument is the tibble
       genre_id, location) # then a list of column names
# ---------------------------------------------------------------------


a1 <- filter(authors, birth_date > as.Date("1900-01-01"))
paste(a1$pen_name,"--",a1$birth_date)
# ---------------------------------------------------------------------


authors           %>%
 count(author_id) %>%
 filter(n > 1)    %>%
 nrow()
# ---------------------------------------------------------------------


author_book     %>%
  count(author) %>%
  filter(n > 1)
# ---------------------------------------------------------------------


library(sqldf)
# Because we have RMySQL loaded (and we don't want to unload it) sqldf will
# default to using that engine to run the queries. If we want it to use the 
# R environment and data frames, then use the following line:
options(sqldf.driver = "SQLite")
# ---------------------------------------------------------------------


# Now you can use SQL syntax on R-data-frames. Imagine that we need to find the
# titles of books of the authors with name ending in 'Brouwer':
sqldf("SELECT B.title FROM authors AS A, author_book as AB, books AS B
              WHERE A.author_id = AB.author AND AB.book = B.book_id
                    AND full_name LIKE '%Brouwer';")
# ---------------------------------------------------------------------


detach("package:sqldf", unload=TRUE)
# ---------------------------------------------------------------------


a2 <- books                                         %>%
  inner_join(genres, by = c("genre" = "genre_id"))  
paste(a2$title, "-->", a2$location)
# ---------------------------------------------------------------------


a3 <- authors                                              %>%
  inner_join(author_book, by = c("author_id" = "author"))  %>%
  inner_join(books,       by = c("book"      = "book_id")) %>%
  inner_join(genres,      by = c("genre"     = "genre_id"))%>%
  dplyr::select(pen_name, location)                        %>%
  arrange(location)                                        %>%
  unique()                                                 %>%
  print()

# Note the difference with the base-R code below!
b <- merge(authors, author_book, by.x="author_id", 
                                 by.y = "author")
b <- merge(b, books,  by.x="book",  by.y = "book_id")
b <- merge(b, genres, by.x="genre", by.y = "genre_id")
b <- cbind(b$pen_name, b$location)  # colnames disappear 
colnames(b) <- c("pen_name", "location")
b <- as.data.frame(b)
b <- b[order(b$location), ] # sort for data frames is order
b <- unique(b)
print(b)
# ---------------------------------------------------------------------


arrange(desc(location))
# ---------------------------------------------------------------------


a3[!duplicated(a3$location), ]
# ---------------------------------------------------------------------


inner_join(A, B, by = c("z", "z")  # ambiguous, but works
inner_join(A, B, by = "z")         # shorter
inner_join(A, B)                   # shortest
# ---------------------------------------------------------------------


ab <- authors                                              %>%
  inner_join(author_book, by = c("author_id" = "author"))  %>%
  inner_join(books,       by = c("book"      = "book_id")) %>%
  add_count(author_id)
ab$n
# ---------------------------------------------------------------------


genres                        %>%
   mutate(genre = genre_id)   %>%  # add column genre
   inner_join(books)          %>%  # leave out the "by="
   dplyr::select(c(title, location))
# ---------------------------------------------------------------------


t <- authors                                                   %>%
     mutate(short_name = str_sub(pen_name,1,7))                %>%
     mutate(x_name = if_else(str_length(pen_name) > 15,
                            paste(str_sub(pen_name,1,8),
                                  "...",
                                  str_sub(pen_name, 
                                         start = -3),
                                  sep=''),
                            pen_name,
                            "pen_name is NA"
                           )
           )                                                   %>%
     mutate(is_alive = 
       if_else(!is.na(birth_date) & is.na(death_date), 
            "YES",
            if_else(death_date < Sys.Date(), 
                "no", 
                "maybe"),
             "NA")
            )                                                  %>%
    dplyr::select(c(x_name, birth_date, death_date, is_alive)) %>%
    print()
# ---------------------------------------------------------------------


authors    %>%
  transmute(name = full_name, my_date = as.Date(birth_date) -5)
# ---------------------------------------------------------------------


authors    %>%
  filter(!is.na(birth_date) & is.na(death_date)) %>%
  transmute(name = full_name, my_date = as.Date(birth_date) -5)  
# ---------------------------------------------------------------------


# Define two sets (with one column):
A <- tibble(col1 = c(1L:4L)) 
B <- tibble(col1 = c(4L,4L,5L))

# Study some of the set-operations:
dplyr::intersect(A,B)
union(A,B)
union_all(A,B)
setdiff(A,B)
setequal(A,B)

# The next example uses a data-frame with two columns:
A <- tibble(col1 = c(1L:4L), 
            col2 = c('a', 'a', 'b', 'b'))
B <- tibble(col1 = c(4L,4L,5L), 
            col2 = c('b', 'b', 'c'))

# Study the same set-operations:
dplyr::intersect(A,B)
union(A,B)
union_all(A,B)
setdiff(A,B)
setequal(A,B)
# ---------------------------------------------------------------------


library(tidyverse)
library(stringr)

# define strings
s1 <- "Hello"  # double quotes are fine
s2 <- 'world.' # single quotes are also fine

# Return the length of a string:
str_length(s1)

# Concatenate strings:
str_c(s1, ", ", s2)       # str_c accepts many strings
str_c(s1, s2, sep = ", ") # str_c also has a 
# ---------------------------------------------------------------------


apropos('str_')
# ---------------------------------------------------------------------


s <- 'World'
str_c('Hello, ', s, '.')
# ---------------------------------------------------------------------


library(stringr)                    # or library(tidyverse)
sVector <- c("Hello", ", ", "world", "Philippe")

str_sub (sVector,1,3)               # the first 3 characters
str_sub (sVector,-3,-1)             # the last 3 characters
str_to_lower(sVector[4])            # convert to lowercase
str_to_upper(sVector[4])            # convert to uppercase
str_c(sVector, collapse=" ")        # collapse into one string
str_flatten(sVector, collapse=" ")  # flatten string
str_length(sVector)                 # length of a string
# ---------------------------------------------------------------------


# Nest the functions:
str_c(str_to_upper(str_sub(sVector[4],1,4)),
      str_to_lower(str_sub(sVector[4],5,-1))
     )

# Use pipes:
sVector[4]        %>%
   str_sub(1,4)   %>%
   str_to_upper()
# ---------------------------------------------------------------------


str <- "abcde"

# Replace from 2nd to 4th character with "+"
str_sub(str, 2, 4) <- "+"  
str
# ---------------------------------------------------------------------


str <- "F0"
str_dup(str, c(2,3))  # duplicate a string
# ---------------------------------------------------------------------


str <- c(" 1 ", "  abc", "Philippe De Brouwer   ")
str_pad(str, 5)  # fills with white-space to x characters
# str_pad never makes a string shorter!
# So to make all strings the same length we first truncate:
str      %>%
  str_trunc(10) %>%
  str_pad(10,"right")   %>%
  print
  
# Remove trailing and leading white space:
str_trim(str)
str_trim(str,"left")

# Modify an existing string to fit a line length:
"The quick brown fox jumps over the lazy dog. "  %>%
   str_dup(5)    %>%
   str_c         %>%  # str_flatten also removes existing \n
   str_wrap(50)  %>%  # Make lines of 50 characters long.
   cat                # or writeLines (print shows "\n")
# ---------------------------------------------------------------------


str <- c("a", "z", "b", "c")

# str_order informs about the order of strings (rank number):
str_order(str)

# Sorting is done with str_sort:
str_sort(str)
# ---------------------------------------------------------------------


library(stringr)   # or library(tidyverse)
sV <- c("philosophy", "physiography", "phis", 
        "Hello world", "Philippe", "Philosophy", 
        "physics", "philology")

# Extracting substrings that match a regex pattern:
str_extract(sV, regex("Phi"))
str_extract(sV, "Phi")        # the same, regex assumed
# ---------------------------------------------------------------------


str_extract(sV, "(p|P)hi")

# Or do it this way:
str_extract(sV, "(phi|Phi)")
# ---------------------------------------------------------------------


# Match also i and y:
str_extract(sV, "(p|P)h(i|y)")

# This is equivalent to:
str_extract(sV, "(phi|Phi|phy|Phy)")
# ---------------------------------------------------------------------


str_extract(sV, "(p|P)h(i|y)[^lL]")
# ---------------------------------------------------------------------


str_extract(sV, "(?i)Ph(i|y)[^(?i)L]")
# ---------------------------------------------------------------------


# First create some example email addresses:
emails <- c("god@heaven.org", "philippe@de-brouwer.com",
            "falsemaail@nothingmy", "mistaken.email.@com")
            
# Define the regex:
regX <- "^([a-zA-Z0-9_\\-\\.]+)@([a-zA-Z0-9_\\-\\.]+)\\.([a-zA-Z]{2,5})$"
 
# Use it to match the sample email adresses:
str_extract(emails, regX)
# ---------------------------------------------------------------------


str_extract("Philippe", "Ph\\w*")  # is greedy
str_extract("Philippe", "Ph\\w*?") # is lazy
# ---------------------------------------------------------------------


# Load the library rex:
library(rex)

# In this example we construct the regex to match a valid URL, and will
# define the valid characters first:
valid_chars <- rex(one_of(regex('a-z0-9\u00a1-\uffff')))
# ---------------------------------------------------------------------


# Then build the regex:
expr <- rex(
  start,       # start of the string: ^

  # Protocol identifier (optional) + //
  group(list('http', maybe('s')) %or% 'ftp', '://'),

  # User: pass authentication (optional)
  maybe(non_spaces,
    maybe(':', zero_or_more(non_space)),
    '@'),

  # Host name:
  group(zero_or_more(valid_chars, 
        zero_or_more('-')), 
        one_or_more(valid_chars)),

  # Domain name:
  zero_or_more('.', 
              zero_or_more(valid_chars, 
              zero_or_more('-')), 
	      one_or_more(valid_chars)),

  # Top Level Domain (TLD) identifier
  group('.', valid_chars %>% at_least(2)),

  # Server port number (optional)
  maybe(':', digit %>% between(2, 5)),

  # Resource path (optional):
  maybe('/', non_space %>% zero_or_more()),
  
  end
)
# ---------------------------------------------------------------------


# Print the result elegantly:
substring(expr, seq(1,  nchar(expr)-1, 40),
                seq(41, nchar(expr),   40))      %>%
  str_c(sep="\n")                              
# ---------------------------------------------------------------------


# for example:
str_extract("www.de-brouwer.com", expr)
str_extract("http://www.de-brouwer.com", expr)
str_extract("error=www.de-brouwer.com", expr)
# ---------------------------------------------------------------------


string <- c("one:1", "NO digit", "c5c5c5", "d123d", "123", 6)
pattern <- "\\d"
# ---------------------------------------------------------------------


# grep() returns the whole string if a match is found:
grep(pattern, string, value = TRUE) 

# The default for value is FALSE -> only returns indexes:
grep(pattern, string)               

# L for returning a logical variable:
grepl(pattern, string)              

# --- stringr ---
# similar to grepl (note order of arguments!)
str_detect(string, pattern)        
# ---------------------------------------------------------------------


# Locate the first match (the numbers are the position in the string):
regexpr (pattern, string)  

# grepexpr() finds all matches and returns a list:
gregexpr(pattern, string)  

# --- stringr ---
# Find the first match and returns a matrix:
str_locate(string, pattern) 

# Find all matches and returns a list (same as grepexpr):
str_locate_all(string, pattern)
# ---------------------------------------------------------------------


# First, we need additionally a replacement (repl)
repl <- "___"

# sub() replaces the first match:
sub(pattern, repl, string)

# gsub() replaces all matches:
gsub(pattern, repl, string)

# --- stringr ---
# str_replace() replaces the first match:
str_replace(string, pattern, repl)

# str_replace_all() replaces all mathches:
str_replace_all(string, pattern, repl)
# ---------------------------------------------------------------------


# regmatches() with regexpr() will extract only the first match:
regmatches(string, regexpr(pattern, string))

# regmatches() with gregexpr() will extract all matches:
regmatches(string, gregexpr(pattern, string)) # all matches

# --- stringr --- 
# Extract the first match:
str_extract(string, pattern)

# Similar as str_extract, but returns column instead of row:
str_match(string, pattern)

# Extract all matches (list as return):
str_extract_all(string, pattern)

# To get a neat matrix output, add simplify = T:
str_extract_all(string, pattern, simplify = TRUE)

# Similar to str_extract_all (but returns column instead of row):
str_match_all(string, pattern)
# ---------------------------------------------------------------------


# --- base-R ---
strsplit(string, pattern)

# --- stringr ---
str_split(string, pattern)
# ---------------------------------------------------------------------


# Load the tidyverse for its functionality such as pipes:
library(tidyverse)

# Lubridate is not part of the core-tidyverse, so we need
# to load it separately:
library(lubridate)  
# ---------------------------------------------------------------------


as.numeric(Sys.time())   # the number of seconds passed since 1 January 1970
as.numeric(Sys.time()) / (60 * 60 * 24 * 365.2422)
# ---------------------------------------------------------------------


# There is a list of functions that convert to a date
mdy("04052018")
mdy("4/5/2018")
mdy("04052018")
mdy("4052018")  # ambiguous formats are refused!
dmy("04052018") # same string, different date
# ---------------------------------------------------------------------


dt <- ymd(20180505)  %>% print
as.numeric(dt)
ymd(17656)
# ---------------------------------------------------------------------


yq("201802")
# ---------------------------------------------------------------------


ymd_hms("2018-11-01T13:59:00") 
dmy_hms("01-11-2018T13:59:00") 

ymd_hm("2018-11-01T13:59") 
ymd_h("2018-11-01T13") 

hms("13:14:15")
hm("13:14")
# ---------------------------------------------------------------------


as_date("2018-11-12")
as_date(0)
as_date(-365)
as_date(today()) - as_date("1969-02-21")
# ---------------------------------------------------------------------


today()
now()
# ---------------------------------------------------------------------


# Note it converts the system time-zone to UTC:
as_datetime("2006-07-22T14:00")  

# Force time-zone:
as_datetime("2006-07-22T14:00 UTC")

as_datetime("2006-07-22 14:00 Europe/Warsaw") #Fails silently!

dt <- as_datetime("2006-07-22 14:00", tz = "Europe/Warsaw") %>% 
      print  

# Get the same date-time numerals in a different time-zone:
force_tz(dt, "Pacific/Tahiti")

# Get the same cosmic moment in a new time-zone
with_tz(dt,  "Pacific/Tahiti")
# ---------------------------------------------------------------------


today(tzone = "Pacific/Tahiti")
date_decimal(2018.521, tz = "UTC")
# ---------------------------------------------------------------------


dt1 <- make_datetime(year = 1890, month = 12L, day = 29L, 
              hour = 8L, tz = 'MST')
dt1
# ---------------------------------------------------------------------


# We will use the date from previous hint:
dt1

year(dt)      # extract the year
month(dt)     # extract the month
week(dt)      # extract the week
day(dt)       # extract the day
wday(dt)      # extract the day of the week as number
qday(dt)      # extract the day of the quarter as number
yday(dt)      # extract the day of the year as number
hour(dt)      # extract the hour
minute(dt)    # extract the minutes
second(dt)    # extract the seconds
quarter(dt)   # extract the quarter 
semester(dt)  # extract the  semester
am(dt)        # TRUE if morning
pm(dt)        # TRUE if afternoon
leap_year(dt) # TRUE if leap-year
# ---------------------------------------------------------------------


# We will use the date from previous example:
dt1

# Experiment changing it:
update(dt, month = 5)
update(dt, year  = 2018)
update(dt, hour  = 18)
# ---------------------------------------------------------------------


moment1 <- as_datetime("2018-10-28 01:59:00", tz = "Europe/Warsaw")
moment2 <- as_datetime("2018-10-28 02:01:00", tz = "Europe/Warsaw")

moment2 - moment1  # Is it 2 minutes or 1 hour and 3 minutes?
moment3 <- as_datetime("2018-10-28 03:01:00", tz = "Europe/Warsaw")

# The clocks were put back in this tz from 3 to 2am.
# So, there is 2 hours difference between 2am and 3am!
moment3 - moment1 
# ---------------------------------------------------------------------


# Calculate the duration in seconds:
dyears(x = 1/365)
dweeks(x = 1)
ddays(x = 1)
dhours(x = 1)
dminutes(x = 1)
dseconds(x = 1)
dmilliseconds(x = 1) 
dmicroseconds(x = 1) 
dnanoseconds(x = 1) 
dpicoseconds(x = 1) 
# ---------------------------------------------------------------------


# Note that a duration object times a number is again a Duration object
# and it allows arithmetic:
dpicoseconds(x = 1) * 10^12
# ---------------------------------------------------------------------


# Investigate the object type:
dur <- dnanoseconds(x = 1)
class(dur)
str(dur)
print(dur)
# ---------------------------------------------------------------------


# Useful for automation:
duration(5, unit = "years") 

# Coerce and logical:
dur <- dyears(x = 10)
as.duration(60 * 60 * 24)
as.duration(dur)
is.duration(dur)
is.difftime(dur)
as.duration(dur)
make_difftime(60, units="minutes")
# ---------------------------------------------------------------------


years(x = 1)
months(x = 1)
weeks(x = 1)
days(x = 1)
hours(x = 1)
minutes(x = 1)
seconds(x = 1)
milliseconds(x = 1)
microseconds(x = 1)
nanoseconds(x = 1)
picoseconds(x = 1)

# Investigate the object type:
per <- days(x = 1)
class(per)
str(per)
print(per)
# ---------------------------------------------------------------------


# For automations:
period(5, unit = "years") 

# Coerce timespan to period:
as.period(5, unit="years") 


as.period(10)
p <- seconds_to_period(10) %>%
     print
period_to_seconds(p)
# ---------------------------------------------------------------------


years(1) + months(3) + days(13)
# ---------------------------------------------------------------------


d1 <- ymd_hm("1939-09-01 09:00", tz = "Europe/Warsaw")
d2 <- ymd_hm("1945-08-15 12:00", tz = "Asia/Tokyo")

interval(d1, d2)  # defines the interval

# Or use the operator %--%:
ww2 <- d1 %--% d2 # defines the same interval

ww2 / days(1)   # the period expressed in days
ww2 / ddays(1)  # duration in terms of days
# The small difference is due to DST and equals one hour:
(ww2 / ddays(1) - ww2 / days(1)) * 24

# Allow the interval to report on its length:
int_length(ww2) / 60 / 60 / 24
# ---------------------------------------------------------------------


d_date <- ymd("19450430")

# Is a date or interval in another:
d_date %within% ww2
ph <- interval(ymd_hm("1941-12-07 07:48", tz = "US/Hawaii"),
                     ymd_hm("1941-12-07 09:50", tz = "US/Hawaii")
		     )
ph %within% ww2    # is ph in ww2?
int_aligns(ph, ww2) # do ww2 and ph share start or end?

# Shift forward or backward:
int_shift(ww2, years(1))
int_shift(ww2, years(-1))

# Swap start and end moment
flww2 <- int_flip(ww2)

# Coerce all to "positive" (start-date before end-date)
int_standardize(flww2)

# Modify start or end date
int_start(ww2) <- d_date; print(ww2)
int_end(ww2)  <-  d_date; print(ww2)
# ---------------------------------------------------------------------


dts <- c(ymd("2000-01-10"), ymd("1999-12-28"),
         ymd("1492-01-01"), ymd("2100-10-15")
	 )
round_date(dts, unit="month")
floor_date(dts, unit="month")
ceiling_date(dts, unit="month")
# Change a date to the last day of the previous month or
# to the first day of the month with rollback()
rollback(dts, roll_to_first = FALSE, preserve_hms = TRUE) 
# ---------------------------------------------------------------------


set.seed(1911)
s <- tibble(reply = runif(n = 1000, min = 0, max = 13))
hml <- function (x = 0) {
  if (x < 0)  return(NA)
  if (x <= 4) return("L")
  if (x <= 8) return("M")
  if (x <= 12) return("H")
  return(NA)
  }
surv <- apply(s, 1, FUN = hml)  # output is a vector
surv <- tibble(reply = surv)  # coerce back to tibble
surv
# ---------------------------------------------------------------------


# 1. Define the factor-levels in the right order:
f_levels <- c("L", "M", "H")

# 2. Define our data as factors:
survey <- parse_factor(surv$reply, levels = f_levels)
# ---------------------------------------------------------------------


summary(survey)
plot(survey, col="khaki3",
     main = "Customer Satisfaction",
     xlab = "Response to the last survey"
     )
# ---------------------------------------------------------------------


surv2 <- parse_factor(surv$reply, levels = unique(surv$reply))

# Note that the labels are in order of first occurrence
summary(surv2)
# ---------------------------------------------------------------------


# Count the labels:
fct_count(survey)
# ---------------------------------------------------------------------


# Relabel factors with fct_relabel:
HML <- function (x = NULL) {
  x[x == "L"] <- "Low"
  x[x == "M"] <- "Medium/High"
  x[x == "H"] <- "Medium/High"
  x[!(x %in% c("High", "Medium/High", "Low"))] <- NA
  return(x)
  }
f <- fct_relabel(survey, HML)
summary(f)
plot(f, col="khaki3",
     main = "Only one third of customers is not happy",
     xlab = "Response to the expensive survey"
     )
# ---------------------------------------------------------------------


HMLregex <- function (x = NULL) {
  x[grepl("^L$", x)] <- "Low"
  x[grepl("^M$", x)] <- "Medium/High"
  x[grepl("^H$", x)] <- "Medium/High"
  x[!(x %in% c("High", "Medium/High", "Low"))] <- NA
  return(x)
  }
# This would do exactly the same, but it is a powerful 
# tool with many other possibilities.
# ---------------------------------------------------------------------


num_obs <- 1000  # the number of observations in the survey
# Start from a new survey: srv
srv <- tibble(reply = 1:num_obs)
srv$age <- rnorm(num_obs, mean=50,sd=20)
srv$age[srv$age < 15] <- NA
srv$age[srv$age > 85] <- NA

hml <- function (x = 0) {
  if (x < 0)  return(NA)
  if (x <= 4) return("L")
  if (x <= 8) return("M")
  if (x <= 12) return("H")
  return(NA)
  }

for (n in 1:num_obs) {
  if (!is.na(srv$age[n])) {
     srv$reply[n] <- hml(rnorm(n = 1, mean = srv$age[n] / 7, sd = 2))
   }
   else {
     srv$reply[n] <- hml(runif(n = 1, min = 1, max = 12))
   }
}
f_levels <- c("L", "M", "H")
srv$fct <- parse_factor(srv$reply, levels = f_levels)
# ---------------------------------------------------------------------


# From most frequent to least frequent:
srv$fct                    %>%
fct_infreq(ordered = TRUE) %>%
  levels()
# ---------------------------------------------------------------------


# From least frequent to more frequent:
srv$fct      %>%
 fct_infreq  %>%
  fct_rev    %>% 
  levels
# ---------------------------------------------------------------------


# Reorder the reply variable in function of median age:
fct_reorder(srv$reply, srv$age) %>% 
   levels
# ---------------------------------------------------------------------


# Add the function min() to order based on the minimum
# age in each group (instead of default median):
fct_reorder(srv$reply, srv$age, min) %>% 
   levels
# ---------------------------------------------------------------------


# Show the means per class of satisfaction in base-R style:
by(srv$age, srv$fct, mean, na.rm = TRUE)

# Much more accessible result with the dplyr:
satisf <- srv            %>%
          group_by(fct)  %>%
	  summarize(
	     age = median(age, na.rm = TRUE),
	     n = n()
	     )           %>%
         print
# ---------------------------------------------------------------------


# Show the impact of age on satisfaction visually:
par(mfrow = c(1,2))
barplot(satisf$age,  horiz=TRUE, names.arg = satisf$fct,
        col=c("khaki3","khaki3","khaki3","red"), 
	main = "Median age per group")
barplot(satisf$n,  horiz = TRUE, names.arg = satisf$fct,
        col=c("khaki3","khaki3","khaki3","red"), 
	main = "Frequency per group")
# ---------------------------------------------------------------------


srv                                 %>%
 mutate("fct_ano" = fct_anon(fct))  %>%
 print
# ---------------------------------------------------------------------


set.seed(1890)
# Get the data:
d1       <- d0  <- iris

# Introduce the missing values:
i        <- sample(1:nrow(d0), round(0.20 * nrow(d0)))
d1[i,1]  <- NA
i        <- sample(1:nrow(d0), round(0.30 * nrow(d0)))
d1[i,2]  <- NA

# Show a part of the resulting dataset:
head(d1, n=10L)
# ---------------------------------------------------------------------


#install.packages('mice')  # uncomment if necessary
library(mice)              # load the package

# mice provides the improved visualization function md.pattern():
md.pattern(d1)  # function provided by mice
# ---------------------------------------------------------------------


d2_imp <- mice(d1, m = 5, maxit = 25, method = 'pmm', seed = 1500)
# ---------------------------------------------------------------------


# Choose set number 3:
d3_complete <- complete(d2_imp, 3)
# ---------------------------------------------------------------------


# install.packages('missForest') # only first time
library(missForest)              # load the library
d_mf <- missForest(d1)           # using the same data as before

# access the imputed data in the ximp attribute:
head(d_mf$ximp)

# normalized MSE of imputation:
d_mf$OOBerror
# ---------------------------------------------------------------------


# Install the package first via:
# install.packages('Hmisc')
library(Hmisc)

# impute using mean:
SepLImp_mean <- with(d1, impute(Sepal.Length, mean))

# impute a randomly chosen value:
SepLImp_rand <- with(d1, impute(Sepal.Length, 'random'))

# impute the maximum value:
SepLImp_max <- with(d1, impute(Sepal.Length, max))

# impute the minimum value:
SepLImp_min <- with(d1, impute(Sepal.Length, min))

# note the '*' next to the imputed values"
head(SepLImp_min, n = 10L)
# ---------------------------------------------------------------------


aregImp <- aregImpute(~ Sepal.Length + Sepal.Width 
                        + Petal.Length + Petal.Width + Species, 
			data = d1, n.impute = 4)

print(aregImp)

# n.impute = 4 produced 4 sets of imputed values
# Access the second imputed data set as follows:
head(aregImp$imputed$Sepal.Length[,2], n = 10L)
# ---------------------------------------------------------------------


set.seed(1890)
d <- rnorm(90)
par(mfrow=c(1,2))
hist(d, breaks=70, col="khaki3")
hist(d, breaks=12, col="khaki3")
# ---------------------------------------------------------------------


# Try a possible cut
c <- cut(d, breaks = c(-3, -1, 0, 1, 2, 3))
table(c)

# This is not good, it will not make solid predictions for the last bin.
# So, we neet to use other bins:
c <- cut(d, breaks = c(-3, -0.5, 0.5, 3))
table(c)

# We have now a similar number of observations in each bin.
# Is that the only thing to think about?
# ---------------------------------------------------------------------


# install.packages('binr) # do once
library(binr)
b <- bins.quantiles(d, target.bins=5, max.breaks=10)
b$binct
# ---------------------------------------------------------------------


set.seed(1890)
age <- rlnorm(1000, meanlog = log(40), sdlog = log(1.3))
y <- rep(NA, length(age))
for(n in 1:length(age)) {
  y[n] <- max(0, 
              dnorm(age[n], mean= 40, sd=10) 
	         + rnorm(1, mean = 0, sd = 10 * dnorm(age[n], 
		   mean= 40, sd=15)) * 0.075)
}
y <- y / max(y)
plot(age, y,
     pch = 21, col = "blue", bg = "red",
     xlab = "age",
     ylab = "spending ratio"
     )

# Assume this data is:
#   age            = age of customer
#   spending_ratio = R : = S_n/ (S_{n-1} + S_n)
#                       (zero if both are zero)
#      with S_n the spending in month n
dt <- tibble (age = age, spending_ratio = y)
# ---------------------------------------------------------------------


# Leave out NAs (in this example redundant):
d1 <- dt[complete.cases(dt),]  

# order() returns sorted indices, so this orders the vector:
d1 <- d1[order(d1$age),]  

# Fit a loess:
d1_loess <- loess(spending_ratio ~ age, d1)

# Add predictions:
d1_pred_loess <- predict(d1_loess)

# Plot the results:
par(mfrow=c(1,2))
plot(d1$age, d1$spending_ratio, pch=16,
     xlab = 'age', ylab = 'spending ratio')
lines(d1$age, d1_pred_loess, lwd = 7, col = 'dodgerblue4')
hist(d1$age, col = 'dodgerblue4', xlab = 'age')
par(mfrow=c(1,1))
# ---------------------------------------------------------------------


# Fit the logistic regression directly on the data without binning:
lReg1 <- glm(formula = spending_ratio ~ age, 
             family = quasibinomial, 
             data = dt)
# ---------------------------------------------------------------------


# Investigate the model:
summary(lReg1)

# Calculate predictions and means square error:
pred1 <- 1 / (1 + exp( -(coef(lReg1)[1] + dt$age * coef(lReg1)[2])))
SE1   <-  (pred1 - dt$spending_ratio)^2
MSE1  <- sum(SE1) / length(SE1)
# ---------------------------------------------------------------------


# Bin the variable age:
c <- cut(dt$age, breaks = c(15, 30, 55, 90))

# Check the binning:
table(c)
# We have one big bucket and two smaller (with the smallest
# more than 10% of our dataset.

lvls <- unique(c)      # find levels
lvls                   # check levels order

# Create the tibble (a data-frame also works):
dt <- as_tibble(dt)                            %>%
      mutate(is_L = if_else(age <= 30, 1, 0))  %>%
      mutate(is_H = if_else(age > 55 , 1, 0))

# Fit the logistic regression with is_L and is_H:
# (is_M is not used because it is correlated with the previous)
lReg2 <- glm(formula = spending_ratio ~ is_L + is_H, 
             family = quasibinomial, data = dt)
# ---------------------------------------------------------------------


# Investigate the logistic model:
summary(lReg2)

# Calculate predictions for our model and calculate MSE:
pred2 <- 1 / (1+ exp(-(coef(lReg2)[1] + dt$is_L * coef(lReg2)[2] 
                     + dt$is_H * coef(lReg2)[3])))
SE2  <-  (pred2 - dt$spending_ratio)^2
MSE2 <- sum(SE2) / length(SE2)
# ---------------------------------------------------------------------


# Compare the MSE of the two models:
MSE1
MSE2
# ---------------------------------------------------------------------


# Load libraries and define parameters:
library(tidyverse) # provides tibble (only used in next block)
set.seed(1880)     # to make results reproducible
N <- 500           # number of rows

# Ladies first:
# age will function as our x-value:
age_f   <- rlnorm(N, meanlog = log(40), sdlog = log(1.3))
# x is a temporary variable that will become the propensity to buy:
x_f <- abs(age_f + rnorm(N, 0, 20))    # Add noise & keep positive
x_f <- 1 - (x_f - min(x_f)) / max(x_f) # Scale between 0 and 1
x_f <- 0.5 * x_f / mean(x_f)           # Coerce mean to 0.5
# This last step will produce some outliers above 1
x_f[x_f > 1] <- 1   # Coerce those few that are too big to 1

# Then the gentlemen:
age_m   <- rlnorm(N, meanlog = log(40), sdlog = log(1.3))
x_m <- abs(age_m + rnorm(N, 0, 20))    # Add noise & keep positive
x_m <- 1 - (x_m - min(x_m)) / max(x_m) # Scale between 0 and 1
x_m <- 0.5 * x_m / mean(x_m)           # Coerce mean to 0.5
# This last step will produce some outliers above 1
x_m[x_m > 1] <- 1   # Coerce those few that are too big to 1
x_m <- 1 - x_m                         # relation to be increasing

# Rename (p_x is not the gendered propensity to buy)
p_f <- x_f
p_m <- x_m
# ---------------------------------------------------------------------


# We want a double plot, so change plot params & save old values:
oldparams <- par(mfrow=c(1,2)) 

plot(age_f, p_f,
     pch = 21, col = "blue", bg = "red",
     xlab = "Age",
     ylab = "Spending probability",
     main = "Females"
    )
plot(age_m, p_m,
     pch = 21, col = "blue", bg = "red",
     xlab = "Age",
     ylab = "Spending probability",
     main = "Males"
    )
par(oldparams)   # Reset the plot parameters after plotting
# ---------------------------------------------------------------------


d1 <- t[complete.cases(t),]  

d1 <- d1[order(d1$age),]  
d1_age_loess <- loess(is_good ~ age, d1)
d1_age_pred_loess <- predict(d1_age_loess)

d1 <- d1[order(d1$sexM),]  
d1_sex_loess <- loess(is_good ~ sexM, d1)
d1_sex_pred_loess <- predict(d1_sex_loess)

# Plot the results:
par(mfrow=c(2,2))
d1 <- d1[order(d1$age),]  
plot(d1$age, d1$is_good, pch=16,
     xlab = 'Age', ylab = 'Spending probability')
lines(d1$age, d1_age_pred_loess, lwd = 7, col = 'dodgerblue4')
hist(d1$age, col = 'khaki3', xlab = 'age')

d1 <- d1[order(d1$sexM),]  
plot(d1$sexM, d1$is_good, pch=16,
     xlab = 'Gender', ylab = 'Spending probability')
lines(d1$sexM, d1_sex_pred_loess, lwd = 7, col = 'dodgerblue4')
hist(d1$sexM, col = 'khaki3', xlab = 'gender')
par(mfrow=c(1,1))
# ---------------------------------------------------------------------


# Note that we can feed "sex" into the model and it will create
# for us a variable "sexM" (meaning the same as ours)
# To avoid this confusion, we put in our own variable.
regr1 <- glm(formula = is_good ~ age + sexM, 
             family = quasibinomial,
             data = t)
# ---------------------------------------------------------------------


# assess the model:
summary(regr1)

pred1 <- 1 / (1+ exp(-(coef(regr1)[1] + t$age * coef(regr1)[2] 
                     + t$sexM * coef(regr1)[3])))
SE1   <- (pred1 - t$is_good)^2
MSE1  <- sum(SE1) / length(SE1)
# ---------------------------------------------------------------------


# 1. Check the potential cut:
c <- cut(t$age, breaks = c(min(t$age), 35, 55, max(t$age)))
table(c)
# ---------------------------------------------------------------------


# 2. Create the matrix variables:
t <- as_tibble(t)                                               %>%
    mutate(is_LF = if_else((age <= 35) & (sex == "F"), 1L, 0L)) %>%
    mutate(is_HF = if_else((age >  50) & (sex == "F"), 1L, 0L)) %>%
    mutate(is_LM = if_else((age <= 35) & (sex == "M"), 1L, 0L)) %>%
    mutate(is_HM = if_else((age >  50) & (sex == "M"), 1L, 0L)) %>%
    print
# ---------------------------------------------------------------------


# 3. Check if the final bins aren't too small:
t[,5:8] %>% map_int(sum)
# ---------------------------------------------------------------------


regr2 <- glm(formula = is_good ~ is_LF + is_HF + is_LM + is_HM,
             family = quasibinomial,
             data = t)
# ---------------------------------------------------------------------


# Assess the model:
summary(regr2)
# ---------------------------------------------------------------------


# Calculate the MSE for model 2:
pred2 <- 1 / (1+ exp(-(coef(regr2)[1] + 
                     + t$is_LF * coef(regr2)[2] 
		     + t$is_HF * coef(regr2)[3] 
                     + t$is_LM * coef(regr2)[4] 
		     + t$is_HM * coef(regr2)[5] 
                      )))
SE2 <-  (pred2 - t$is_good)^2
MSE2 <- sum(SE2) / length(SE2)
# ---------------------------------------------------------------------


MSE1
MSE2
# ---------------------------------------------------------------------


t <- mutate(t, "is_good" = if_else(is_good >= 0.5, 1L, 0L))
# ---------------------------------------------------------------------


# We start from this dataset used in previous section:
print(t)
# ---------------------------------------------------------------------


#install.packages("InformationValue") 
library(InformationValue)

WOETable(X = factor(t$sexM), Y = t$is_good, valueOfGood=1)    %>%
   knitr::kable(format.args = list(big.mark = " ", digits=2))

## also functions WOE() and IV(), e.g.
# IV of a categorical variable is the sum of IV of its categories
IV(X = factor(t$sexM), Y = t$is_good, valueOfGood=1)
# ---------------------------------------------------------------------


WOETable(X = factor(t$is_LF), Y = t$is_good, valueOfGood=1) %>%
   knitr::kable(digits=2)

# The package porvides also functions WOE() and IV().
# The IV of a categorical variable is the sum of IV of its categories.
IV(X = factor(t$is_LF), Y = t$is_good, valueOfGood=1)
# ---------------------------------------------------------------------


loadings(fit)     # show PC loadings
# ---------------------------------------------------------------------


head(fit$scores)  # the first principal components 
# ---------------------------------------------------------------------


# plot the loadings (output see figure):
plot(fit,type="b", col='khaki3') 

# show the biplot:
biplot(fit)       
# ---------------------------------------------------------------------


# Maximum Likelihood Factor Analysis

# Extracting 3 factors with varimax rotation:
fit <- factanal(mtcars, 3, rotation = "varimax")
print(fit, digits = 2, cutoff = .3, sort = TRUE)
# ---------------------------------------------------------------------


# plot factor 1 by factor 2
load <- fit$loadings[,1:2]
plot(load, type = "n")                     # plot the loads
text(load, labels = colnames(mtcars), 
     cex = 1.75, col = 'blue')             # add variable names 
# ---------------------------------------------------------------------


# load the library nFactors:
library(nFactors)
# ---------------------------------------------------------------------


# Get the eigenvectors:
eigV <- eigen(cor(mtcars))


# Get mean and selected quantile of the distribution of eigen-
# values of correlation or a covariance matrices of standardized
# normally distributed variables:
aPar <- parallel(subject = nrow(mtcars),var = ncol(mtcars),
                 rep = 100, cent = 0.05)
  
# Get the optimal number of factors analysis:
nScr <- nScree(x = eigV$values, aparallel = aPar$eigen$qevpea)

# See the result
nScr
# and plot it.
plotnScree(nScr) 
# ---------------------------------------------------------------------


# PCA Variable Factor Map
library(FactoMineR)

# PCA will generate two plots as side effect (not shown here)
result <- PCA(mtcars) 
