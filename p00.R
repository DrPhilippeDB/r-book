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
 
thm <- knit_theme$get("moe")
knit_theme$set(thm)  #set the knitr color scheme
                    # list: http://animation.r-forge.r-project.org/knitr/
 					  # good ones: acid, moe, default
 					  #            acid: highlights %>% very well
 					  #            moe: a notch better than fruit (nice highlight of %>%)
 					  # edit-vim: white bg (code merges with text)
 					  # fine-blue: fails
 					  # biogoo: bg a notch too dark
 					  # edit-kwrite: little too boring
 					  # fruit: too pink (all comments)
 					  # autumn: hard to read (fg colors too similar to bg colors)
 					  # nuvola: lame and too pink
 					  # seashell: too pink
 					  # easter bg color is too yellow, fg too light
 
 					  
 options(width=58)      # output width (works only for wanted output, not error/warning messages)
 options(cex.lab=1.5)   # doesnt work :-(
 options(cex.axis=1.5)  # doesnt work :-(
 
 opts_knit$set(width = 58) # same output for knitr
 
 rm(list=ls())    # To clear namespace
 library(knitr)   # load knitr
 #opts_chunk$set(cache=TRUE, autodep=TRUE)
 opts_chunk$set(# not used in LaTeX: fig.width=12, fig.height=8, 
                fig.path='./figure/',  #\FIGheight\textheightdefault is also ./figure/
                # highlight = FALSE,
                echo=TRUE,
                warning=TRUE,
 	       message=TRUE,
 	       #cache=TRUE,
                #
                # font size
                size='footnotesize',
                #
                # for figures:
                out.width='\\FIGwidth\\linewidth',
                out.height='\\FIGheight\\textheight',
                fig.align='center',
                fig.pos='!h'
                )
# ---------------------------------------------------------------------


# Code to install all necessary packages:
p <- c("DiagrammeR",
"tidyverse",
"ggvis",
"pryr",
"rex",
"xlsx",
"RMySQL",
"MASS",
"carData",
"titanic",
"rpart",
"rpart.plot",
"ROCR",
"pROC",
"ggplot2",
"viridisLite",   # color schemes
"vioplot",
"grid",
"gridExtra",
"forecast",
"randomForest",
"RCurl",
"XML",
"stringr",
"plyr",
"reshape",
"sandwich",
"msm",
"quantmod",
"tm",
"SnowballC",
"RColorBrewer",
"wordcloud",
"quantmod",
"neuralnet",
"sodium",
"binr",
"diagram",
"latex2exp",
"ggfortify",
"cluster",
"flexdashboard",
"shiny",
"shinydashboard",
"mice",
"VIM",
"missForest",
"Hmisc",
"mi",
"ggrepel",
"plot3D",
"plotly",
"class",
"RColorBrewer",
"InformationValue",
"e1071",
"psych",
"nFactors",
"FactoMineR",
"knitr",
"devtools",
"roxygen2",
"zoo",
"xts",
"compiler",
"Rcpp",
"profr",
"proftools",
"SparkR",
"sparklyr",
"DBI",
"gpuR",
"parallel",
"Rcrawler",
"sqldf"
)

# Clear the bibTex file:
system("cat /dev/null > bibTexPackagesNEW.bib")

# Cycle through all packages:
for (i in 1:length(p)){
 print(paste("===",i, ": ",p[i],"==="))
 
 # If the package is not installed, install it:
 if (!require(p[i], character.only = TRUE)) {install.packages(p[i])}
 
 # Add to the bibTex file to be used at the end of the book:
 ref <- toBibtex(citation(p[i]))
 cat(ref, file="bibTexPackagesNEW.bib", append=TRUE, sep = "\n")
 }
# ---------------------------------------------------------------------


ExtractChunks <- function(file.in, file.out, add.headers = FALSE, ...){
  isRnw <- grepl(".Rnw$",file.in)
  if(!isRnw) stop("file.in should be an Rnw file")

  thelines <- readLines(file.in)

  startid <- grep("^[^%].+>>=$", thelines)
  nocode    <- grep("^ ?<<", thelines[startid+1]) 
  codestart <- startid[-nocode]
    
  out <- sapply(codestart, function(i){
      if (add.headers) {
       tmp   <- thelines[-seq_len(i - 1)]
       } else {
       tmp   <- thelines[-seq_len(i)]
       }
     endid <- grep("^@",tmp)[1]  # might be trailing spaces or comments
     c("# ---------------------------------------------------------------------\n\n", tmp[seq_len(endid-1)])
  })

  writeLines(unlist(out),file.out)
}

# Apply the function:
ExtractChunks(file.in = "r-book.Rnw", file.out = "r-book.R", add.headers = TRUE)
