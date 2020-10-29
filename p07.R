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


# install once: install.packages('ggplot2')
library(ggplot2)
# ---------------------------------------------------------------------


p <- ggplot(mtcars, aes(x=wt, y=mpg))
# So far printing p would result in an empty plot.
# We need to add a geom to tell ggplot how to plot.
p <- p + geom_point()

# Now, print the plot:
p
# ---------------------------------------------------------------------


# Note that we can reuse the object p
p <- p + geom_smooth(method = "loess", span = 0.9, alpha = 0.3)
p <- p + xlab('Weight') + ggtitle('Fuel consumption explained')
p <- p + ylab('Fuel consumption in mpg')
p <- p + theme(axis.text  = element_text(size=14),
               axis.title = element_text(size=16),
	       plot.title = element_text(size = 22, face = 'bold'))
p
# ---------------------------------------------------------------------


# Try *one* of the following:
p <- p + theme_minimal(base_size = 16)  # change also font size
p <- p + theme_light()                  # light theme
p <- p + theme_dark()                   # dark theme
p <- p + theme_classic()                # classic style
p <- p + theme_minimal()                # minimalistic style
# Much more themes are available right after loading ggplot2.

# Change the theme for the entire session:
theme_set(theme_classic(base_size = 12))
# ---------------------------------------------------------------------


# We start from the previously generated plot p:
p <- p + aes(colour = factor(am))
p <- p + aes(size   = qsec)
p
# ---------------------------------------------------------------------


library(tidyverse)  # provides the pipe: %>%
mtcars %>%
  ggplot(aes(x = wt, y = mpg)) + 
        geom_point() +
        geom_smooth(method = "loess", span = 0.99, alpha = 0.3) + 
	xlab('Weight') + 
	ggtitle('MPG in function of weight per nbr cylinders') + 
	facet_wrap( ~ cyl, nrow = 2) + 
	theme_classic(base_size = 14) # helps the grey to stand out
# ---------------------------------------------------------------------


# Set the seed to allow results to be replicated:
set.seed(1868)

# Generate the data:
LTI <- runif(10000, min = 0, max = 1)
DPD <- abs(rnorm(10000, 
                 mean = 70 * ifelse(LTI < 0.5, LTI, 0.5), 
		 sd = 30 * sqrt(LTI) + 5))

# Plot the newly generated data and try to make it not cluttered:
plot(LTI, DPD, 
     pch=19,          # small dot
     ylim =c(0, 100)) # not show outliers, hence zooming in
# ---------------------------------------------------------------------


# We add also the colour schemes of viridisLite:
library(viridisLite)

d <- data.frame(LTI = LTI, DPD = DPD)
p <- ggplot(d, aes(x = LTI, y = DPD)) + 
     stat_density_2d(geom = "raster", aes(fill = ..density..), 
                     contour = FALSE) + 
     geom_density_2d() + 
     scale_fill_gradientn(colours = viridis(256, option = "D")) + 
     ylim(0,100)
p
# Note that ggplot will warn us about the data that is not shown
# due to the cut-off on the y-axis -- via the function ylim(0, 100).
# ---------------------------------------------------------------------


# Loess smoothing with ggplot:
# Note tha we have commented out the limits on the y-axis.
ggplot(d, aes(x = LTI, y = DPD)) + geom_point(alpha=0.25) +
       geom_smooth(method='loess') # + ylim(0,100)
# ---------------------------------------------------------------------


library(ggplot2)
library(viridisLite)

# take a subset of the dataset diamonds
set.seed(1867)
d <- diamonds[sample(nrow(diamonds), 1500),]
p <- ggplot(d, aes(x, price)) +
     stat_density_2d(geom = "raster", aes(fill = ..density..), 
                     contour = FALSE) + 
#     geom_density_2d() + 
     facet_grid(. ~ cut) +
     scale_fill_gradientn(colours = viridis(256, option = "D")) +
     ggtitle('Diamonds per cut type')
p
# ---------------------------------------------------------------------


---
title: "R Markdown"
author: "Philippe De Brouwer"
date: "January 1, 2020"
output: beamer_presentation
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = FALSE)
```

## R Markdown
This is an R Markdown presentation. Markdown is a simple formatting 
syntax for authoring HTML, PDF, and MS Word documents. For more 
details on using R Markdown see <http://rmarkdown.rstudio.com>.

When you click the **Knit** button a document will be generated 
that includes both content as well as the output of any embedded R 
code chunks within the document.

## Slide with Bullets
- Bullet 1
- Bullet 2
- Bullet 3

## Slide with R Output
```{r cars, echo = TRUE}
summary(cars)
```

## Slide with Plot
```{r pressure}
plot(pressure)
```
# ---------------------------------------------------------------------


# Level 1: title slide

## level 2: title or new slide
This line goes on slide the slide with title 'level 2: title 
or new slide'

### level 3: box on slide
This line goes in the body of the box that will have the title
'level 3: box on slide'
# ---------------------------------------------------------------------


$
# ---------------------------------------------------------------------


## 50 random numbers
```{r showPlot}
hist(runif(50))
```
# ---------------------------------------------------------------------


library(knitr);                       # load the knitr package
setpwd('/path/to/my/file')            # move to the appropriate directory
knit("latex_article.Rnw")             # creat a .tex file by executing all R-code
system('pdflatex latex_article.tex')  # compile the .tex file to a .pdf
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------


# install.packages('shiny') # if necessary of course
library(shiny)
runExample("01_hello")
# ---------------------------------------------------------------------


# The filename must be app.R to execute it on a server.

# The name of the server function must match the argument in the 
# shinyApp function also it must take the arguments input and output.
server <- function(input, output) {
  output$distPlot <- renderPlot({
    # any plot must be passed through renderPlot()
    hist(rnorm(input$nbr_obs), col = 'khaki3', border = 'white', 
         breaks = input$breaks,
         main   = input$title,
         xlab   = "random observations")
    })
 output$nbr_txt <- renderText({ 
   paste("You have selected", input$nbr_obs, "observations.") 
 })  # note the use of brackets ({ })
}

# The name of the ui object must match the argument in the shinyApp
# function and we must provide an object ui (that holds the html 
# code for our page).
ui <- fluidPage(
  titlePanel("Our random simulator"),
  sidebarLayout(
    sidebarPanel(
      sliderInput("nbr_obs", "Number of observations:", 
                  min = 10, max = 500, value = 100),
      sliderInput("breaks", "Number of bins:", 
                  min = 4, max = 50, value = 10),
      textInput("title", "Title of the plot", 
                value = "title goes here")
    ),
    mainPanel(plotOutput("distPlot"),
              h4("Conclusion"),
	          p("Small sample sizes combined with a high number of bins
                might provide a visual image of the distribution that does 
                not resemble the underlying dynamics."),
             "Note that we can provide text, but not  <b>html code</b> directly.",
             textOutput("nbr_txt")  # object name in quotes
	      )
  )
)

# finally we call the shinyApp function
shinyApp(ui = ui, server = server)
# ---------------------------------------------------------------------


# Load the library:
library(rsconnect)

# Upload the app (default filename is app.R)
rsconnect::deployApp('path/to/your/app', 
                     server="shinyapps.io", account="xxx")
# ---------------------------------------------------------------------


library(leaflet)
content <- paste(sep = "<br/>",
  "<b><a href='http://www.de-brouwer.com/honcon/'>Honorary Consulate of Belgium</a></b>",
  "ul. Marii Grzegorzewskiej 33A",
  "30-394 Krakow"
  )
map <- leaflet()                                    %>%
  addProviderTiles(providers$OpenStreetMap)         %>%
  addMarkers(lng = 19.870188, lat = 50.009159)      %>%
  addPopups(lng = 19.870188, lat = 50.009159, content,
    options = popupOptions(closeButton = TRUE))     %>%
  setView(lat = 50.009159,lng = 19.870188, zoom = 12)
map
# ---------------------------------------------------------------------


library(titanic)    # for the data
library(tidyverse)  # for the tibble
library(ggvis)      # for the plot

titanic_train$Age     %>% 
    as_tibble         %>%
    na.omit           %>%
    ggvis(x = ~value) %>%
    layer_densities(
      adjust = input_slider(.1, 2, value = 1, step = .1, 
                            label = "Bandwidth"),
      kernel = input_select(
        c("Gaussian"     = "gaussian",
          "Epanechnikov" = "epanechnikov",
          "Rectangular"  = "rectangular",
          "Triangular"   = "triangular",
          "Biweight"     = "biweight",
          "Cosine"       = "cosine",
          "Optcosine"    = "optcosine"),
        label = "Kernel")
    )
# ---------------------------------------------------------------------


library(ggvis)

function(input, output, session) {
  # A reactive subset of mtcars:
  reacCars <- reactive({ mtcars[1:input$n, ] })

  # Register observers and place the controls:
  reacCars %>%  
     ggvis(~wt, ~mpg, fill=(~cyl)) %>%  
     layer_points() %>%
     layer_smooths(span = input_slider(0.5, 1, value = 1, 
        label = 'smoothing span:')) %>%
     bind_shiny("plot1", "plot_ui_div")

  output$carsData <- renderTable({ reacCars()[, c("wt", "mpg")] })
}
# ---------------------------------------------------------------------


library(ggvis)

fluidPage(sidebarLayout(
  sidebarPanel(
    # Explicit code for a slider-bar:
    sliderInput("n", "Number of points", min = 1, max = nrow(mtcars),
                value = 10, step = 1),
    # No code needed for the smoothing span, ggvis does this:
    uiOutput("plot_ui_div") # produces a <div> with id corresponding
                            # to argument in bind_shiny
  ),
  mainPanel(
    # Place the plot "plot1" here:
    ggvisOutput("plot1"),    # matches argument to bind_shiny()
    # under this the table of selected card models:
    tableOutput("carsData")  # parses the result of renderTable()
  )
))
# ---------------------------------------------------------------------


shiny::runApp("/path/to/my/app")
# ---------------------------------------------------------------------


library(googleVis)
demo(package='googleVis')
# ---------------------------------------------------------------------


# diversity
# Calculates the entropy of a system with equiprobable states.
# Arguments:
#    x -- numeric vector -- observed probabilities of classes
# Returns:
#    numeric -- the entropy / diversity measure
diversity <- function(x) {
  f <- function(x) x * log(x)
  x1 <- mapply(FUN = f, x)
  - sum(x1) / log(length(x))
  }
# ---------------------------------------------------------------------


# diversity
# Calculates the entropy of a system with discrete states.
# Arguments:
#   x     -- numeric vector -- observed probabilities of classes
#   prior -- numeric vector -- prior probabilities of the classes
# Returns:
#    numeric -- the entropy / diversity measure
diversity <- function(x, prior = NULL) {
  if (min(x) <= 0) {return(0);} # the log will fail for 0
  # If the numbers are higher than 1, then not probabilities but 
  # populations sizes are given, so we rescale to probabilities:
  if (sum(x) != 1) {x <- x / sum(x)}
  N <- length(x)
  if(!is.null(prior)) {
    for (i in (1:N)) {
      a <- (1 - 1 / (N * prior[i])) / (1 - prior[i])
      b <- (1 - N * prior[i]^2) / (N * prior[i] * (1 - prior[i]))
      x[i] <- a * x[i]^2 + b * x[i]
    }
   }
  f <- function(x) x * log(x)
  x1 <- mapply(FUN = f, x)
  - sum(x1) / log(N)    # this is the value that is returned
  }
# ---------------------------------------------------------------------


# Consider the following prior probabilities:
pri <- c(0.1,0.5,0.4)

# No prior priorities supplied, so 1/N is most diverse:
diversity(c(0.1,0.5,0.4))      

# The population matches prior probabilities, so index should be 1:
diversity(c(0.1,0.5,0.4), prior = pri) 

# Very non-diverse population:
diversity(c(0.999,0.0005,0.0005), prior = pri) 

# Only one sub-group is represented (no diversity):
diversity(c(1,0,0), prior = pri)         

# Numbers instead of probabilities provided, also this works:
diversity(c(100,150,200)) 
# ---------------------------------------------------------------------


females <- seq(from = 0, to = 1, length.out = 100)
div <- numeric(0)
for (i in (1:length(females))) {
  div[i] <- diversity (c(females[i], 1 - females[i]))
  }
  
d <- as.data.frame(cbind(females, div)  )
colnames(d) <- c('percentage females', 'diversity index')
library(ggplot2)
p <- ggplot(data = d, 
        aes(x = `percentage females`, y = `diversity index`)) +
     geom_line(color = 'red', lwd = 3) + 
     ggtitle('Diversity Index') + 
     xlab('percentage females') + ylab('diversity index')
p
# ---------------------------------------------------------------------


library(tidyverse)
N <- 200
set.seed(1866)

d0 <- data.frame("ID"      = 1:N,

          # Log-normal age distribution
          "age"         = round(rlnorm(N, log(30), log(1.25))),
	  # A significant bias towards the old continent:
          "continent"   = ifelse(runif(N) < 0.3, "America", 
	                   ifelse(runif(N) < 0.7,"Europe","Other")),
          # A mild bias towards males:
          "gender"      = ifelse(runif(N) < 0.45, "F", "M"),
	  # Grade will be filled in later:
          "grade"       = 0,
	  # Three teams of different sizes:
          "team"        = ifelse(runif(N) < 0.6, "bigTeam", 
	                    ifelse(runif(N) < 0.6, 
                            "mediumTeam", 
		            ifelse(runif(N) < 0.8, "smallTeam", 
			           "XsmallTeam"))),
          # Most people have little people depending on them:
          "dependents"  = round(rlnorm(N,log(0.75),log(2.5))),
	  # Random performance (no bias linked, but different group sizes):
          "performance" = ifelse(runif(N) < 0.1, "L", 
	                    ifelse(runif(N) < 0.6, "M", 
                            ifelse(runif(N) < 0.7, "H", "XH"))),
          # Salary will be filled in later:
          "salary"      = 0,
	  # We make just a snapshot dashboard, so we do not need this now, 
	  # but we could use this later to show evolution:
          "timestamp"   = as.Date("2020-01-01")
          )

# Now we clean up age and fill in grade, salary and lastPromoted without 
# any bias for gender, origin -- but with a bias for age.
d1 <- d0                                                  %>%
  mutate(age    = ifelse((age < 18), age + 10, age))      %>%
  mutate(grade  = ifelse(runif(N) * age < 20, 0, 
                    ifelse(runif(N) * age < 25, 1, 
                    ifelse(runif(N) * age < 30, 2, 3))))  %>%
  mutate(salary = round(exp(0.75*grade)*4000 + 
                    rnorm(N,0,1500)))                     %>%
  mutate(lastPromoted = round(exp(0.05*(3-grade))*1 + 
                    abs(rnorm(N,0,5))) -1)
# ---------------------------------------------------------------------


# If not done yet, install the package:
# install.packages('flexdashboard')

# Then load the package:
library(flexdashboard)
# ---------------------------------------------------------------------


---
title: "Divsersity in Action"
output: 
  flexdashboard::flex_dashboard:
    theme: cosmo
    orientation: rows
    vertical_layout: fill
    #storyboard: true
    social: menu
    source: embed
---

```{r}
# (C) Philippe J.S. De Brouwer -- 2019
# demo: http://rpubs.com/phdb/diversity_dash01
```

```{r setup, include=FALSE}
library(flexdashboard)
library(tidyverse)
library(ggplot2)
library(knitr)
library(gridExtra)
#install.packages('plotly')
library(plotly)
N <- 150
set.seed(1865)

d0 <- data.frame("ID"         = 1:N,
                "age"         = round(rlnorm(N, log(30), log(1.25))),
                "continent"   = ifelse(runif(N) < 0.3, "America", ifelse(runif(N) < 0.7, "Europe","Other")),
                "gender"      = ifelse(runif(N) < 0.4, "F", "M"),
                "grade"       = 0,
                "team"        = ifelse(runif(N) < 0.6, "bigTeam", ifelse(runif(N) < 0.6, 
                               "mediumTeam", ifelse(runif(N) < 0.8, "smallTeam", "XsmallTeam"))),
                "dependents"  = round(rlnorm(N,log(0.65),log(1.5))),
                "performance" = ifelse(runif(N) < 0.1, "L", ifelse(runif(N) < 0.6, "M", 
                                ifelse(runif(N) < 0.7, "H", "XH"))),
                "salary"      = 0,
                "timestamp"   = as.Date("2020-01-01")
                )

d1 <- d0 %>%
  mutate(age    = ifelse((age < 18), age + 10, age)) %>%
  mutate(grade  = ifelse(runif(N) * age < 20, 0, ifelse(runif(N) * age < 25, 1, ifelse(runif(N) * age < 30, 2, 3)))) %>%
  mutate(salary = round(exp(0.75*grade)*4000 + rnorm(N,0,2500)))  %>%
  mutate(lastPromoted = round(exp(0.05*(3-grade))*1 + abs(rnorm(N,0,5))) -1)
```

Overview
========

Row 
-------------------------------------

```{r}
# our diversity function
diversity <- function(x, prior = NULL) {
  if (min(x) <= 0) {return(0);} # the log will fail for 0
  # if the numbers are higher than 1, then not probabilities but 
  # populations are given, so we rescale to probabilities:
  if (sum(x) != 1) {x <- x / sum(x)}
  N <- length(x)
  if(!is.null(prior)) {
    for (i in (1:N)) {
      a <- (1 - 1 / (N * prior[i])) / (1 - prior[i])
      b <- (1 - N * prior[i]^2) / (N * prior[i] * (1 - prior[i]))
      x[i] <- a * x[i]^2 + b * x[i]
    }
   }
  f <- function(x) x * log(x)
  x1 <- mapply(FUN = f, x)
  - sum(x1) / log(N)
}
# the gauges for the different dimensions
```

### Gender 
```{r}
# ranges:
rGreen <- c(0.900001, 1)
rAmber <- c(0.800001, 0.9)
rRed   <- c(0, 0.8)
iGender <- round(diversity(table(d1$gender)),3)
gauge(iGender, min = 0, max = 1, gaugeSectors(
  success = rGreen, warning = rAmber, danger = rRed
  ))
kable(table(d1$gender))
```

### Age
```{r}
# consider each band of ten years as a group
iAge <- round(diversity(table(round(d1$age/10))),3)
gauge(iAge, min = 0, max = 1, gaugeSectors(
  success = rGreen, warning = rAmber, danger = rRed
  ))
kable(table(round(d1$age/10)*10))
```


### Roots
```{r}
iRoots <- round(diversity(table(d1$continent)),3)
gauge(iRoots, min = 0, max = 1, gaugeSectors(
  success = rGreen, warning = rAmber, danger = rRed
  ))
kable(table(d1$continent))
```

### Dependents
```{r}
# we only monitor if someone has dependents
xdep <- ifelse(d1$dependents >= 1, 1, 0)
iDep <- round(diversity(table(xdep)),3)
gauge(iDep, min = 0, max = 1, gaugeSectors(
  success = rGreen, warning = rAmber, danger = rRed
  ))When the aforementioned code is executed, 
kable(table(d1$dependents))
```

Row
-------------------------------------
### Info 
The diversity indeces show how diverse our workforce is. They are calculated similar to entropy: $I = -\frac{1}{\log(N)} \sum_i^N {p_i \log p_i}$, where there are $N$ possible and mutually exclusive states $i$. They range from $0$ to $1$.

### Average Diversity Index
```{r}
x  <- mean(c(iGender, iAge, iDep, iRoots))
valueBox(x, 
         icon  = ifelse(x > 0.9, "fa-smile" , ifelse(x > 0.8, "fa-meh", "fa-sad-tear")),
         color = ifelse(x > 0.9, "success" , ifelse(x > 0.8, "warning", "danger"))
         )
```


Gender
======================================

Row {.tabset}
-------------------------------------

### Composition
```{r}
p2 <- ggplot(data = d1, aes(x=gender, fill=gender)) +
  geom_bar(stat="count", width=0.7) + 
  facet_grid(rows=d1$grade) + 
  ggtitle('workforce composition i.f.o. salary grade (level in the company)')
ggplotly(p2)
```

### Salary
```{r}
p1 <- ggplot(data = d1, aes(x=gender, y=salary, fill=gender)) +
  geom_boxplot() + 
  facet_grid(rows=d1$grade) + 
  ggtitle('The salary gap per salary grade (level in the company)')
ggplotly(p1)
```

### Promotions
```{r}
d1$promoted = ifelse(d1$lastPromoted <= 2,1,0)
p <- ggplot(data = d1, aes(x=gender, fill=gender, y=promoted)) +
  stat_summary(fun.y="mean", geom="bar") +
  facet_grid(rows=d1$grade) + 
  ggtitle('promotion propensity per grae')
p
ggplotly(p)
```

Age
===
Row {.tabset}
-------------------------------------

### Histogram
    
```{r}
qplot(d1$age, geom="histogram", binwidth=5,
        fill=I("steelblue"), col=I("black")
        ) +
  xlab('Age') +
  ggtitle('Histogram for Age')
```

Roots {.tabset}
===============
Row {.tabset}
-------------------------------------

### rworldmap
    
```{r warning=FALSE}When the aforementioned code is executed, 
# the default R-approach:
install.packages('rworldmap')
library(rworldmap)

nbrPerCountry = read.table(text="
country value
Poland 100
Ukraine 65
UK 2
USA 1
China 3
Germany 0
France 1
Italy 20
Greece 25
Spain 13
Portugal 7
Mexico 55
Belarus 5
Russia 7
Vietnam 1
India 25
Belgium 2
Chile 6
Congo 1
", header=T)

x <- joinCountryData2Map(nbrPerCountry, joinCode="NAME", nameJoinColumn="country")

mapCountryData(x, nameColumnToPlot="value", catMethod="fixedWidth")
```

### leaflet
```{r}
#install.packages('maps')
library(maps)
#install.packages('sp')
library(sp)
#install.packages('leaflet')
library(leaflet)
map <- leaflet() %>%
  addProviderTiles(providers$OpenStreetMap) %>%
  setView(lng = 0, lat = 0, zoom = 2)
map
```

Dependents
==========
Row {.tabset}
-------------------------------------

### Histogram
    
```{r}
qplot(d1$dependents, geom="histogram", binwidth=1,
        fill=I("steelblue"), col=I("black")
        ) +
  xlab('Dependents') +
  ggtitle('Histogram for number of dependents')
```
# ---------------------------------------------------------------------


---
title: "Diversity in Action"
output: 
  flexdashboard::flex_dashboard:
    theme: cosmo
    orientation: rows
    vertical_layout: fill
    social: menu
    source: embed
---

```{r setup, include=FALSE}
# load packages and functions
```
Overview
========
Row 
-------------------------------------
### Gender
```{r}
code to generate the first gauge
```

Row
-------------------------------------

Gender
======================================

Row {.tabset}
-------------------------------------

Age
======================================
Row {.tabset}
-------------------------------------
### first tab
### second tab

Roots
======================================
Row {.tabset}
-------------------------------------

Dependants
======================================
Row {.tabset}
-------------------------------------
# ---------------------------------------------------------------------


Overview
========

Row 
-------------------------------------

```{r}
# here goes the R-code to get data and calculate diversity indices.
```

### Gender 
```{r genderGauge}
# ranges:
rGreen   <- c(0.900001, 1)
rAmber   <- c(0.800001, 0.9)
rRed     <- c(0, 0.8)
iGender  <- round(diversity(table(d1$gender)),3)
gauge(iGender, min = 0, max = 1, gaugeSectors(
      success = rGreen, warning = rAmber, danger = rRed
      ))
kable(table(d1$gender))
```
### This is only the first gauge, the next one can be described below.
# ---------------------------------------------------------------------


gauge <-  gvisGauge(iGender, 
             options=list(min = 0, max = 1, 
	             greenFrom = 0.9,greenTo = 1, 
		     yellowFrom = 0.8, yellowTo = 0.8, 
		     redFrom = 0, redTo = 0.8, 
		     width = 400, height = 300))
# ---------------------------------------------------------------------


Gender
======================================

Row {.tabset}
-------------------------------------

### Composition
```{r}
p2 <- ggplot(data = d1, aes(x=gender, fill=gender)) +
  geom_bar(stat="count", width=0.7) + 
  facet_grid(rows=d1$grade) + 
  ggtitle('workforce composition i.f.o. salary grade')
ggplotly(p2)
```

### Tab2
```{r RCodeForTab2}
# etc.
```
# ---------------------------------------------------------------------


# this file should be called app.R
library(shinydashboard)

# general code (not part of server or ui function)
my_seed <- 1865
set.seed(my_seed)
N <- 150

# user interface ui
ui <- dashboardPage(
  dashboardHeader(title = "ShinyDashBoard"),
  dashboardSidebar(
      title = "Choose ...",
      numericInput('N', 'Number of data points:', N),
      numericInput('my_seed', 'seed:', my_seed),
      sliderInput("bins", "Number of bins:",
                  min = 1, max = 50, value = 30)
  ),
  dashboardBody(
    fluidRow(
      valueBoxOutput("box1"),
      valueBoxOutput("box2")
      ),
    plotOutput("p1", height = 250)
    )
  )
# ---------------------------------------------------------------------


# server function
server <- function(input, output) {
  d <- reactive({
    set.seed(input$my_seed)
    rnorm(input$N)
    })
  output$p1 <- renderPlot({
    x    <- d()
    bins <- seq(min(x), max(x), length.out = input$bins + 1)
    hist(x, breaks = bins, col = 'deepskyblue3', border = 'gray')
    })
  output$box1 <- renderValueBox({
    valueBox(
      value    = formatC(mean(d()), digits = 2, format = "f"),
      subtitle = "mean", icon     = icon("globe"),
      color    = "light-blue")
    })
  output$box2 <- renderValueBox({
      valueBox(
        value    = formatC(sd(d()), digits = 2, format = "f"),
        subtitle = "standard deviation",  icon     = icon("table"),
        color    = "light-blue")
      })
  
}

# load the app
shinyApp(ui, server)
