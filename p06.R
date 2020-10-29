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


r_w <- 0.05              # the interest rate per week 
r_y <- (1 + r_w)^52 - 1  # assume 52 weeks per year
paste0("the APR is: ", round(r_y * 100, 2),"%!")
# ---------------------------------------------------------------------


r   <- 0.1
CFs <- c(-100, 100, 100)
t   <- c(   0,   5,   7)
NPV <- sum(CFs / (1 + r)^t)
print(round(NPV, 2))
# ---------------------------------------------------------------------


# bond_value
# Calculates the fair value of a bond
# Arguments:
#    time_to_mat -- time to maturity in years
#    coupon      -- annual coupon in $
#    disc_rate   -- discount rate (risk free + risk premium)
#    nominal     -- face value of the bond in $
# Returns:
#    the value of the bond in $
bond_value <- function(time_to_mat, coupon, disc_rate, nominal){
  value  <- 0
  # 1/ all coupons
  for (t in 1:time_to_mat) {
     value <- value + coupon * (1 + disc_rate)^(-t)
     }
  # 2/ end payment of face value
  value <- value + nominal * (1 + disc_rate)^(-time_to_mat)
  value
}
# ---------------------------------------------------------------------


# The fair value of the bond is then:
bond_value(time_to_mat = 5, coupon = 5, disc_rate = 0.03, 
           nominal = 100)
# ---------------------------------------------------------------------


bond_value(time_to_mat = 5, coupon = 5, disc_rate = 0.035, 
           nominal = 100)
# ---------------------------------------------------------------------


V <- bond_value(time_to_mat = 5, coupon = 5, disc_rate = 0.03, 
           nominal = 100)
CFs <- c(seq(5, 5, length.out=4), 105)
t   <- c(1:5)
r   <- 0.03
MacD <- 1/V * sum(t * CFs / (1 + r)^t)
print(MacD)
# ---------------------------------------------------------------------


R_A <- 0.02 + 1.25 * (0.10 - 0.02)
print(paste0('The RR for company A is: ', 
             round(R_A, 2) * 100, '%'))
# ---------------------------------------------------------------------


R_B <- 0.02 + 0.75 * (0.10 - 0.02)
print(paste0('The RR for B is: ', round(R_B, 2) * 100, '%'))

# Additionally, we also compare this with previous example:
print(paste0('The beta changed by ',
             round((0.75 / 1.25 - 1) * 100, 2), 
             '% and the RR by ', 
	     round((R_B / R_A - 1) * 100, 2), '%.'))
# ---------------------------------------------------------------------


V_0 <- 10 * (1 + 0.00) / (0.01 + 0.05 - 0.00)
print(round(V_0,2))
# ---------------------------------------------------------------------


V_0 <- 10 * (1 + 0.02) / (0.01 + 0.05 -0.02)
print(round(V_0,2))
# ---------------------------------------------------------------------


V_0 <- 10 * (1 + 0.02) / (0.01 + 1.5 * 0.05 - 0.02)
print(round(V_0,2))
# ---------------------------------------------------------------------


V_0 <- 10 * (1 + 0.02) / (0.01 + 1.5 * 0.05 - 0.10)
print(round(V_0,2))
# ---------------------------------------------------------------------


# Let us plot the value of the call in function of the strike
FS <- seq(80, 120, length.out=150) # future spot price
X  <- 100                          # strike
P  <- 5                            # option premium
T  <- 3                            # time to maturity
r  <- 0.03                         # discount rate
payoff <- mapply(max, FS-X, 0)
profit <- payoff - P * (1 + r)^T

# Plot the results:
plot(FS, payoff,
     col='red', lwd = 3, type='l',
     main='LONG CALL value at maturity',
     xlab='Future strike price',
     ylab='$',
     ylim=c(-10,20)
     )
lines(FS, profit,
      col='blue', lwd=2)
text(105,8, 'Payoff', col='red')
text(115,5, 'Profit', col='blue')
# ---------------------------------------------------------------------


FS <- seq(80, 120, length.out=150) # future spot price
X  <- 100                          # strike
P  <- 5                            # option premium
T  <- 3                            # time to maturity
r  <- 0.03                         # discount rate
payoff <- - mapply(max, FS-X, 0)
profit <- P * (1 + r)^T + payoff

# Plot the results:
plot(FS, payoff,
     col='red', lwd = 3, type='l',
     main='SHORT CALL value at maturity',
     xlab='Future spot price',
     ylab='$',
     ylim=c(-20,10)
     )
lines(FS, profit,
      col='blue', lwd=2)
text(90,1.5, 'Payoff', col='red')
text(90,7, 'Profit', col='blue')
# ---------------------------------------------------------------------


FS <- seq(80, 120, length.out=150) # future spot price
X  <- 100                          # strike
P  <- 5                            # option premium
T  <- 3                            # time to maturity
r  <- 0.03                         # discount rate

# the long put:
payoff <- mapply(max, X - FS, 0)
profit <- payoff - P * (1 + r)^T

par(mfrow=c(1,2))

plot(FS, payoff,
     col='red', lwd = 3, type='l',
     main='LONG PUT at maturity',
     xlab='Future spot price',
     ylab='$',
     ylim=c(-20,20)
     )
lines(FS, profit,
      col='blue', lwd=2)
text(110,1,  'Payoff', col='red')
text(110,-4, 'Profit', col='blue')

# the short put:
payoff <- - mapply(max, X - FS, 0)
profit <- payoff + P * (1 + r)^T

plot(FS, payoff,
     col='red', lwd = 3, type='l',
     main='SHORT PUT at maturity',
     xlab='Future spot price',
     ylab='',
     ylim=c(-20,20)
     )
lines(FS, profit,
      col='blue', lwd=2)
text(110,1, 'Payoff', col='red')
text(110,6, 'Profit', col='blue')

par(mfrow=c(1,1))  # reset the plot interface
# ---------------------------------------------------------------------


# call_intrinsicVal
# Calculates the intrinsic value for a call option
# Arguments:
#    Spot   -- numeric -- spot price
#    Strike -- numeric -- the strike price of the option
# Returns
#    numeric -- intrinsic value of the call option.
call_intrinsicVal <- function(Spot, Strike) {max(Spot - Strike, 0)}
# ---------------------------------------------------------------------


# put_intrinsicVal
# Calculates the intrinsic value for a put option
# Arguments:
#    Spot   -- numeric -- spot price
#    Strike -- numeric -- the strike price of the option
# Returns
#    numeric -- intrinsic value of the put option.
put_intrinsicVal <- function(Spot, Strike) {max(-Spot + Strike, 0)}
# ---------------------------------------------------------------------


# call_price
# The B&S price of a call option before maturity
# Arguments:
#    Spot   -- numeric -- spot price in $ or %
#    Strike -- numeric -- the strike price of the option  in $ or %
#    T      -- numeric -- time to maturity in years
#    r      -- numeric -- interest rates (e.g. 0.02 = 2%)
#    vol    -- numeric -- standard deviation of underlying in $ or %
# Returns
#    numeric -- value of the call option in $ or %
#
call_price <- function (Spot, Strike, T, r, vol)
 {
  d1 <- (log(Spot / Strike) + (r + vol ^ 2/2) * T) / (vol * sqrt(T))
  d2 <- (log(Spot / Strike) + (r - vol ^ 2/2) * T) / (vol * sqrt(T))
  pnorm(d1) * Spot - pnorm(d2) * Strike * exp(-r * T)
  }
# ---------------------------------------------------------------------


# put_price
# The B&S price of a put option before maturity
# Arguments:
#    Spot   -- numeric -- spot price in $ or %
#    Strike -- numeric -- the strike price of the option  in $ or %
#    T      -- numeric -- time to maturity in years
#    r      -- numeric -- interest rates (e.g. 0.02 = 2%)
#    vol    -- numeric -- standard deviation of underlying in $ or %
# Returns
#    numeric -- value of the put option in $ or %
#
put_price <- function(Spot, Strike, T, r, vol)
 {
 Strike * exp(-r * T) - Spot + call_price(Spot, Strike, T, r, vol)
 }
# ---------------------------------------------------------------------


# Examples:
call_price (Spot = 100, Strike = 100, T = 1, r = 0.02, vol = 0.2)
put_price  (Spot = 100, Strike = 100, T = 1, r = 0.02, vol = 0.2)
# ---------------------------------------------------------------------


# Long call
spot <- seq(50,150, length.out=150)
intrinsic_value_call <- apply(as.data.frame(spot), 
                              MARGIN=1, 
			      FUN=call_intrinsicVal, 
			      Strike=100)
market_value_call    <- call_price(Spot = spot, Strike = 100, 
                                   T = 3, r = 0.03, vol = 0.2)
                                   
# Plot the results:
plot(spot, market_value_call,
     type = 'l', col= 'red', lwd = 4,
     main = 'European Call option',
     xlab = 'Spot price',
     ylab = 'Option value')
text(115, 40, 'Market value', col='red')
lines(spot, intrinsic_value_call,
      col= 'forestgreen', lwd = 4)
text(130,15, 'Intrinsic value', col='forestgreen')
# ---------------------------------------------------------------------


# Long put
spot <- seq(50,150, length.out=150)
intrinsic_value_put <- apply(as.data.frame(spot), 
                             MARGIN=1, 
			     FUN=put_intrinsicVal, 
			     Strike=100)
market_value_put    <- put_price(Spot = spot, Strike = 100, 
                                 T = 3, r = 0.03, vol = 0.2)
                                 
# Plot the results:
plot(spot, market_value_put,
     type = 'l', col= 'red', lwd = 4,
     main = 'European Put option',
     xlab = 'Spot price',
     ylab = 'Option value')
text(120, 8, 'market value', col='red')
lines(spot, intrinsic_value_put,
      col= 'forestgreen', lwd = 4)
text(75,10, 'intrinsic value', col='forestgreen')
# ---------------------------------------------------------------------


# CRR_price
# Calculates the CRR binomial model for an option
# Arguments:
#         S0 -- numeric -- spot price today (start value)
#         SX -- numeric -- strike, e.g. 100
#      sigma -- numeric -- the volatility over the maturity period, 
#                          e.g. ca. 0.2 for shares on 1 yr
#        Rrf -- numeric -- the risk free interest rate (log-return)
# optionType -- character -- 'lookback' for lookback option, 
#                            otherwise vanilla call is assumed
#    maxIter -- numeric -- number of iterations 
# Returns:
#    numeric -- the value of the option given parameters above
CRR_price <- function(S0, SX, sigma, Rrf, optionType, maxIter)
  {
  Svals <- mat.or.vec(2^(maxIter), maxIter+1)
  probs <- mat.or.vec(2^(maxIter), maxIter+1)
  Smax  <- mat.or.vec(2^(maxIter), maxIter+1)
  Svals[1,1] <- S0
  probs[1,1] <- 1
  Smax[1,1]  <- S0
  dt <-  1 / maxIter
  u  <- exp(sigma * sqrt(dt))
  d  <- exp(-sigma * sqrt(dt))
  p  = (exp(Rrf * dt) - d) / (u - d)
  for (n in 1:(maxIter))
    {
    for (m in 1:2^(n-1))
     {
     Svals[2*m-1,n+1] <- Svals[m,n] * u
     Svals[2*m,n+1]   <- Svals[m,n] * d
     probs[2*m-1,n+1] <- probs[m,n] * p
     probs[2*m,n+1]   <- probs[m,n] * (1 - p)
     Smax[2*m-1,n+1]  <- max(Smax[m,n], Svals[2*m-1,n+1])
     Smax[2*m,n+1]    <- max(Smax[m,n], Svals[2*m,n+1])
     }
    }
  if (optionType == 'lookback')
   {
     exp.payoff <- (Smax - SX)[,maxIter + 1]  * probs[,maxIter + 1]
   }  # lookback call option
   else
    {
     optVal <- sapply(Svals[,maxIter + 1] - SX,max,0)
     exp.payoff <- optVal * probs[,maxIter + 1]
    }  # vanilla call option
  sum(exp.payoff) / (1 + Rrf)
  }
# ---------------------------------------------------------------------


library(ggplot2)
# Plot the convergence of the CRR algorithm for a call option.
plot_CRR("Call", maxIter = 20)
# ---------------------------------------------------------------------


# Plot the convergence of the CRR algorithm for a call option.
plot_CRR("lookback", maxIter = 15)
# ---------------------------------------------------------------------


# We still use ggplot2
library(ggplot2)
# ---------------------------------------------------------------------


# plot_price_evol
# Plots the evolution of Call price in function of a given variable
# Arguments:
#    var       -- numeric   -- vector of values of the variable
#    varName   -- character -- name of the variable to be studied
#    price     -- numeric   -- vector of prices of the option
#    priceName -- character -- the name of the option
#    reverseX  -- boolean   -- TRUE to plot x-axis from high to low
# Returns
#    ggplot2 plot
#
plot_price_evol <- function(var, varName, price, priceName, 
                            reverseX = FALSE)
{
  d <- data.frame(var, price)
  colnames(d) <- c('x','y')
  p <- qplot(x, y, data = d, geom = "line", size = I(2) )
  p <- p + geom_line()
  if (reverseX) {p <- p + xlim(max(var), min(var))}  # reverse axis
  p <- p + xlab(varName ) + ylab(priceName)
  p   # return the plot
}
# ---------------------------------------------------------------------


# Define the default values:
t      <- 1
Spot   <- 100
Strike <- 100
r      <- log(1 + 0.03)
vol    <- 0.2


## ... time
T <- seq(5, 0.0001, -0.01)
Call <- c(call_price (Spot, Strike, T, r, vol))
p1 <- plot_price_evol(T, "Time to maturity (years)", Call, "Call", 
                      TRUE)

## ... interest
R <- seq(0.001, 0.3, 0.001)
Call <- c(call_price (Spot, Strike, t, R, vol))
p2 <- plot_price_evol(R, "Interest rate", Call, "Call")

## ... volatility
vol <- seq(0.00, 0.2, 0.001)
Call <- c(call_price (Spot, Strike, t, r, vol))
p3 <- plot_price_evol(vol, "Volatility", Call, "Call")

## ... strike
X <- seq(0, 200, 1)
Call <- c(call_price (Spot, X, t, r, vol))
p4 <- plot_price_evol(X, "Strike", Call, "Call")

## ... Spot
spot <- seq(0, 200, 1)
Call <- c(call_price (spot, Strike, t, r, vol))
p5 <- plot_price_evol(spot, "Spot price", Call, "Call")

# In the next line we use the function grid.arrange()
# from the gridExtra package
library(gridExtra)
grid.arrange(p1, p2, p3, p4, p5, nrow = 3)
# ---------------------------------------------------------------------


# Define the default values:
t      <- 1
Spot   <- 100
Strike <- 100
r      <- log(1 + 0.03)
vol    <- 0.2


## ... time
T <- seq(5, 0.0001, -0.01)
Call <- c(put_price (Spot, Strike, T, r, vol))
p1 <- plot_price_evol(T, "Time to maturity (years)", 
                      Call, "Call", TRUE)

## ... interest
R <- seq(0.001, 0.3, 0.001)
Call <- c(put_price (Spot, Strike, t, R, vol))
p2 <- plot_price_evol(R, "Interest rate", Call, "Call")

## ... volatility
vol <- seq(0.00, 0.2, 0.001)
Call <- c(put_price (Spot, Strike, t, r, vol))
p3 <- plot_price_evol(vol, "Volatility", Call, "Call")

## ... strike
X <- seq(0, 200, 1)
Call <- c(put_price (Spot, X, t, r, vol))
p4 <- plot_price_evol(X, "Strike", Call, "Call")

## ... Spot
spot <- seq(0, 200, 1)
Call <- c(put_price (spot, Strike, t, r, vol))
p5 <- plot_price_evol(spot, "Spot price", Call, "Call")

# In the next line we use the function grid.arrange()
# from the gridExtra package
library(gridExtra)
grid.arrange(p1, p2, p3, p4, p5, nrow = 3)
# ---------------------------------------------------------------------


# Define the functions to calculate the price of the delta

# call_delta
# Calculates the delta of a call option
# Arguments:
#    S      -- numeric -- spot price
#    Strike -- numeric -- strike price
#    T      -- numeric -- time to maturity
#    r      -- numeric -- interest rate
#    vol    -- numeric -- standard deviation of underlying
call_delta  <- function (S, Strike, T, r, vol)
 {
  d1 <- (log (S / Strike)+(r + vol ^2 / 2) * T) / (vol * sqrt(T))
  pnorm(d1)
  }


# put_delta
# Calculates the delta of a put option
# Arguments:
#    S      -- numeric -- spot price
#    Strike -- numeric -- strike price
#    T      -- numeric -- time to maturity
#    r      -- numeric -- interest rate
#    vol    -- numeric -- standard deviation of underlying
put_delta  <- function (S, Strike, T, r, vol)
 {
  d1 <- (log (S / Strike)+(r + vol ^2 / 2) * T) / (vol * sqrt(T))
  pnorm(d1) - 1
  }


## DELTA CALL
spot <- seq(0,200, 1)
delta <- c(call_delta(spot, Strike, t, r, vol))
p1 <- plot_price_evol(spot, "Spot price", delta, "Call delta")

## DELTA PUT
spot <- seq(0,200, 1)
delta <- c(put_delta(spot, Strike, t, r, vol))
p2 <- plot_price_evol(spot, "Spot price", delta, "Put delta")

# plot the two visualizations:
grid.arrange(p1, p2, nrow = 2)
# ---------------------------------------------------------------------


# load the plotting library
library(ggplot2)
# ---------------------------------------------------------------------


# portfolio_plot
# Produces a plot of a portfolio of the value in function of the 
# spot price of the underlying asset.
# Arguments:
#   portf            - data.frame - composition of the portfolio
#                   with one row per option, structured as follows:
#                       - ['long', 'short'] - position
#                       - ['call', 'put']   - option type
#                       - numeric           - strike
#                       - numeric           - gearing (1 = 100%)
#   structureName="" - character - label of the portfolio
#   T = 1            - numeric   - time to maturity (in years)
#   r = log(1 + 0.02) - numeric   - interest rate (per year) 
#                                    as log-return
#   vol = 0.2        - numeric   - annual volatility of underlying
#   spot.min = NULL  - NULL for automatic scaling x-axis, value for min
#   spot.max = NULL  - NULL for automatic scaling x-axis, value for max
#   legendPos=c(.25,0.6) - numeric vector - set to 'none' to turn off
#   yLims = NULL     - numeric vector - limits y-axis, e.g. c(80, 120)
#   fileName = NULL  - character - filename, NULL for no saving
#   xlab = "default" - character - x axis label, NULL to turn off
#   ylab = "default" - character - y axis label, NULL to turn off
# Returns (as side effect)
#   ggplot plot
#   pdf file of this plot (in subdirectory ./img/)

portfolio_plot <- function(portf, 
	   structureName="", # name of the option strategy
	   T = 1,            # time to maturity (in years)
	   r = log(1 + 0.02),# interest rate (per year)
	   vol = 0.2,        # annual volatility of the underlying
	   spot.min = NULL,  # NULL for automatic scaling x-axis
	   spot.max = NULL,  # NULL for automatic scaling x-axis
	   legendPos=c(.25,0.6), # set to 'none' to turn off
	   yLims = NULL,     # limits of y-axis, e.g. c(80, 120)
	   fileName = NULL,  # NULL for no saving plot into file
	   xlab = "default", # set to NULL to turn off (or string)
	   ylab = "default"  # set to NULL to turn off (or string)
	   ) {
# portf = data frame with: long/short, call/put, strike, gearing

the_S = 100       # The spot price today is always 100
# 
  strikes <- as.numeric(portf[,3])
  strike.min <- min(strikes)
  strike.max <- max(strikes)
  if (is.null(spot.min)) {
    spot.min <- min(0.8*strike.min, max(0,2*strike.min - strike.max))
    }
  if (is.null(spot.max)) {
    spot.max <- max(1.2 * strike.max, 2 * strike.max - strike.min)}
  if (structureName == ""){
    structureName<- paste(deparse(substitute(fileName)), 
                          collapse = "", sep="")
    }
  nbrObs   <- 200
  spot     <- seq(spot.min,spot.max,len=nbrObs)
  val.now  <- seq(0,0,len=nbrObs)
  val.end  <- seq(0,0,len=nbrObs)
  for (k in 1:nrow(portf))
    {
     Strike  <- as.numeric(portf[k,3])
     gearing <- as.numeric(portf[k,4])
     if (portf[k,1] == 'long'){theSign <- 1}else{theSign = -1}
     if (portf[k,2] == 'call')
       {
        purchasePrice <- call_price(the_S, Strike, T, r, vol)
        callVal  <- sapply(spot, call_price, Strike=Strike, T=T, 
	                   r=r, vol=vol)
        val.now.incr <- callVal - purchasePrice 
        val.end.incr <- sapply(spot, call_intrinsicVal, 
	                Strike = Strike) - purchasePrice
       }
       else
       {
        if (portf[k,2] == 'put')
          {
          purchasePrice <- put_price(the_S, Strike, T, r, vol)
          callVal  <- sapply(spot, put_price, Strike=Strike, T=T, 
	                     r=r, vol=vol)
          val.now.incr <- callVal - purchasePrice 
          val.end.incr <- sapply(spot, put_intrinsicVal, 
	                  Strike = Strike) - purchasePrice
          }
          else # then it is 'underlying'
          {
          val.now.incr <- spot - Strike
          val.end.incr <- spot - Strike
          }
       }
     val.now <- val.now + val.now.incr * gearing * theSign
     val.end <- val.end + val.end.incr * gearing * theSign
     }
  d1 <- data.frame(spot, val.end, 
          paste('intrinsic value',structureName,sep=" "), 3)
  d2 <- data.frame(spot, val.now,  
          paste('value 1 year to maturity',structureName,sep=" "), 2)
  colnames(d1) <- c('spot', 'value', 'legend')
  colnames(d2) <- c('spot', 'value', 'legend')
  dd <- rbind(d1,d2)
  p <- qplot(spot, value, data=dd, color = legend, 
             geom = "line",size=I(2) )
  if(is.null(xlab)) {
      p <- p + theme(axis.title.x = element_blank())
      } else { 
	if(xlab == "default") {p <- p + xlab('spot price')
	   } else {p <- p + xlab(xlab)}}
  if(is.null(ylab)) {
      p <- p + theme(axis.title.y = element_blank())
      } else { 
	if(ylab == "default") {p <- p + ylab('Value')
	   } else {p <- p + ylab(ylab)}}
  p <- p + ylab('Value')
  p <- p + theme(legend.position=legendPos)
  if(legendPos == "none") {p <- p + ggtitle(structureName)}
  if (!is.null(yLims)) {p <- p + scale_y_continuous(limits=yLims)}
  if(!is.null(fileName)) {
    # remove punctuation:
    fileName <- str_replace_all(fileName, "[[:punct:]]", "")
    # remove spaces:
    fileName <- str_replace_all(fileName, " ", "")            
    # save file in sub-directory img
    ggsave(paste('img/',fileName,'.pdf',sep=''), 
           width = 6, height = 3)
    }
  # return the plot:
  p
}
# ---------------------------------------------------------------------


# long call
portfolio <- rbind(c('long','call',100,1))
p1 <- portfolio_plot(portfolio, 'Long call', 
                     legendPos="none", xlab = NULL)

# short call
portfolio <- rbind(c('short','call',100,1))
p2 <- portfolio_plot(portfolio, 'Short call', legendPos="none", 
                     xlab = NULL, ylab = NULL)

# long put
portfolio <- rbind(c('long','put',100,1))
p3 <- portfolio_plot(portfolio, 'Long put', legendPos="none", 
                     xlab=NULL)

# short put
portfolio <- rbind(c('short','put',100,1))
p4 <- portfolio_plot(portfolio, 'Short put', legendPos="none", 
                     xlab = NULL, , ylab = NULL)

# -- long call and short put
portfolio <- rbind(c('long','call',100,1))
portfolio <- rbind(portfolio, c('short','put',100,1))
p5 <- portfolio_plot(portfolio, 'Long call + short put', 
                     legendPos="none", xlab = NULL)

# -- call
portfolio <- rbind(c('long','call',100,1))
p6 <- portfolio_plot(portfolio, 'Call', legendPos="none", 
                     xlab = NULL, ylab = NULL)

# -- put
portfolio <- rbind(c('long','put',100,1))
p7 <- portfolio_plot(portfolio, 'Put', legendPos="none")

# -- callput
portfolio <- rbind(c('short','put',100,1))
portfolio <- rbind(portfolio, c('long','call',100,1))
p8 <- portfolio_plot(portfolio, 'Call + Put', legendPos="none",
                     ylab = NULL)


# show all visualizations:
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, nrow = 4)
# ---------------------------------------------------------------------


# -- callspread
portfolio <- rbind(c('short','call',120,1))
portfolio <- rbind(portfolio, c('long','call',100,1))
p1 <- portfolio_plot(portfolio, 'CallSpread',
                     legendPos="none", xlab = NULL)

# -- short callspread
portfolio <- rbind(c('long','call',120,1))
portfolio <- rbind(portfolio, c('short','call',100,1))
p2 <- portfolio_plot(portfolio, 'Short allSpread', 
                     legendPos="none", xlab = NULL, ylab = NULL)

# -- callspread differently
portfolio <- rbind(c('short','put',120,1))
portfolio <- rbind(portfolio, c('long','put',100,1))
p3 <- portfolio_plot(portfolio, 'Short putSpread',
                     legendPos="none", xlab = NULL)

# -- putspread
portfolio <- rbind(c('short','put',80,1))
portfolio <- rbind(portfolio, c('long','put',100,1))
p4 <- portfolio_plot(portfolio, 'PutSpread',
                     legendPos="none", xlab = NULL, ylab = NULL)

# -- straddle
portfolio <- rbind(c('long','call',100,1))
portfolio <- rbind(portfolio, c('long','put',100,1))
p5 <- portfolio_plot(portfolio, 'Straddle', spot.min = 50, 
                     spot.max = 150,legendPos="none", xlab = NULL)
# Note that our default choices for x-axis range are not suitable 
# for this structure. Hence, we add spot.min and spot.max

# -- short straddle
portfolio <- rbind(c('short','call',100,1))
portfolio <- rbind(portfolio, c('short','put',100,1))
p6 <- portfolio_plot(portfolio, 'Short straddle',spot.min = 50, 
                     spot.max = 150, legendPos="none", 
		     xlab = NULL, ylab = NULL)

# -- strangle
portfolio <- rbind(c('long','call',110,1))
portfolio <- rbind(portfolio, c('long','put',90,1))
p7 <- portfolio_plot(portfolio, 'Strangle', 
                     spot.min = 50, spot.max = 150,
                     legendPos="none", xlab = NULL)

# -- butterfly
portfolio <- rbind(c('long','call',120,1))
portfolio <- rbind(portfolio, c('short','call',100,1))
portfolio <- rbind(portfolio, c('long','put',80,1))
portfolio <- rbind(portfolio, c('short','put',100,1))
p8 <- portfolio_plot(portfolio, 'Butterfly',
                     spot.min = 50, spot.max = 150,
                     legendPos="none", xlab = NULL, ylab = NULL)


# show all visualizations:
grid.arrange(p1, p2, p3, p4, p5, p6, p7, p8, nrow = 4)
# ---------------------------------------------------------------------


# -- condor
portfolio <- rbind(c('long','call',140,1))
portfolio <- rbind(portfolio, c('short','call',120,1))
portfolio <- rbind(portfolio, c('long','put',60,1))
portfolio <- rbind(portfolio, c('short','put',80,1))
p1 <- portfolio_plot(portfolio, 'Condor',spot.min = 40, 
                     spot.max = 160, legendPos="none", 
		     xlab = NULL)

# -- short condor
portfolio <- rbind(c('short','call',140,1))
portfolio <- rbind(portfolio, c('long','call',120,1))
portfolio <- rbind(portfolio, c('short','put',60,1))
portfolio <- rbind(portfolio, c('long','put',80,1))
p2 <- portfolio_plot(portfolio, 'Short Condor',spot.min = 40, 
                     spot.max = 160, legendPos="none", 
		     xlab = NULL, ylab = NULL)

# -- geared call
portfolio <- rbind(c('long','call',100.0,2))
p3 <- portfolio_plot(portfolio, 
                     structureName="Call with a gearing of 2",
		     legendPos="none", xlab = NULL)

# -- nearDigital (approximate a digital option with a geared call)
portfolio <- rbind(c('short','call',100.1,10))
portfolio <- rbind(portfolio, c('long','call',100,10))
p4 <- portfolio_plot(portfolio, 'Near digital',
                     legendPos="none", xlab = NULL, ylab = NULL)

# -- a complex structure:
portfolio <- rbind(c('long','call',110,1))
portfolio <- rbind(portfolio, c('short','call',105,1))
portfolio <- rbind(portfolio, c('short','put',95,1))
portfolio <- rbind(portfolio, c('long','put',90,1))
portfolio <- rbind(portfolio, c('long','put',80,1))
portfolio <- rbind(portfolio, c('long','call',120,1))
portfolio <- rbind(portfolio, c('short','call',125,1))
portfolio <- rbind(portfolio, c('short','put',70,10))
portfolio <- rbind(portfolio, c('short','put',75,1))
portfolio <- rbind(portfolio, c('short','call',130,10))
portfolio <- rbind(portfolio, c('long','call',99,10))
portfolio <- rbind(portfolio, c('short','call',100,10))
portfolio <- rbind(portfolio, c('short','put',100,10))
portfolio <- rbind(portfolio, c('long','put',101,10))
p5 <- portfolio_plot(portfolio, 'Fun',legendPos='none', 
                     spot.min=60, spot.max=140,
		     yLims=c(-0,25))

# show all visualizations:
# Pasing a layout_matrix to the function grid.arrange()
# allows to make the lower plot bigger:
layoutM <- rbind(c(1,2),
                 c(3,4),
                 c(5,5))
grid.arrange(p1, p2, p3, p4, p5, nrow = 3, layout_matrix = layoutM)
# ---------------------------------------------------------------------


## --- covered call ----

nbrObs     <- 100
the.S      <- 100
the.Strike <- 100
the.r      <- log (1 + 0.03)
the.T      <- 1
the.vol    <- 0.2
Spot.min   = 80
Spot.max   = 120
LegendPos  = c(.5,0.2)
Spot   <- seq(Spot.min,Spot.max,len=nbrObs)
val.end.call <-  - sapply(Spot, call_intrinsicVal, Strike = 100)
call.value <- call_price(the.S, the.Strike, the.T, the.r, the.vol)

d.underlying <- data.frame(Spot, Spot - 100,  'Underlying',   1)
d.shortcall  <- data.frame(Spot, val.end.call,  'Short call', 1)
d.portfolio  <- data.frame(Spot, 
                           Spot + val.end.call + call.value - 100,
			   'portfolio', 1.1)
colnames(d.underlying) <- c('Spot', 'value', 'Legend','size')
colnames(d.shortcall)  <- c('Spot', 'value', 'Legend','size')
colnames(d.portfolio)  <- c('Spot', 'value', 'Legend','size')
dd <- rbind(d.underlying, d.shortcall, d.portfolio)
p <- qplot(Spot, value, data = dd, color = Legend, geom = "line",
           size=size )
p <- p + xlab('Value of the underlying' ) + ylab('Profit at maturity')
p <- p + theme(legend.position = LegendPos)
p <- p + scale_size(guide = 'none')
print(p) 
# ---------------------------------------------------------------------


## --- married put ----

LegendPos = c(.8,0.2)
Spot        <- seq(Spot.min,Spot.max,len=nbrObs)
val.end.put <-  sapply(Spot, put_intrinsicVal, Strike = 100)
put.value   <- - put_price(the.S, the.Strike, the.T, the.r, the.vol)

d.underlying <- data.frame(Spot, Spot - 100,  'Underlying', 1)
d.shortput   <- data.frame(Spot, val.end.put,  'Long put',  1)
d.portfolio  <- data.frame(Spot, 
                           Spot + val.end.put + put.value - 100,
			   'portfolio',
			   1.1)
colnames(d.underlying) <- c('Spot', 'value', 'Legend','size')
colnames(d.shortput)   <- c('Spot', 'value', 'Legend','size')
colnames(d.portfolio)  <- c('Spot', 'value', 'Legend','size')
dd <- rbind(d.underlying,d.shortput,d.portfolio)
p  <- qplot(Spot, value, data = dd, color = Legend, geom = "line",
            size = size )
p <- p + xlab('Value of the underlying' ) + ylab('Profit at maturity')
p <- p + theme(legend.position = LegendPos)
p <- p + scale_size(guide = 'none')
print(p) 
# ---------------------------------------------------------------------


## --- collar ----

# Using the same default values as for the previous code block:
LegendPos = c(.6,0.25)
Spot         <-   seq(Spot.min,Spot.max,len=nbrObs)
val.end.call <- - sapply(Spot, call_intrinsicVal, Strike = 110)
val.end.put  <- + sapply(Spot, put_intrinsicVal, Strike = 95)
call.value   <- call_price(the.S, the.Strike, the.T, the.r, the.vol)
put.value    <- put_price(the.S, the.Strike, the.T, the.r, the.vol)

d.underlying <- data.frame(Spot, Spot - 100,   'Underlying', 1)
d.shortcall  <- data.frame(Spot, val.end.call, 'Short call', 1)
d.longput    <- data.frame(Spot, val.end.put,  'Long call',  1)
d.portfolio  <- data.frame(Spot, Spot + val.end.call + call.value + 
                     val.end.put - put.value - 100, 'portfolio',1.1)
colnames(d.underlying) <- c('Spot', 'value', 'Legend','size')
colnames(d.shortcall)  <- c('Spot', 'value', 'Legend','size')
colnames(d.longput)    <- c('Spot', 'value', 'Legend','size')
colnames(d.portfolio)  <- c('Spot', 'value', 'Legend','size')
dd <- rbind(d.underlying,d.shortcall,d.longput,d.portfolio)
p  <- qplot(Spot, value, data=dd, color = Legend, geom = "line",
            size=size )
p <- p + xlab('Value of the underlying' ) + ylab('Profit at maturity')
p <- p + theme(legend.position = LegendPos)
p <- p + scale_size(guide = 'none')
print(p) 
# ---------------------------------------------------------------------


##--------- the example of the capital protected structure
N           <- 5
nominal     <- 1000
inDeposit   <- nominal * (1.02)^(-N)
cst         <- 0.01               # in PERCENT
pvCosts     <- N * cst * nominal  # one should rather use the present value here
rest4option <- 1000 - inDeposit - pvCosts
callPrice   <- call_price (100, Strike = 100, T = 5, r = 0.02, vol = 0.02)
# reformulate this price as a percentage and then adjust to nominal
callPrice   <- callPrice / 100 * 1000   
gearing     <- rest4option / callPrice
paste('The gearing is:', round(gearing * 100, 2))
