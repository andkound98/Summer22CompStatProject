###############################################################################
# Trees, Forests and the Classification Problem: 
# Simulations & Application to Redistribution Preferences 
# Andreas Koundouros | Computational Statistics Project | Summer Term 2022
###############################################################################
# R Code
# Main File
###############################################################################





###############################################################################
###############################################################################
###############################################################################
# Preliminaries 
###############################################################################
###############################################################################
###############################################################################

# Install (if necessary) and load the following required packages
suppressMessages(library(MASS))             # For multivariate data simulation
suppressMessages(library(dplyr))            # For data wrangling
suppressMessages(library(rpart))            # For tree estimation
suppressMessages(library(rpart.plot))       # For tree plotting
suppressMessages(library(randomForest))     # For random forests 
suppressMessages(library(foreign))          # For importing STATA data files
suppressMessages(library(latex2exp))        # For TeX in plots
suppressMessages(library(tictoc))           # For measuring running time
rm(list=ls())                               # Clean the current environment 
seed <- 123                                 # Seed used throughout for reproducibility 
setwd("~/Documents/Uni Master SS22/Computational Statistics/Project Redistribution Preferences/Project Final/Code_and_Data")
# NOTE: Set this working directory as appropriate
source("R_Functions.R") # Source function file





###############################################################################
###############################################################################
###############################################################################
# Part 1: Simulation Studies 
# (Section 3 in the Jupyter Notebook)
###############################################################################
###############################################################################
###############################################################################

# Simulation Study 1 ###########################################################
tictoc::tic()###################################################################
N <- 1200 # Number of observations
error_mn <- 0 # Mean of the error terms 
error_sd <- 10 # Standard deviation of the error terms
p <- 3 # Number of features 
mu <- c(rep(0, times = p)) # Vector of means of the features
beta <- c(rep(0.5, times = p)) # Some coefficient vector 
reps <- 300 # Last value in loop
reps.vec <- seq(from = 1, to = reps, by = 4) # Sequence for the simulations
noseed <- 100 # Number of seeds
errors.tr <- matrix(data = NA, nrow = noseed, ncol = length(reps.vec)) 
# CT container
errors.rf <- matrix(data = NA, nrow = noseed, ncol = length(reps.vec)) 
# RF container
for (s in 1:noseed) {# Loop over seeds
  for (t in reps.vec) { # Simulation 
    Sigma <- matrix(data = c(1,           1/(t^2),         1/sqrt(t),
                             1/(t^2),     1,               1/sqrt(t^(1/2)), 
                             1/sqrt(t),   1/sqrt(t^(1/2)), 1), # Correlation
                    # matrix as the diagonal is normalised to unity; the rest of 
                    # the terms go to zero with varying speeds
                    ncol = 3, nrow = 3, byrow = TRUE) # Sequence of matrices with 
    # decreasing correlation of features; if p > 3, Sigma needs to be adjusted  
    set.seed(s)
    dt <- data.gen.sim1(obs = N, x.mean = mu, x.cov = Sigma, 
                        error.mean = error_mn, error.sd = error_sd, 
                        coeff = beta) # Obtain data from data generator 
    train.i <- sample(1:nrow(dt), (dim(dt)[1]/2), replace = FALSE) # Split data
    train.data <- dt[train.i,] # Training data
    test.data <- dt[-train.i,] # Test data
    
    res.tr <- tree.fun(form = (y.cat ~ .), dt.train = train.data, # CT 
                       dt.test = test.data, test.Y = test.data$y.cat)
    
    opt.m <- rf.opt.m(p = length(mu), equation = (y.cat ~ .), 
                      dt.tr = train.data, dt.te = test.data, 
                      te.y = test.data$y.cat) # Find optimal RF
    res.rf <- rf.fun(form = (y.cat ~ .), dt.train = train.data, 
                     dt.test = test.data, test.Y = test.data$y.cat, imp = FALSE, 
                     m = opt.m) # Optimal RF 
    
    k <- match(t, reps.vec) # In order to be able to store the results correctly 
    errors.tr[s, k] <- res.tr$misclass.error # Misclassification rate CT
    errors.rf[s, k] <- res.rf$misclass.error # Misclassification rate RF
  }
}
plot(reps.vec, colMeans(errors.tr), type = "b", las = 1, col = "orange", 
     xlab = "Decreasing Correlation", ylab = "Misclassification Rate", lwd = 2,
     ylim = c(0.475, 
              max(colMeans(errors.tr), colMeans(errors.rf)))) # Plot CT's
# misclassification rate depending on the features' correlation
lines(reps.vec, colMeans(errors.rf), type = "b", lwd = 2, col = "deepskyblue")#RF
legend("bottomright", legend = c("Classification Tree", "Random Forest"),
       col = c("orange", "deepskyblue"), lwd = 2, cex = 1) # Legend
text(27, 0.4752, "0.4517") # Mark the outlier 
arrows(x0 = 12, y0 = 0.478, x1 = 12, y1 = 0.4751, code = 2)
tictoc::toc()###################################################################
beep()

# Simulation Study 2 ###########################################################
tictoc::tic()###################################################################
N <- 1200 # Number of observations
error_mn <- 0 # Mean for normally distributed errors
error_sd <- 10 # Standard deviation for normally distributed errors
p <- 3 # Number of features
mu <- c(rep(0, times = p)) # Vector of means of the features
Sigma <- matrix(data = c(1, 0.3, -0.1, 0.3, 1, 0.9, -0.1, 0.9, 1), nrow = p, 
                ncol = p, byrow = TRUE) 
# Positive definite covariance matrix; if p > 3,Sigma needs to be adjusted 
beta <- c(rep(0.5, times = p)) # Some coefficient vector 
J <- 15 # Maximum number of categories in the outcome
noseed <- 200 # Number of seeds
errors.tr <- matrix(data = NA, nrow = noseed, ncol = (J-1)) # CT container
errors.rf <- matrix(data = NA, nrow = noseed, ncol = (J-1)) # RF container
for (s in 1:noseed) { # Loop over seeds
  set.seed(s)
  data <- data.gen.sim2(obs = N, x.mean = mu, x.cov = Sigma, 
                        error.mean = error_mn, error.sd = error_sd,
                        coeff = beta) # Data with normally distributed errors
  for (t in 2:J) { # Loop over number of categories in the outcome
    dt <- make.cat(dt = data, nocat = t) # Create t categories in the outcome 
    train.i <- sample(1:nrow(dt), (dim(dt)[1]/2), replace = FALSE) # Split data
    train.data <- dt[train.i,] # Training data
    test.data <- dt[-train.i,] # Test data
    
    cnt1 <- train.data %>% count(y.cat, .drop = FALSE) # Count observations per 
    # class in training data 
    cnt2 <- test.data %>% count(y.cat, .drop = FALSE) # Count observations per 
    # class in test data 
    cnt <- rbind(cnt1, cnt2) 
    if(0 %in% cnt[,2] == TRUE){ # If empty classes, skip estimations
      errors.tr[s, t-1] <- NA
      errors.rf[s, t-1] <- NA
    }else{ # If no empty classes, perform estimations
      res.tr <- tree.fun(form = (y.cat ~ .), dt.train = train.data, 
                         dt.test = test.data, test.Y = test.data$y.cat) # CT 
      
      opt.m <- rf.opt.m(p = length(mu), equation = (y.cat ~ .), 
                        dt.tr = train.data, dt.te = test.data, 
                        te.y = test.data$y.cat)
      res.rf <- rf.fun(form = (y.cat ~ .), dt.train = train.data, 
                       dt.test = test.data, test.Y = test.data$y.cat,
                       imp = FALSE, m = opt.m) # Optimal RF 
      
      errors.tr[s, t-1] <- res.tr$misclass.error # Misclassification rate CT
      errors.rf[s, t-1] <- res.rf$misclass.error # Misclassification rate RF 
    }
  }
}
errors.mean <- rbind(colMeans(na.omit(errors.tr)), colMeans(na.omit(errors.rf)))
y.limits <- c(min(errors.mean), max(errors.mean))
plot(2:J, errors.mean[1,], type = "b", las = 1, lwd = 2, col = "orange",
     xlab = "Number of Categories in y", ylab = "Misclassification Rate", 
     ylim = y.limits) # Plot the misclassification rate of CT depending on 
# the number of categories in y
lines(2:J, errors.mean[2,], type = "b", col = "deepskyblue", lwd = 2) # Add RF
legend("topleft", legend = c("Classification Tree", "Random Forest"),
       col = c("orange", "deepskyblue"), lwd = 2, cex = 1.3) # Legend
tictoc::toc()###################################################################
beep()

# Simulation Study 3 ###########################################################
tictoc::tic()###################################################################
N <- 1200 # Number of observations
error_mn <- 0 # Mean for normally distributed errors
error_sd <- 10 # Standard deviation for normally distributed errors
beta <- c(rep(0.5, times = 4)) # Some coefficient vector 
noseed <- 100 # Number of seeds
errors.sim3 <- matrix(data = NA, nrow = noseed, ncol = 4, dimnames = 
                   list(c(1:noseed), c("CTnoX_4","CTwX_4","RFnoX_4","RFwX_4")))
# Container for results
for (s in 1:noseed) { # Loop over seeds
  set.seed(s)
  dt <- data.gen.sim3(obs = N, error.mean = error_mn, error.sd = error_sd,
                      coeff = beta)
  train.i <- sample(1:nrow(dt), (dim(dt)[1]/2), replace = FALSE) # Split data
  train.data <- dt[train.i,] # Training data 
  test.data <- dt[-train.i,] # Test data
  
  # Without X[,4]
  res.tr.wo4 <- tree.fun(form = (y.cat ~ . - num), dt.train = train.data, 
                     dt.test = test.data, test.Y = test.data$y.cat) # CT 
  
  opt.m.wo4 <- rf.opt.m(p = (length(beta) - 1), equation = (y.cat ~ . - num), 
                    dt.tr = train.data, dt.te = test.data, 
                    te.y = test.data$y.cat) # Optimal m
  res.rf.wo4 <- rf.fun(form = (y.cat ~ . - num), dt.train = train.data, 
                   dt.test = test.data, test.Y = test.data$y.cat, 
                   imp = FALSE, m = opt.m.wo4) # Optimal RF 
  
  errors.sim3[s, 1] <- res.tr.wo4$misclass.error # Misclassification rate CT
  errors.sim3[s, 3] <- res.rf.wo4$misclass.error # Misclassification rate RF 
  
  # With X[,4]
  res.tr.w4 <- tree.fun(form = (y.cat ~ .), dt.train = train.data, 
                     dt.test = test.data, test.Y = test.data$y.cat) # CT
  
  opt.m.w4 <- rf.opt.m(p = (length(beta)), equation = (y.cat ~ .), 
                        dt.tr = train.data, dt.te = test.data, 
                        te.y = test.data$y.cat) # Optimal m
  res.rf.w4 <- rf.fun(form = (y.cat ~ .), dt.train = train.data, 
                      dt.test = test.data, test.Y = test.data$y.cat, 
                      imp = FALSE, m = opt.m.w4) # Optimal RF
  
  errors.sim3[s, 2] <- res.tr.w4$misclass.error # Misclassification rate CT
  errors.sim3[s, 4] <- res.rf.w4$misclass.error # Misclassification rate RF 
}
boxplot(errors.sim3, las = 1, col = c(rep("orange", 2), rep("deepskyblue", 2)),
        names = c(expression("CT w/o"~X[4]), expression("CT with"~X[4]), 
                  expression("RF w/o"~X[4]), expression("RF with"~X[4])))
tictoc::toc()###################################################################
beep()





###############################################################################
###############################################################################
###############################################################################
# Part 2: Economic Application:Redistribution Preferences 
# (Section 4 in the Jupyter Notebook)
###############################################################################
###############################################################################
###############################################################################

# Loading and Cleaning the GSS Data ############################################
gss.data.full <- read.dta("./Data/GSS2018.dta")
gss.data <- select(gss.data.full, c("sex", "race", "reg16", "born", "parborn", 
                                    "madeg", "relig16", "fund16", "age",
                                    "realinc", "eqwlth"))
gss.data <- na.omit(gss.data) # Omit NAs
gss.data <- droplevels(gss.data) # Drop unused levels
rownames(gss.data) <- 1:nrow(gss.data) # Re-brand observation numbering
gss.data$eqwlth <- factor(gss.data$eqwlth) # Convert EQWLTH into factor

# Splitting the Data into Training and Test Data ###############################
set.seed(seed)
train.i <- sample(1:nrow(gss.data), (dim(gss.data)[1]/2), replace = FALSE)
gss.train.data <- gss.data[train.i,] # Training data 
gss.test.data <- gss.data[-train.i,] # Test data

# Classification Tree (without REALINC) ########################################
tr.gss <- tree.fun(form = (eqwlth ~ . - realinc), dt.train = gss.train.data, 
                   dt.test = gss.test.data, test.Y = gss.test.data$eqwlth, 
                   plot = TRUE)
print(1 - tr.gss$misclass.error) # Share of correct classifications

# Random Forest (without REALINC) ##############################################
set.seed(seed)
rf.gss <- rf.fun(form = (eqwlth ~ . - realinc), dt.train = gss.train.data, 
                 dt.test = gss.test.data, test.Y = gss.test.data$eqwlth, 
                 m = 1, own.vi.plot = TRUE) # m = 1 from appendix B
print(1 - rf.gss$misclass.error) # Share of correct classifications

# Classification Tree (with REALINC) ###########################################
tr.gss.rinc <- tree.fun(form = (eqwlth ~ .), dt.train = gss.train.data, 
                        dt.test = gss.test.data, test.Y = gss.test.data$eqwlth, 
                        plot = TRUE)
print(1 - tr.gss.rinc$misclass.error) # Share of correct classifications

# Random Forest (with REALINC) #################################################
set.seed(seed)
rf.gss.rinc <- rf.fun(form = (eqwlth ~ .), dt.train = gss.train.data, 
                      dt.test = gss.test.data, test.Y = gss.test.data$eqwlth, 
                      m = 1, own.vi.plot = TRUE) # m = 1 from appendix B
print(1- rf.gss.rinc$misclass.error) # Share of correct classifications





###############################################################################
###############################################################################
###############################################################################
# Part 3: Appendix
# (Appendix Section in the Jupyter Notebook)
###############################################################################
###############################################################################
###############################################################################

# Overview of the data #########################################################
head(gss.data) # The first 6 observations of the clean data set 
summary(gss.data) # Get summary of the included variables 

# Summary Statistics of the Data ###############################################
gss.sumstats <- data.matrix(gss.data[sapply(gss.data, is.factor)])
round(colMeans(gss.sumstats), 2) # Means
round(apply(gss.sumstats, 2, sd), 2) # Standard deviations

# Summary Statistics for Section 3.3 ###########################################
table(gss.data$race)[1]/dim(gss.data)[1] # Share of ethnicity "white"
table(gss.data$race)[2]/dim(gss.data)[1] # Share of ethnicity "black"
min(gss.data$age) # Minimum age 
max(gss.data$age) # Maximum age 
mean(gss.data[gss.data$sex == "female", "realinc"]) # Mean family income of women 
mean(gss.data[gss.data$sex == "male", "realinc"]) # Mean family income of men
sd(gss.data[gss.data$sex == "female", "realinc"]) # Standard deviation family 
# income of women
sd(gss.data[gss.data$sex == "male", "realinc"]) # Standard deviation family
# income of men

# Find Optimal m in RF (without REALINC) #######################################
p <- 9
res.m <- rep(NA, times = p)
for (q in 1:p) {
  set.seed(seed)
  rf.m <- rf.fun(form = (eqwlth ~ . - realinc), dt.train = gss.train.data, 
                 dt.test = gss.test.data, test.Y = gss.test.data$eqwlth, 
                 m = q)
  res.m[q] <- rf.m$misclass.error
}
min.m <- match(min(res.m), res.m)
print(min.m)

# Find Optimal m in RF (with REALINC) ##########################################
p <- 10
res.m.rinc <- rep(NA, times = p)
for (q in 1:p) { 
  set.seed(seed)
  rf.m.rinc <- rf.fun(form = (eqwlth ~ .), dt.train = gss.train.data, 
                      dt.test = gss.test.data, test.Y = gss.test.data$eqwlth, 
                      m = q)
  res.m.rinc[q] <- rf.m.rinc$misclass.error
}
min.m.rinc <- match(min(res.m.rinc), res.m.rinc)
print(min.m.rinc)
