###############################################################################
# Trees, Forests and the Classification Problem: 
# Simulations & Application to Redistribution Preferences 
# Andreas Koundouros | Computational Statistics Project | Summer Term 2022
###############################################################################
# R Code
# Function File
###############################################################################





###############################################################################
###############################################################################
###############################################################################
# Data Generating Processes (DGP)
###############################################################################
###############################################################################
###############################################################################


# Simulation 1 #################################################################
data.gen.sim1 <- function(obs, x.mean, x.cov, error.mean, error.sd, coeff){
  # Data generator for simulation 1
  
  # obs: number of observations; x.mean: vector of the means of the features; 
  # x.cov: variance-covariance matrix of the features; error.mean: mean of the 
  # error term; error.sd: standard deviation of the error term; coeff: vector
  # of coefficients 
  
  X <- mvrnorm(n = obs, mu = x.mean, Sigma = x.cov, empirical = TRUE) # Generate 
  # continuous, multivariate normally distributed features
  error <- rnorm(n = obs, mean = error.mean, sd = error.sd) # Generate error terms
  
  y <- X[,1]*(X[,2] + X[,3] - X[,3]^2) + X %*% coeff + error # Non-linear DGP
  y.cat <- factor(ifelse(y >= mean(y), "1", "0")) # Categorical target
  
  data <- data.frame("y.cat" = y.cat, "X" = X)
  return(data)
}

# Simulation 2 #################################################################
data.gen.sim2 <- function(obs, x.mean, x.cov, 
                          error.mean, error.sd, 
                          coeff){
  # Data generator for simulation 2
  
  # obs: number of observations; x.mean: vector of the means of the features;
  # x.cov: variance-covariance matrix of the features; error.norm: TRUE/FALSE 
  # depending on whether errors shall be normally distributed or not; 
  # error.unif: TRUE/FALSE depending on whether errors shall be uniformly 
  # distributed or not; error.mean: mean of the errors in case of normal 
  # distribution; error.sd: standard deviation of the error term in case of 
  # normal distribution; error.min: minimum of errors in case of uniform 
  # distribution; error.max: maximum of error in case of uniform distribution;
  # coeff: vector of coefficients
  
  X <- mvrnorm(n = obs, mu = x.mean, Sigma = x.cov) # Generate continuous 
  # features
  error <- rnorm(n = obs, mean = error.mean, sd = error.sd) # Generate normally
  # distributed error terms 
  
  y <- X[,1]*(X[,2] + X[,3] - X[,3]^2) + X %*% coeff + error # Non-linear DGP
  
  data <- data.frame("y" = y, "X" = X)
  return(data)
}

make.cat <- function(dt, nocat){ 
  # Create categorical target from numerical variable with "nocat" many
  # categories 
  
  # dt: data set; nocat > 2: number of categories in the response
  
  rg <- range(dt$y)
  sq <- seq(rg[1], rg[2], length = (nocat + 1))
  cat <- as.factor(c(1:nocat))
  y.cat <- c(rep(NA, times = length(dt$y)))
  for (i in 1:length(dt$y)) {
    for (j in 1:nocat) {
      if(dt$y[i] >= sq[j] & dt$y[i] <= sq[j + 1]){
        y.cat[i] <- cat[j]
      }
    }
  }
  
  y.cat <- as.factor(y.cat) # Convert y.cat into factor
  data <- data.frame(dt, "y.cat" = y.cat)
  
  cnt <- data %>% count(y.cat) # Count the observations per constructed class
  if(0 %in% cnt[,2]){ 
    print("Warning: empty categories!") # If there is a category with zero
    # observations, the code does not return the data but issues a warning  
  }else{return(data)} # If all categories have at least one observation each, 
  # then the code returns the data 
}

# Simulation 3 #################################################################
data.gen.sim3 <- function(obs, 
                          prob.wh = 0.74, prob.bl = 0.17, 
                          int.min = 18, int.max = 90, 
                          num.mean.m = 3.4, num.mean.w = 3.3, 
                          num.sd.m = 0.8, num.sd.w = 0.7,
                          error.mean, error.sd, 
                          coeff){ 
  # Data generator for simulation 3
  
  # obs: number of observations; prob.wh/prob.bl: theoretical shares of 
  # "wh"/"bl" in the population; int.min/int.max: minimum and maximum value of 
  # the integer feature; num.mean.m/num.mean.w: theoretical means of the 
  # continuous feature conditional on "m"/"w"; num.sd.m/num.sd.w: theoretical
  # standard deviations of the continuous feature conditional on "m"/"w"
  
  cat1 <- sort(sample(c("m","w"), size = obs, replace = TRUE)) # Binary feature
  cat2 <- sample(c("wh", "bl", "ot"), size = obs, replace = TRUE, 
                 prob = c(prob.wh, prob.bl, (1-prob.wh-prob.bl))) # Categorical 
  # feature 
  int <- sample(x = int.min:int.max, size = obs, replace = TRUE) # Integer-valued 
  # feature
  num.m <- rlnorm(n = table(cat1)[1], meanlog = num.mean.m, sdlog = num.sd.m)*1000
  num.w <- rlnorm(n = table(cat1)[2], meanlog = num.mean.w, sdlog = num.sd.w)*1000
  num <- round(c(num.m, num.w), 2) # Continuous feature with distribution 
  # conditional on the binary feature 
  
  X <- data.frame("cat1" = factor(cat1), "cat2" = factor(cat2), 
                         "int" = int, "num" = num) # Create features with 
  # default values
  error <- rnorm(n = obs, mean = error.mean, sd = error.sd) # Generate normally
  # distributed error terms 
  
  num.X <- data.frame("cat1" = as.numeric(X[,1]), "cat2" = as.numeric(X[,2]), 
                      "int" = as.numeric(X[,3]), "num" = X[,4]) 
  num.X <- as.matrix(num.X) # Convert features to make numeric calculations
  
  y <- (1/1000)*(num.X[,1]*(num.X[,2] + num.X[,3]^(5/2) - num.X[,4]) + num.X %*% coeff) + error
  # Non-linear DGP
  y.cat <- factor(ifelse(y >= median(y), "1", "0")) # Categorical target
  
  data <- data.frame(X, "y.cat" = y.cat)
  return(data)
}





###############################################################################
###############################################################################
###############################################################################
# Fitting Trees and Random Forests
###############################################################################
###############################################################################
###############################################################################


# Classification Tree and Share of Correct Predictions #########################
tree.fun <- function(form, dt.train, dt.test, test.Y, plot = FALSE){ 
  # Fit classification trees and calculate their share of correct predictions
  
  # form: formula for estimation; dt.train: training data set; dt.test: test 
  # data set; test.Y: the test target vector; plot: TRUE/FALSE depending on 
  # whether a plot should be generated
  
  tr <- rpart(formula = form, data = dt.train, method = "class") # Pruned tree
  tr.pred <- predict(tr, dt.test, type = "class") # Prediction
  class.table <- table(tr.pred, test.Y) # Confusion Matrix
  error <- 1 - (sum(diag(class.table)) / sum(class.table)) # Share of wrongly 
  # classified observations 
  
  if(plot == TRUE){
    prp(tr) # Tree plot
  }
  
  list <- list("tree" = tr, "misclass.error" = error)
  return(list)
}

# Random Forests, Variable Importance and Share of Correct Predictions #########
rf.fun <- function(form, dt.train, dt.test, test.Y, m, ntr = 200, imp = TRUE, 
                   vi.plot = FALSE, own.vi.plot = FALSE){ 
  # Fit random forests and calculate their share of correct predictions
  
  # form: formula for estimation; dt.train: training data set; dt.test: test 
  # data set; test.Y: the test target vector; m: number of random features 
  # considered in each split of each tree; ntr: number of trees grown; 
  # imp: TRUE/FALSE depending on whether the variable importance shall be 
  # computed; vi.plot: TRUE/FALSE depending on whether the inbuilt plot of 
  # variable importance measure(s) is desired; own.vi.plot: same as vi.plot only
  # for a self-created (bar)plot of the variable importance 
  
  rf <- randomForest(formula = form, # Formula 
                     data = dt.train, 
                     mtry = m, # Default is square root of p
                     ntree = ntr, # Default is 200
                     importance = imp) # Variable importance measure
  rf.pred <- predict(rf, newdata = dt.test, method = "class") # Prediction
  class.table.rf <- table(rf.pred, test.Y) # Confusion Matrix
  error <- 1 - (sum(diag(class.table.rf)) / sum(class.table.rf)) # Share of 
  # wrongly classified observations 
  
  if(vi.plot == TRUE){
    varImpPlot(rf) # Inbuilt variable importance measure plot
  }
  
  if(own.vi.plot == TRUE){
    VI <- (rf$importance[,9]/max(rf$importance[,9]))*100 # Save and scale 
    # variable importance measures
    barplot(sort(VI), col = "deepskyblue", horiz = TRUE, las = 1,
            main = "Variable Importance (Random Forest)", 
            xlab = TeX(r'(\%)')) # Bar plot
  }
  
  list <- list("rf" = rf, "misclass.error" = error)
  return(list)
}

# Optimal m in Random Forests ##################################################
rf.opt.m <- function(p, equation, dt.tr, dt.te, te.y, notree = 200){
  # Compute the optimal m for the random forest algorithm
  
  # p: number of features, i.e. the maximum m; equation: formula for estimation;
  # dt.tr: training data set; dt.te: test data set; te.y: the test target vector; 
  # notree: number of trees grown
  
  res.opt.m <- rep(NA, times = p) # Container for RF loop 
  for (q in 1:length(res.opt.m)) { # Find optimal random forest 
    set.seed(seed) 
    rf.opt.m <- rf.fun(form = equation, dt.train = dt.tr, dt.test = dt.te, 
                   test.Y = te.y, imp = FALSE, ntr = notree, m = q) 
    res.opt.m[q] <- rf.opt.m$misclass.error
  } 
  min.opt.m <- match(min(res.opt.m), res.opt.m)
  return("opt.m" = min.opt.m)
}
