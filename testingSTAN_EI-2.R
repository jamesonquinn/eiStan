setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
dirname(rstudioapi::getActiveDocumentContext()$path)

library(rstan)
library(MCMCpack)
library(data.table)

# Using the model in rxc_beta_binomial.stan

###################################################
# Parameters to set:
stan_testing = 1
ER_testing = 0
manually_create_data = 1

set.seed(2018)
R = 2 # number of races
C = 3 # number of candidates
M = 2  # number of precincts

# For simulating data
Nmin = 1000 # min number of people per precinct
Nmax = 1200 # max number of people per precinct
x_prior = c(1,2) # Dirichlet prior for values of x (length = R)
lambda = 0.01 # rate of exponential from which we draw the alphas
####################################################

# If we're not simulating data, but entering it by hand:
if (manually_create_data == 1){
  # N should be a list of M numbers (precinct populations)
  N = c(1000,1500)

  # x should be an MxR matrix with rows summing to 1 (proportion of each race in each precinct)
  # Enter each row of x separately
  x=array(dim=c(M,R))
  x[1,1:R] = c(0.2,0.8)
  x[2,1:R] = c(0.8,0.2)
  
  # p should be an MxRxC matrix (for each precinct, the proportion of each race voting for each candidate) 
  # In the C direction, each row should sum to 1. Enter row by row.
  p = array(dim = c(M,R,C))
  # Precinct 1
  p[1,1,1:C] = c(.7,.3, 0) # Race 1
  p[1,2,1:C] = c(0,.7, .3) # Race 2

  # Precinct 2
  p[2,1,1:C] = c(.8,.2,0) # Race 1
  p[2,2,1:C] = c(0,.8,.2) # Race 2

  # The rest of the quantities are computed
  q = array(dim = c(M,R,C))
  v = array(dim = c(M,C))
  for (m in 1:M) {
    q[m,1:R,1:C] = N[m]*diag(x[m,1:R])%*%p[m,1:R,1:C]
    v[m,1:C] = rep(1,R)%*%q[m,1:R,1:C]/N[m]
  }  
}

#################################################################

# If we're simulating test data from the model:
if (manually_create_data==0){

# Data not sampled from the model (N and x)
# ----------------------------------------------------------
# N = number of people per precinct
N <- sample(Nmin:Nmax, M, replace=T)
# x (percent of each race) for each precinct
x <- rdirichlet(M, x_prior)

# Generate parameters and data from model:
# ----------------------------------------
alphas = array(dim=c(R,C))
betas = array(dim = c(M,R,C))
q = array(dim = c(M,R,C))
p = array(dim = c(M,R,C))
v = array(dim = c(M,C))

# Parameters for each race:
for (r in 1:R) {
  # Dirichlet parameters for each race and candidate
  alphas[r,1:C] <- rexp(n=C,rate=lambda) 
  
  # Precinct probabilities for each race and candidate
  for (j in 1:M) {
    betas[j,r,1:C] <- rdirichlet(1,alphas[r,1:C])
      
  # Actual precinct numbers (q) and percentages (p) for each race and candidate
    if (x[j,r]==0) {
      q[j,r,1:C] = rep(0,C)
      p[j,r,1:C] = rep(0,C)
    }
    else {
      q[j,r,1:C] <- rmultinom(1, round(x[j,r]*N[j]), betas[j,r,1:C])
      p[j,r,1:C] <- q[j,r,1:C]/round(x[j,r]*N[j])
    }
  }
}

# Voting data for each precinct
for (j in 1:M) {
  for (c in 1:C) {
    v[j,c] = sum(x[j,1:R] * p[j,1:R,c])
  }
}
}

###############################################
# Inference using Stan (model in rxc_beta_binomial.stan)

data <- list(R=R, C=C, M=M, N=N, v=v, x=x)

#my_model <- stan_model(file = "rxc_beta_binomial.stan")
my_model <- stan_model(file = "nodirichlet.stan")

show(my_model)

zero_vector = rep(0,(C-1)*M*(R-1))
zero_array = array(zero_vector, dim=c(M, R-1,C-1))
init_list = list(y=zero_array)

# vb_fit <- stan(my_model, data = data, init = 0, seed = 2018, iter = 1000, chains = 5)
vb_fit <- vb(my_model, data = data, output_samples = 2000, init = 0, seed = 2018)
params = extract(vb_fit)





################################################
# Testing ER (for R=3 and C=2)

if (ER_testing==1){
  # The overall probabilities for each race
  print(alphas[1:3,1]/(alphas[1:3,1]+alphas[1:3,2]))

  # ER results
  ER <- lm(v[1:M,1]~ x[1:M,2] + x[1:M,3])
  print(ER$coefficients[1:3])
}


