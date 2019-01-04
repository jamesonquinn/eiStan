functions {
  real cmonom_lpdf(real y, real p) {
    //a continuous approximation of a "monomial" (one term in a multinomial)
    return (y*log(p))-lgamma(y+1)
    }
  real cbinom_lpdf(real y, real n, real p) {
    //a continuous approximation of a binomial
    return (y*log(p))+((n-y)*log(1-p))-lgamma(y+1)-lgamma(n-y+1)
    }
}

data{
    int<lower=0> ncols;                   // number of columns (numbers that are defined for each precinct)
    int<lower=0> nprec;                   // number of precincts 
    real y_site_obs[J];               // observed site-specific mean
    real<lower=0> sigma_site_obs[J];  // observed site-specific sd
}

parameters{
    real mu_true;                 // grand mean
    real<lower=0> sigma_true;     // grand sd
    real<lower=0.1> exposcale;               // scale (inverse rate) for exponential
    vector[J] y_site_true;          // true site-specific mean
}

transformed parameters{
	real<lower=0> scale;
	real location;

	scale = sigma_true * sqrt(1 + exposcale^2);
	location = mu_true + sigma_true * exposcale;
}


model{
    // some weakly informative priors
    mu_true ~ normal(0, 5);
    exposcale ~ normal(0, 2);
    sigma_true ~ normal(0, 2);

    // likelihood
    if (exposcale == 0.0) {
       y_site_true ~ normal(mu_true, sigma_true);
    } else if (exposcale > 0.0){
       y_site_true ~ exp_mod_normal(mu_true, sigma_true, sigma_true/exposcale);
    }
    //else if (exposcale == 0.0) {
    //   y_site_true ~ normal(mu_true, sigma_true);
    //{
    //   target += exp_mod_normal_lpdf(-y_site_true | -mu_true, sigma_true, -sigma_true/exposcale);
    //}
    y_site_obs ~ normal(y_site_true, sigma_site_obs);
}
