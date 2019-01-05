functions {

  vector[] my_simplex(vector y, vector v) {
  // v has dimension K, y has dimension K-1
  // This function maps y into a simplex in such a way that y=0 goes to v;
  // it's just like STAN's built-in way of generating a simplex, except that
  // 0 goes to a different place.
  	int K=rows(v);
  	vector[K] s[2];
  //s[1] is the actual simplex;
  //s[2] is log of the derivative of the function y[i] --> s[1][i]

	int VERBOSE=0;
	real v_res = 1-v[1];
	real s_res = 1;

  // we compute all e^y's once so we don't have to redo it each time

	for (i in 1:K-1) {
		s[1][i]=s_res*v[i]/(v[i]+v_res*exp(-y[i]));
		s[2][i]=log(s[1][i]*(1-s[1][i])/s_res);

		v_res = v_res-v[i+1];
		s_res = s_res - s[1][i];
		}
	s[1][K]=s_res;
	s[2][K]=0;		// The last entry does not contribute to the Jacobian

	if (VERBOSE==1){
		print("Function my_simplex:")
		print("y = ", y);
		print("v= ", v);
		print("s[1]=",s[1]);
		print("s[2]=",s[2]);
		}

	return s;
	}



  real my_multinomial_lpdf(vector[] Q, vector[] Beta) {
  	// Beta[r,c] = probability of person of race r voting for candidate c.
  	// Q[r,c] = "number" of people of race r voting for candidate c
  	// (Q[r,c] can be a real number; when r=R, it can be negative, which is a problem.)

  	// This function gives the log of the unnormalized probability of Q given Beta.
  	// The probability is a product of "multinomials", substituting gamma for factorial.
  	// We omit the numerators of the multinomial coefficients, since they don't depend on Q.

  	int R = size(Q);
  	int C = rows(Q[1]);
  	real ans = 0;


    // Sum log "multinomial" probabilities for all the rows of matrix Q
    //  If has any negative elements, log probability should be -infinity, but we
    // make it a negative number with gradient toward 0, so that the sampler knows which way
    // to go back.
    for (r in 1:R) {
    	for (c in 1:C) {
    		if (Q[r,c]<0) ans+=Q[r,c]-1000000;
    		else ans+=Q[r,c]*log(Beta[r,c]) - lgamma(Q[r,c]+1);
    		}
	}

	return ans;
  }
}

// ************************************************************************************

data{
	int<lower = 1> R;		// number of rows, i.e. racial groups
	int<lower = 2> C;		// number of columRs, i.e. candidates
	int<lower = 2> M;		// number of precincts
	int<lower = 0> N[M];	// number of people in each precinct
	simplex[C] v[M];		// proportion of people for each candidate in each precinct
	simplex[R] x[M];		// proportion of people of each race in each precinct
}


// ************************************************************************************

parameters {
	//real <lower = 0> lambda;
	// Exponential hyperparameter: small lambda means lower
	// variation of betas between precincts within a race,
	// for all races

    real<lower=0>      cStrengthShape; //Shape parameter of a gamma for cStrengths
    vector<lower=0>[C] cStrengths; //unnormalized

    real<lower=0>      racialVariance; //variance of race-specific differences
    vector<lower=0>[C] racialModifiers[R];    // log-odds modifiers to cStrengths for each race

    real<lower=0>      precinctVariance; //variance of race-specific differences
    vector<lower=0>[C] precinctModifiers[M];    // log-odds modifiers to cStrengths for each precinct

    //could include precinctRacialModifiers for fully-saturated model, and
    //precinctRacialModifiers would be a tiny bit interesting, but meh...
    //mostly that's just a lot more calculation for stuff that will show up in q anyway.

    vector [C-1] y[M, R-1];	// Will be transformed into q, the number of voters
    						// for each race +candidate.
}

// ************************************************************************************

transformed parameters {

vector[C] rawBeta[M, R];			// unnormalized betas
simplex[C] beta[M, R];			// Precinct probabilities for each race + candidate

 real cStrengthRate;
	vector [C] q[M, R];		//number of voters for each race and candidate
	real J[M];				//absolute value of the Jacobian for transforming y to q
	vector[C] s[2];			//dummy, ignore

 //calculate cStrengthRate
 cStrengthRate = mean(cStrengths) / cStrengthShape;

 //calculate beta
 for (m in 1:M) {
  for (r in 1:R) {
   rawBeta[m,r] = cStrengths .* exp(racialModifiers[r]) .* exp(precinctModifiers[m]);
   beta[m,r] = rawBeta[m,r] / sum(rawBeta[m,r]);
  }
 }

 //calculate q and J
	for (m in 1:M) {
		J[m]=0;
		q[m,R]=N[m]*v[m];
		for (r in 1:R-1) {
			// print("Running my_simplex for m=",m,"r=",r);
			s=my_simplex(y[m,r],v[m]);  // s[1] is simplex, s[2] has partial derivatives
			q[m,r]=s[1]*N[m]*x[m,r];
			q[m,R] = q[m,R] - q[m,r];
			J[m] += sum(s[2])+C*log(N[m]*x[m,r]);
		}
		//print("Last row of q for m=",m,": ",q[m,R]);
		//if (min(to_array_1d(Q[m,R]))<0)
	}
}

// ************************************************************************************

model{

  cStrengthRate ~ gamma(2.,1.); //otherwise, unidentified
  cStrengths ~ gamma(cStrengthShape,cStrengthRate);

  for (r in 1:R) {
   racialModifiers[r] ~ normal(0.,racialVariance);
  }
  for (m in 1:M) {
      precinctModifiers[m] ~ normal(0.,precinctVariance);

   		 q[m] ~ my_multinomial_lpdf(beta[m]);
   		 // print("target before adding Jacobian = ",target())
   		 target+=J[m];
   		 print("q[1] = ",q[1])
   		 print("q[2] = ",q[2])
	 }

}
