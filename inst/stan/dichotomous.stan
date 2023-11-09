#include /include/license.stan

data {
  int<lower=1> N;                              // Number of observations
  int<lower=1> I;                              // Number of items
  int<lower=1> P;                              // Number of people
  int<lower=1, upper=I> ii[N];                 // Variable indexing the items
  int<lower=1, upper=P> pp[N];                 // Variable indexing the people
  int<lower=0, upper=1> y[N];                  // Binary outcome variable
  int K;                                       // Number of covariates
  matrix[P, K] x;                              // Matrix of person covariates
  real<lower=0> D;                             // Scaling factor, default = 1.701
  int modelType;                               // Dichotomous IRT model: 1 = Rasch / 1-PL, 2 = 2-PL, 3 = 3-PL, 4 = 4-PL
  int priorTheta;                              // Prior for theta: 1 = Normal, 2 = Student-t, 3 = Cauchy, 4 = Hierarchical
  real priorThetaNormalMean;                   // Mean of the normal prior distribution for theta
  real<lower=0> priorThetaNormalSd;            // Standard deviation of the normal prior distribution for theta
  real priorThetaStudentLocation;              // Location of the Student-t prior distribution for theta
  real<lower=0> priorThetaStudentDf;           // Degrees of freedom of the Student-t prior distribution for theta
  real priorThetaCauchyLocation;               // Location of the Cauchy prior distribution for theta
  real<lower=0> priorThetaCauchyScale;         // Degrees of freedom of the Cauchy prior distribution for theta
  real<lower=0> priorThetaHierarchicalSd;      // Standard deviation of the prior on theta_sigma
  int priorAlpha;                              // Prior for alpha: 1 = Lognormal, 2 = Normal, 3 = Student-t, 4 = Cauchy, 5 = Hierarchical
  real priorAlphaLogNormalMean;                // Mean of the lognormal prior distribution for alpha
  real<lower=0> priorAlphaLogNormalSd;         // Standard deviation of the lognormal prior distribution for alpha
  real priorAlphaNormalMean;                   // Mean of the normal prior distribution for alpha
  real<lower=0> priorAlphaNormalSd;            // Standard deviation of the normal prior distribution for alpha
  real priorAlphaStudentLocation;              // Location of the Student-t prior distribution for alpha
  real<lower=0> priorAlphaStudentDf;           // Degrees of freedom of the Student-t prior distribution for alpha
  real priorAlphaCauchyLocation;               // Location of the Cauchy prior distribution for alpha
  real<lower=0> priorAlphaCauchyScale;         // Degrees of freedom of the Cauchy prior distribution for alpha
  real<lower=0> priorAlphaHierarchicalSd;      // Standard deviation of the prior on alpha_sigma
  int priorBeta;                               // Prior for beta: 1 = Normal, 2 = Student-t, 3 = Cauchy, 4 = Hierarchical
  real priorBetaNormalMean;                    // Mean of the normal prior distribution for beta
  real<lower=0> priorBetaNormalSd;             // Standard deviation of the normal prior distribution for beta
  real priorBetaStudentLocation;               // Location of the Student-t prior distribution for beta
  real<lower=0> priorBetaStudentDf;            // Degrees of freedom of the Student-t prior distribution for beta
  real priorBetaCauchyLocation;                // Location of the Cauchy prior distribution for beta
  real<lower=0> priorBetaCauchyScale;          // Degrees of freedom of the Cauchy prior distribution for beta
  real<lower=0> priorBetaHierarchicalSd;       // Standard deviation of the prior on beta_sigma
  int priorGamma;                              // Prior for gamma: 1 = Uniform, 2 = Beta, 3 = Normal, 4 = Student-t, 5 = Cauchy, 6 = Hierarchical
  real<lower=0, upper=1> priorGammaUniformMin; // Minimum of the uniform prior distribution for gamma
  real<lower=0, upper=1> priorGammaUniformMax; // Maximum of the uniform prior distribution for gamma
  real<lower=0.5> priorGammaBetaAlpha;         // First shape parameter of the beta prior distribution for gamma
  real<lower=0.5> priorGammaBetaBeta;          // Second shape parameter of the beta prior distribution for gamma
  real priorGammaNormalMean;                   // Mean of the normal prior distribution for gamma
  real<lower=0> priorGammaNormalSd;            // Standard deviation of the normal prior distribution for gamma
  real priorGammaStudentLocation;              // Location of the Student-t prior distribution for gamma
  real<lower=0> priorGammaStudentDf;           // Degrees of freedom of the Student-t prior distribution for gamma
  real priorGammaCauchyLocation;               // Location of the Cauchy prior distribution for gamma
  real<lower=0> priorGammaCauchyScale;         // Degrees of freedom of the Cauchy prior distribution for gamma
  real<lower=0> priorGammaHierarchicalSd;      // Standard deviation of the prior on gamma_sigma
  int priorDelta;                              // Prior for delta: 1 = Uniform, 2 = Beta, 3 = Normal, 4 = Student-t, 5 = Cauchy, 6 = Hierarchical
  real<lower=0, upper=1> priorDeltaUniformMin; // Minimum of the uniform prior distribution for delta
  real<lower=0, upper=1> priorDeltaUniformMax; // Maximum of the uniform prior distribution for gamma
  real<lower=0.5> priorDeltaBetaAlpha;         // First shape parameter of the beta prior distribution for delta
  real<lower=0.5> priorDeltaBetaBeta;          // Second shape parameter of the beta prior distribution for delta
  real priorDeltaNormalMean;                   // Mean of the normal prior distribution for delta
  real<lower=0> priorDeltaNormalSd;            // Standard deviation of the normal prior distribution for delta
  real priorDeltaStudentLocation;              // Location of the Student-t prior distribution for delta
  real<lower=0> priorDeltaStudentDf;           // Degrees of freedom of the Student-t prior distribution for delta
  real priorDeltaCauchyLocation;               // Location of the Cauchy prior distribution for delta
  real<lower=0> priorDeltaCauchyScale;         // Degrees of freedom of the Cauchy prior distribution for delta
  real<lower=0> priorDeltaHierarchicalSd;      // Standard deviation of the prior on delta_sigma
  int priorZeta;                               // Prior for latent regression coefficients: 1 = Normal, 2 = Student-t, 3 = Cauchy, 4 = Hierarchical
  real priorZetaNormalMean;                    // Mean of the normal prior distribution for the latent regression coefficients
  real<lower=0> priorZetaNormalSd;             // Standard deviation of the normal prior distribution for the latent regression coefficients
  real priorZetaStudentLocation;               // Location of the Student-t prior distribution for the latent regression coefficients
  real<lower=0> priorZetaStudentDf;            // Degrees of freedom of the Student-t prior distribution for the latent regression coefficients
  real priorZetaCauchyLocation;                // Location of the Cauchy prior distribution for the latent regression coefficients
  real<lower=0> priorZetaCauchyScale;          // Degrees of freedom of the Cauchy prior distribution for the latent regression coefficients
  real<lower=0> priorZetaHierarchicalSd;       // Standard deviation of the prior on zeta_sigma
}
parameters {
  vector[P] theta_tmp;          // Temporary vector of P person ability parameters
  real<lower=0> theta_sigma;    // Std. dev of person ability parameters (only used with hierarchical prior)
  vector<lower=0>[I] alpha;     // Vector of I item discrimination parameters
  real<lower=0> alpha_sigma;    // Std. dev of item discrimination parameters (only used with hierarchical prior)
  vector[I] beta;               // Vector of I item difficulty parameters
  real<lower=0> beta_sigma;     // Std. dev of item difficulty parameters (only used with hierarchical prior)
  vector<lower=[(priorGamma==1) ? priorGammaUniformMin : 0][1], upper=[(priorGamma==1) ? priorGammaUniformMax : 1][1]>[I] gamma; // Vector of I item guessing paramaters
  real<lower=0> gamma_sigma;    // Std. dev of item guessing parameters (only used with hierarchical prior)
  vector<lower=[(priorDelta==1) ? priorDeltaUniformMin : 0][1], upper=[(priorDelta==1) ? priorDeltaUniformMax : 1][1]>[I] delta; // Vector of I item slip paramaters
  real<lower=0> delta_sigma;    // Std. dev of item slip parameters (only used with hierarchical prior)
  vector[K] zeta;               // Vector of K latent regression coefficients
  real<lower=0> zeta_sigma;     // Std. dev of latent regression coefficients (only used with hierarchical prior)
  real epsilon;                 // Latent regression residuals
  real<lower=0> epsilon_sigma;  // Latent regression SD residuals
}
transformed parameters {
  vector[P] theta; // Vector of P person ability parameters
  if (K == 0) {
    theta = theta_tmp;
  } else {
    for (i in 1:P) {
      theta[i] = sum(zeta .* to_vector(x[i]) + epsilon); // Latent regression on ability
    }
  }
  vector<lower=0, upper=1>[N] p; // Vector of N probabilities
  if (modelType == 1) {
    p = inv_logit(theta[pp] - beta[ii]); // Rasch
  } else if (modelType == 2) {
    p = inv_logit(D * alpha[ii] .* (theta[pp] - beta[ii])); // 2-PL
  } else if (modelType == 3) {
    p = gamma[ii] + (1 - gamma[ii]) .* inv_logit(D * alpha[ii] .* (theta[pp] - beta[ii])); // 3-PL
  } else if (modelType == 4) {
    p = gamma[ii] + (delta[ii] - gamma[ii]) .* inv_logit(D * alpha[ii] .* (theta[pp] - beta[ii])); // 4-PL
  }
}
model {
  // Prior for person ability parameters
  theta_sigma ~ cauchy(0, priorThetaHierarchicalSd);
  if (priorTheta == 1) { // Independent normal
    theta_tmp ~ normal(priorThetaNormalMean, priorThetaNormalSd);
  } else if (priorTheta == 2) { // Independent Student-t
    theta_tmp ~ student_t(priorThetaStudentDf, priorThetaStudentLocation, 1);
  } else if (priorTheta == 3) { // Independent Cauchy
    theta_tmp ~ cauchy(priorThetaCauchyLocation, priorThetaCauchyScale);
  } else if (priorTheta == 4) { // Hierarchical
    theta_tmp ~ normal(0, theta_sigma);
  }
  // Prior for item discrimination parameters
  alpha_sigma ~ cauchy(0, priorAlphaHierarchicalSd);
  if (priorAlpha == 1) { // Independent lognormal
    alpha ~ lognormal(priorAlphaLogNormalMean, priorAlphaLogNormalSd);
  } else if (priorAlpha == 2) { // Independent normal
    alpha ~ normal(priorAlphaNormalMean, priorAlphaNormalSd);
  } else if (priorAlpha == 3) { // Independent Student-t
    alpha ~ student_t(priorAlphaStudentDf, priorAlphaStudentLocation, 1);
  } else if (priorAlpha == 4) { // Independent Cauchy
    alpha ~ cauchy(priorAlphaCauchyLocation, priorAlphaCauchyScale);
  } else if (priorAlpha == 5) { // Hierarchical
    alpha ~ lognormal(0, alpha_sigma);
  }
  // Prior for item difficulty parameters
  beta_sigma ~ cauchy(0, priorBetaHierarchicalSd);
  if (priorBeta == 1) { // Independent normal
    beta ~ normal(priorBetaNormalMean, priorBetaNormalSd);
  } else if (priorBeta == 2) { // Independent Student-t
    beta ~ student_t(priorBetaStudentDf, priorBetaStudentLocation, 1);
  } else if (priorBeta == 3) { // Independent Cauchy
    beta ~ cauchy(priorBetaCauchyLocation, priorBetaCauchyScale);
  } else if (priorBeta == 4) { // Hierarchical
    beta ~ normal(0, beta_sigma);
  }
  // Prior for item guessing parameters
  gamma_sigma ~ cauchy(0, priorGammaHierarchicalSd);
  if (priorGamma == 1) { // Independent uniform
    gamma ~ uniform(priorGammaUniformMin, priorGammaUniformMax);
  } else if (priorGamma == 2) { // Independent beta
    gamma ~ beta(priorGammaBetaAlpha, priorGammaBetaBeta);
  } else if (priorGamma == 3) { // Independent normal
    gamma ~ normal(priorGammaNormalMean, priorGammaNormalSd);
  } else if (priorGamma == 4) { // Independent Student-t
    gamma ~ student_t(priorGammaStudentDf, priorGammaStudentLocation, 1);
  } else if (priorGamma == 5) { // Independent Cauchy
    gamma ~ cauchy(priorGammaCauchyLocation, priorGammaCauchyScale);
  } else if (priorGamma == 6) { // Hierarchical
    gamma ~ normal(0, gamma_sigma);
  }
  // Prior for item slip parameters
  delta_sigma ~ cauchy(0, priorDeltaHierarchicalSd);
  if (priorDelta == 1) { // Independent uniform
    delta ~ uniform(priorDeltaUniformMin, priorDeltaUniformMax);
  } else if (priorDelta == 2) { // Independent beta
    delta ~ beta(priorDeltaBetaAlpha, priorDeltaBetaBeta);
  } else if (priorDelta == 3) { // Independent normal
    delta ~ normal(priorDeltaNormalMean, priorDeltaNormalSd);
  } else if (priorDelta == 4) { // Independent Student-t
    delta ~ student_t(priorDeltaStudentDf, priorDeltaStudentLocation, 1);
  } else if (priorDelta == 5) { // Independent Cauchy
    delta ~ cauchy(priorDeltaCauchyLocation, priorDeltaCauchyScale);
  } else if (priorDelta == 6) { // Hierarchical
    delta ~ normal(1, delta_sigma);
  }
  // Prior for latent regression coefficients
  zeta_sigma ~ cauchy(0, priorZetaHierarchicalSd);
  if (priorZeta == 1) { // Independent normal
    zeta ~ normal(priorZetaNormalMean, priorZetaNormalSd);
  } else if (priorZeta == 2) { // Independent Student-t
    zeta ~ student_t(priorZetaStudentDf, priorZetaStudentLocation, 1);
  } else if (priorZeta == 3) { // Independent Cauchy
    zeta ~ cauchy(priorZetaCauchyLocation, priorZetaCauchyScale);
  } else if (priorZeta == 4) { // Hierarchical
    zeta ~ normal(0, zeta_sigma);
  }
  // Prior for latent regression residuals
  epsilon_sigma ~ cauchy(0, 5);
  epsilon ~ normal(0, epsilon_sigma);
  // Likelihood
  y ~ bernoulli(p);
}
generated quantities {
  vector[N] y_rep = to_vector(bernoulli_rng(p)); // Model implied data
  real ppp[I]; // Posterior predictive p-value
  real sumy;
  real sumyrep;
  for (i in 1:I) {
    sumy = 0;
    sumyrep = 0;
    for (n in 1:N) {
      if (ii[n] == i) {
        sumy += y[n];
        sumyrep += y_rep[n];
      }
    }
    ppp[i] = step(sumyrep - sumy);
  }
}
