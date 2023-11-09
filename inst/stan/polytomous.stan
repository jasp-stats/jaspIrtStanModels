data {
  int<lower=1> N;                          // Number of observations
  int<lower=1> I;                          // Number of items
  int<lower=1> P;                          // Number of people
  int<lower=1, upper=I> ii[N];             // Variable indexing the items
  int<lower=1, upper=P> pp[N];             // Variable indexing the people
  int<lower=2> C;                          // Number of categories
  int<lower=1, upper=C> y[N];              // Categorical outcome variable
  int K;                                   // Number of covariates
  matrix[P, K] x;                          // Matrix of person covariates
  real<lower=0> D;                         // Scaling factor, default = 1.701
  int modelType;                           // Polytomous IRT model: 1 = Partial credit model, 2 = Generalized partial credit model, 3 = Rating scale model, 4 = Generalized rating scale model, 5 = Graded response model, 6 = Nominal response model
  int priorTheta;                          // Prior for theta: 1 = Normal, 2 = Student-t, 3 = Cauchy, 4 = Hierarchical
  real priorThetaNormalMean;               // Mean of the normal prior distribution for theta
  real<lower=0> priorThetaNormalSd;        // Standard deviation of the normal prior distribution for theta
  real priorThetaStudentLocation;          // Location of the Student-t prior distribution for theta
  real<lower=0> priorThetaStudentDf;       // Degrees of freedom of the Student-t prior distribution for theta
  real priorThetaCauchyLocation;           // Location of the Cauchy prior distribution for theta
  real<lower=0> priorThetaCauchyScale;     // Degrees of freedom of the Cauchy prior distribution for theta
  real<lower=0> priorThetaHierarchicalSd;  // Standard deviation of the prior on theta_sigma
  int priorAlpha;                          // Prior for alpha: 1 = Lognormal, 2 = Normal, 3 = Student-t, 4 = Cauchy, 5 = Hierarchical
  real priorAlphaLogNormalMean;            // Mean of the lognormal prior distribution for alpha
  real<lower=0> priorAlphaLogNormalSd;     // Standard deviation of the lognormal prior distribution for alpha
  real priorAlphaNormalMean;               // Mean of the normal prior distribution for alpha
  real<lower=0> priorAlphaNormalSd;        // Standard deviation of the normal prior distribution for alpha
  real priorAlphaStudentLocation;          // Location of the Student-t prior distribution for alpha
  real<lower=0> priorAlphaStudentDf;       // Degrees of freedom of the Student-t prior distribution for alpha
  real priorAlphaCauchyLocation;           // Location of the Cauchy prior distribution for alpha
  real<lower=0> priorAlphaCauchyScale;     // Degrees of freedom of the Cauchy prior distribution for alpha
  real<lower=0> priorAlphaHierarchicalSd;  // Standard deviation of the prior on alpha_sigma
  int priorBeta;                           // Prior for beta: 1 = Normal, 2 = Student-t, 3 = Cauchy, 4 = Hierarchical
  real priorBetaNormalMean;                // Mean of the normal prior distribution for beta
  real<lower=0> priorBetaNormalSd;         // Standard deviation of the normal prior distribution for beta
  real priorBetaStudentLocation;           // Location of the Student-t prior distribution for beta
  real<lower=0> priorBetaStudentDf;        // Degrees of freedom of the Student-t prior distribution for beta
  real priorBetaCauchyLocation;            // Location of the Cauchy prior distribution for beta
  real<lower=0> priorBetaCauchyScale;      // Degrees of freedom of the Cauchy prior distribution for beta
  real<lower=0> priorBetaHierarchicalSd;   // Standard deviation of the prior on beta_sigma
  int priorThreshold;                      // Prior for threshold parameters: 1 = Normal, 2 = Student-t, 3 = Cauchy, 4 = Uniform
  real priorThresholdNormalMean;           // Mean of the normal prior distribution for threshold parameters
  real<lower=0> priorThresholdNormalSd;    // Standard deviation of the normal prior distribution for threshold parameters
  real priorThresholdStudentLocation;      // Location of the Student-t prior distribution for threshold parameters
  real<lower=0> priorThresholdStudentDf;   // Degrees of freedom of the Student-t prior distribution for threshold parameters
  real priorThresholdCauchyLocation;       // Location of the Cauchy prior distribution for threshold parameters
  real<lower=0> priorThresholdCauchyScale; // Degrees of freedom of the Cauchy prior distribution for threshold parameters
  real priorThresholdUniformMin;           // Minimum of the uniform prior distribution for threshold parameters
  real priorThresholdUniformMax;           // Maximum of the uniform prior distribution for threshold parameters
  int priorZeta;                           // Prior for latent regression coefficients: 1 = Normal, 2 = Student-t, 3 = Cauchy, 4 = Hierarchical
  real priorZetaNormalMean;                // Mean of the normal prior distribution for the latent regression coefficients
  real<lower=0> priorZetaNormalSd;         // Standard deviation of the normal prior distribution for the latent regression coefficients
  real priorZetaStudentLocation;           // Location of the Student-t prior distribution for the latent regression coefficients
  real<lower=0> priorZetaStudentDf;        // Degrees of freedom of the Student-t prior distribution for the latent regression coefficients
  real priorZetaCauchyLocation;            // Location of the Cauchy prior distribution for the latent regression coefficients
  real<lower=0> priorZetaCauchyScale;      // Degrees of freedom of the Cauchy prior distribution for the latent regression coefficients
  real<lower=0> priorZetaHierarchicalSd;   // Standard deviation of the prior on zeta_sigma
}
parameters {
  vector[P] theta_tmp;           // Temporary vector of P person ability parameters
  real<lower=0> theta_sigma;     // Std. dev of person ability parameters (only used with hierarchical prior)
  vector<lower=0>[I] alpha;      // Vector of I item discrimination parameters
  real<lower=0> alpha_sigma;     // Std. dev of item discrimination parameters (only used with hierarchical prior)
  vector[C] alpha_cat_tmp[I];    // Unconstrained discrimination parameters for each category (only used in nominal response model)
  real<lower=0> alpha_cat_sigma; // Std. dev of item discrimination parameters (only used with hierarchical prior)
  vector[C] beta_cat_tmp[I];     // Unconstrained difficulty parameters for each category (only used in nominal response model)
  real<lower=0> beta_cat_sigma;  // Std. dev of item discrimination parameters (only used with hierarchical prior)
  ordered[C - 1] beta[I];        // Ordered vector of I times C - 1 category difficulty parameters
  real<lower=0> beta_sigma;      // Std. dev of category difficulty parameters (only used with hierarchical prior)
  vector[K] zeta;                // Vector of K latent regression coefficients
  real<lower=0> zeta_sigma;      // Std. dev of latent regression coefficients (only used with hierarchical prior)
  real epsilon;                  // Latent regression residuals
  real<lower=0> epsilon_sigma;   // Latent regression SD residuals
  vector[I] beta_rsm;            // Vector of I category difficulty parameters (only used in rsm and grsm models)
  real<lower=0> beta_rsm_sigma;  // Std. dev of item difficulty parameters (only used with hierarchical prior)
  vector<lower=[(priorThreshold==4) ? priorThresholdUniformMin : negative_infinity()][1], upper=[(priorThreshold==4) ? priorThresholdUniformMax : positive_infinity()][1]>[C - 1] threshold; // Vector of C - 1 category thresholds
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
  vector[C] alpha_cat[I]; // Constrained discrimination parameters for each category (only used in nominal response model)
  vector[C] beta_cat[I]; // Constrained difficulty parameters for each category (only used in nominal response model)
  for (i in 1:I) {
    for (c in 1:C) {
      alpha_cat[i, c] = alpha_cat_tmp[i, c] - mean(alpha_cat_tmp[i]); // Constrain slope sum for each item to 0
      beta_cat[i, c] = beta_cat_tmp[i, c] - mean(beta_cat_tmp[i]); // Constrain intercept sum for each item to 0
    }
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
  // Prior for category discrimination parameters in generalized partial credit model, generalized rating scale model, graded response model and nominal response model
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
  // Prior for category discrimination parameters in nominal response model
  alpha_cat_sigma ~ cauchy(0, priorAlphaHierarchicalSd);
  for (i in 1:I) {
    for (c in 1:C) {
      if (priorAlpha == 1) { // Independent lognormal
        alpha_cat_tmp[i, c] ~ normal(priorAlphaLogNormalMean, priorAlphaLogNormalSd);
      } else if (priorAlpha == 2) { // Independent normal
        alpha_cat_tmp[i, c] ~ normal(priorAlphaNormalMean, priorAlphaNormalSd);
      } else if (priorAlpha == 3) { // Independent Student-t
        alpha_cat_tmp[i, c] ~ student_t(priorAlphaStudentDf, priorAlphaStudentLocation, 1);
      } else if (priorAlpha == 4) { // Independent Cauchy
        alpha_cat_tmp[i, c] ~ cauchy(priorAlphaCauchyLocation, priorAlphaCauchyScale);
      } else if (priorAlpha == 5) { // Hierarchical
        alpha_cat_tmp[i, c] ~ normal(0, alpha_cat_sigma);
      }
    }
  }
  // Prior for category difficulty parameters in (generalized) partial credit model and graded response model
  beta_sigma ~ cauchy(0, priorBetaHierarchicalSd);
  for (i in 1:I) {
    for (c in 1:(C - 1)) {
      if (priorBeta == 1) { // Independent normal
        beta[i, c] ~ normal(priorBetaNormalMean, priorBetaNormalSd);
      } else if (priorBeta == 2) { // Independent Student-t
        beta[i, c] ~ student_t(priorBetaStudentDf, priorBetaStudentLocation, 1);
      } else if (priorBeta == 3) { // Independent Cauchy
        beta[i, c] ~ cauchy(priorBetaCauchyLocation, priorBetaCauchyScale);
      } else if (priorBeta == 4) { // Hierarchical
        beta[i, c] ~ normal(0, beta_sigma);
      }
    }
  }
  // Prior for category difficulty parameters in nominal response model
  beta_cat_sigma ~ cauchy(0, priorBetaHierarchicalSd);
  for (i in 1:I) {
    for (c in 1:C) {
      if (priorBeta == 1) { // Independent normal
        beta_cat_tmp[i, c] ~ normal(priorBetaNormalMean, priorBetaNormalSd);
      } else if (priorBeta == 2) { // Independent Student-t
        beta_cat_tmp[i, c] ~ student_t(priorBetaStudentDf, priorBetaStudentLocation, 1);
      } else if (priorBeta == 3) { // Independent Cauchy
        beta_cat_tmp[i, c] ~ cauchy(priorBetaCauchyLocation, priorBetaCauchyScale);
      } else if (priorBeta == 4) { // Hierarchical
        beta_cat_tmp[i, c] ~ normal(0, beta_cat_sigma);
      }
    }
  }
  // Prior for category difficulty parameters in (generalized) rating scale model
  beta_rsm_sigma ~ cauchy(0, priorBetaHierarchicalSd);
  if (priorBeta == 1) { // Independent normal
    beta_rsm ~ normal(priorBetaNormalMean, priorBetaNormalSd);
  } else if (priorBeta == 2) { // Independent Student-t
    beta_rsm ~ student_t(priorBetaStudentDf, priorBetaStudentLocation, 1);
  } else if (priorBeta == 3) { // Independent Cauchy
    beta_rsm ~ cauchy(priorBetaCauchyLocation, priorBetaCauchyScale);
  } else if (priorBeta == 4) { // Hierarchical
    beta_rsm ~ normal(0, beta_rsm_sigma);
  }
  // Prior for threshold parameters (only used in (generalized) rating scale model)
  if (priorThreshold == 1) {
    threshold ~ normal(priorThresholdNormalMean, priorThresholdNormalSd);
  } else if (priorThreshold == 2) {
    threshold ~ student_t(priorThresholdStudentDf, priorThresholdStudentLocation, 1);
  } else if (priorThreshold == 3) {
    threshold ~ cauchy(priorThresholdCauchyLocation, priorThresholdCauchyScale);
  } else if (priorThreshold == 4) {
    threshold ~ uniform(priorThresholdUniformMin, priorThresholdUniformMax);
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
  if (modelType == 1) { // Partial credit model
    for (n in 1:N) {
      vector[C] unsummed = append_row(rep_vector(0.0, 1), theta[pp[n]] - beta[ii[n]]);
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y[n] ~ categorical(probs);
    }
  } else if (modelType == 2) { // Generalized partial credit model
    for (n in 1:N) {
      vector[C] unsummed = append_row(rep_vector(0.0, 1), (D * alpha[ii[n]]) * (theta[pp[n]] - beta[ii[n]]));
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y[n] ~ categorical(probs);
    }
  } else if (modelType == 3) {
    for (n in 1:N) { // Rating scale model
      vector[C] unsummed = append_row(rep_vector(0, 1), theta[pp[n]] - (beta_rsm[ii[n]] + threshold));
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y[n] ~ categorical(probs);
    }
  } else if (modelType == 4) {
    for (n in 1:N) { // Generalized rating scale model
      vector[C] unsummed = append_row(rep_vector(0, 1), (D * alpha[ii[n]]) * (theta[pp[n]] - (beta_rsm[ii[n]] + threshold)));
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y[n] ~ categorical(probs);
    }
  } else if (modelType == 5) { // Graded response model
    y ~ ordered_logistic(theta[pp] .* (D * alpha[ii]), beta[ii]);
  } else if (modelType == 6) {
    for (n in 1:N) { // Nominal response model
      y[n] ~ categorical_logit(beta_cat[ii[n]] + theta[pp[n]] * (D * alpha_cat[ii[n]]));
    }
  }
}
generated quantities {
  vector[N] y_rep; // Model implied data
  for (n in 1:N) {
    if (modelType == 1) {
      vector[C] unsummed = append_row(rep_vector(0.0, 1), theta[pp[n]] - beta[ii[n]]);
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y_rep[n] = categorical_rng(probs);
    } else if (modelType == 2) {
      vector[C] unsummed = append_row(rep_vector(0.0, 1), (D * alpha[ii[n]]) * (theta[pp[n]] - beta[ii[n]]));
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y_rep[n] = categorical_rng(probs);
    } else if (modelType == 3) {
      vector[C] unsummed = append_row(rep_vector(0, 1), theta[pp[n]] - (beta_rsm[ii[n]] + threshold));
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y_rep[n] = categorical_rng(probs);
    } else if (modelType == 4) {
      vector[C] unsummed = append_row(rep_vector(0, 1), (D * alpha[ii[n]]) * (theta[pp[n]] - (beta_rsm[ii[n]] + threshold)));
      vector[C] probs = softmax(cumulative_sum(unsummed));
      y_rep[n] = categorical_rng(probs);
    } else if (modelType == 5) {
      y_rep[n] = ordered_logistic_rng(theta[pp[n]] * (D * alpha[ii[n]]), beta[ii[n]]);
    } else if (modelType == 6) {
      y_rep[n] = categorical_logit_rng(beta_cat[ii[n]] + theta[pp[n]] * (D * alpha_cat[ii[n]]));
    }
  }
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
