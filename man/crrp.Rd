\name{crrp}
\alias{crrp}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Penalized variable selection at the individual level in competing risks regression
}
\description{
Extends R package \pkg{ncvreg} to the proportional subdistribution hazards model. Penalties include LASSO, SCAD, and MCP. User-specified weights can be assigned to the penalty for each coefficient.
}
\usage{
crrp(time, fstatus, X, failcode = 1, cencode = 0, 
penalty = c("MCP", "SCAD", "LASSO"), gamma = switch(penalty, SCAD = 3.7, 2.7), 
alpha = 1, lambda.min = 0.001, nlambda = 50, lambda, eps = 0.001, 
max.iter = 1000, penalty.factor = rep(1, ncol(X)), weighted = FALSE)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{time}{
    vector of failure/censoring times
  }  
  \item{fstatus}{
    vector with a unique code for each failure type and a separate code for
    censored observations 
  } 
  \item{X}{
    design matrix; \code{crrp} standardizes \code{X} by default
  } 
  \item{failcode}{
    code of fstatus that denotes the failure type of interest
  }  
  \item{cencode}{
    code of fstatus that denotes censored observations
  }
  \item{penalty}{
    penalty to be applied to the model. Either "LASSO", "SCAD", or "MCP"
  }  
  \item{gamma}{tuning parameter of the MCP/SCAD penalty. Default is 2.7 for MCP and 3.7 for SCAD}  
  \item{alpha}{tuning parameter indicating contributions from the MCP/SCAD penalty and the L2
               penalty.  \code{alpha=1} is equivalent to MCP/SCAD penalty,
               whereas \code{alpha=0} would be equivalent to ridge regression. Default is 1
               }
  \item{lambda.min}{the smallest value for \code{lambda}. Default is .001}  
  \item{nlambda}{number of \code{lambda} values.  Default is 50}  
  \item{lambda}{a user-specified sequence of \code{lambda} values.  If not specified,
                a sequence of values of length \code{nlambda} is provided} 
  \item{eps}{
  iteration stops when the relative change in any coefficient is less than \code{eps}.  Default is 0.001
  }  
  \item{max.iter}{
  maximum number of iterations.  Default is 1000
  } 
  \item{penalty.factor}{
  a vector of weights applied to the penalty for each coefficient. 
  The length of the vector must be equal to the number of columns of \code{X}
  }  
  \item{weighted}{if \code{TRUE}, weights must be provided by users. Default is \code{FALSE}}
}
\details{
  The \code{crrp} function penalizes the partial likelihood of the proportional subdistribution hazards model from Fine and Gray(1999) with penalty LASSO, SCAD, and MCP. The coordinate algorithm is used for implementation. The criteria \code{BIC} and \code{GCV} are used to select the optimal tuning parameter. 
}
\value{
  Return a list of class \code{crrp} with components
  \item{$beta}{fitted coefficients matrix with \code{nvars} row and \code{nlambda} columns}
  \item{$iter}{number of iterations until convergence for each \code{lambda}}
  \item{$lambda}{sequence of tuning parameter values}
  \item{$penalty}{same as above}
  \item{$gamma}{same as above}
  \item{$alpha}{same as above}
  \item{$loglik}{log likelihood of the fitted model at each value of
              \code{lambda}}
  \item{$GCV}{generalized cross validation of the fitted model at each value of
           \code{lambda}}
  \item{$BIC}{Bayesian information criteria of the fitted model at each value of
           \code{lambda}}
  \item{$SE}{matrix of standard errors with \code{nvars} row and \code{nlambda} columns}           
}
\references{
\itemize{
\item Breheny, P. and Huang, J. (2011) Coordinate descent
            algorithms for nonconvex penalized regression, with applications to
            biological feature selection. \emph{Ann. Appl. Statist.}, 5: 232-253.
\item Fine J. and Gray R. (1999) A proportional hazards model for the
subdistribution of a competing risk.  \emph{JASA} 94:496-509.
\item Fu Z., Parikh C. and Zhou B.(2015). Penalized variable selection in competing risks regression. Manuscript submitted for publication.
}
}
\author{
Zhixuan Fu <zhixuan.fu@yale.edu>
}

\seealso{gcrrp, cmprsk, ncvreg}
\examples{
  #simulate competing risks data
  set.seed(10)
  ftime <- rexp(200)
  fstatus <- sample(0:2,200,replace=TRUE)
  cov <- matrix(runif(1000),nrow=200)
  dimnames(cov)[[2]] <- c('x1','x2','x3','x4','x5')
  
  #fit LASSO
  fit <- crrp(ftime, fstatus, cov, penalty="LASSO")
  #use BIC to select tuning parameters
  beta <- fit$beta[, which.min(fit$BIC)]
  beta.se <- fit$SE[, which.min(fit$BIC)]
  
  #fit adaptive LASSO
  weight <- 1/abs(crr(ftime, fstatus, cov)$coef)
  fit2 <-crrp(ftime, fstatus, cov, penalty="LASSO", penalty.factor=weight, weighted=TRUE)
  beta2 <- fit2$beta[, which.min(fit2$BIC)]
  beta2.se <- fit2$SE[, which.min(fit2$BIC)]
  
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{survival}% __ONLY ONE__ keyword per line
