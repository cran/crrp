\name{gcrrp}
\alias{gcrrp}
%- Also NEED an '\alias' for EACH other topic documented here.
\title{
  Group penalized variable selection in competing risks regression
}
\description{
Extends R package \pkg{grpreg} to the proportional subdistribution hazards (PSH) model (Fine and Gray, 1999). Performs penalized variable selection at the group level. Penalties include group LASSO, adaptive group LASSO, group SCAD, and group MCP. 
}
\usage{
  gcrrp(time, fstatus, X, failcode = 1, cencode = 0, group=1:ncol(X),
        penalty=c("gLASSO", "gMCP", "gSCAD"),gamma=switch(penalty, SCAD=3.7, 2.7),
        alpha=1, lambda.min=0.001, nlambda=50, lambda, eps=.001, 
        max.iter=1000, weighted=FALSE)
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
   design matrix; \code{crrp} standardizes and orthogonizes \code{X} by default
  } 
  \item{failcode}{
    code of fstatus that denotes the failure type of interest
  }  
  \item{cencode}{
    code of fstatus that denotes censored observations
  }
  \item{group}{vector of group indicator (see details)}
  \item{penalty}{penalty to be applied to the model. Either "gLASSO", "gSCAD", or "gMCP"}  
  \item{gamma}{tuning parameter of the gMCP/gSCAD penalty. Default is 2.7 for group MCP and 3.7 for group SCAD.}  
  \item{alpha}{tuning parameter indicating contributions from the MCP/SCAD penalty and the L2
               penalty. 
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
  \item{weighted}{Default is \code{FALSE}. If \code{TRUE}, it must be used with gLASSO to produce adaptive group LASSO penalty(see details)}
}
\details{
The \code{group} vector indicates the grouping of variables. For greatest efficiency, \code{group} should be a vector of consecutive integers, although unordered groups are also allowed.  

Penalties include group LASSO, group SCAD, and group MCP. We also include adaptive group LASSO by putting \code{weighted=TRUE}. The \code{gcrrp} function calculates data-adaptive weights formulated by the maximum parital likelihood estimator(MPLE) of the PSH model. The weight for each group is the inverse of the norm of the corresponding sub-vector of MPLE. The algorithm employed is the group coordinate descent algorithm. 
}
\value{
  Return a list of class \code{gcrrp} with components
  \item{$beta}{fitted coefficients matrix with \code{nvars} row and \code{nlambda} columns}
  \item{$iter}{number of iterations until convergence for each \code{lambda}}
  \item{$group}{same as above}
  \item{$lambda}{sequence of tuning parameter values}
  \item{$penalty}{same as above.}
  \item{$gamma}{same as above.}
  \item{$alpha}{same as above.}
  \item{$loglik}{log likelihood of the fitted model at each value of
              \code{lambda}}
  \item{$GCV}{generalized cross validation of the fitted model at each value of
           \code{lambda}}
  \item{$BIC}{Bayesian information criteria of the fitted model at each value of
           \code{lambda}}
}
\references{
\itemize{
  \item Breheny, P. and Huang, J. (2012) Group descent algorithms for
    nonconvex penalized linear and logistic regression models with grouped
    predictors.  \emph{Statistics and Computing}
  \item Fine J. and Gray R. (1999) A proportional hazards model for the
  subdistribution of a competing risk.  \emph{JASA} 94:496-509.
  \item Fu Z., Parikh C. and Zhou B.(2015). Penalized variable selection in competing risks regression. Manuscript submitted for publication.
  \item Huang J., Breheny, P. and Ma, S. (2012). A selective
    review of group selection in high dimensional
    models. \emph{Statistical Science}, 27: 481-499.
}
}
\author{
  Zhixuan Fu <zhixuan.fu@yale.edu>
}

\seealso{crrp, cmprsk, grpreg}
\examples{
  set.seed(10)
  ftime <- rexp(200)
  fstatus <- sample(0:2,200,replace=TRUE)
  cov <- matrix(runif(2000),nrow=200)
  dimnames(cov)[[2]] <- paste("x", 1:ncol(cov))
  group <- c(1,1,2,2,2,3,4,4,5,5)
  #fit gSCAD penalty
  fit1 <- gcrrp(ftime, fstatus, cov, group=group, penalty="gSCAD")
  beta1 <- fit1$beta[, which.min(fit1$BIC)]
  #fit adaptive gLASSO
  fit2 <- gcrrp(ftime, fstatus, cov, group=group, penalty="gLASSO", weighted=TRUE)
  beta2 <- fit2$beta[, which.min(fit2$BIC)]
}
% Add one or more standard keywords, see file 'KEYWORDS' in the
% R documentation directory.
\keyword{survival}% __ONLY ONE__ keyword per line
