#include <math.h>
#include <string.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <R.h>
#include <R_ext/Applic.h>
#include <stdlib.h>
#define LEN sizeof(double)

void scorehessian (double *t2, int *ici, double *x, int *ncov, int *nin, double *wt, double *eta, double *st, double *w, double *lik) 
{
	const int np=ncov[0], n=nin[0];
	int i, j2, k, j;
	double likli=0, s0, *a, wye[n];
// a is the matrix of covariates
//s1 is the score
	a=(double*)malloc(n*np*LEN);
//initialization


    for (int i = 0; i < n; i++)
        for (int k = 0; k < np; k++)
				*(a+i*np+k)= x[i + n * k];
   
//start here
    for (i = 0; i < n; i++)
	{
		if (ici[i] != 1) continue;
		likli += eta[i];
		st[i]+=1;
		// score
		s0 = 0;
  		for (j = 0; j < n; j++)
		     wye[j] = 0;
		for (k = 0; k < n; k++)
		{
	
			if (t2[k] < t2[i] && ici[k] <= 1) continue;//leave those out of risk set
	            if (t2[k] >= t2[i]) 
                wye[k]=exp(eta[k]);				
			else 
                wye[k] = exp(eta[k]) * wt[i] / wt[k];
                

          s0 += wye[k];
        }		
		for (j2=0; j2<n; j2++){
            st[j2] += -wye[j2]/s0;
            w[j2]+=wye[j2]/s0-pow(wye[j2],2)/(s0*s0);
            }
            
        
        likli -= log(s0);
		//end of score
  }
  
  *lik=likli;
  free(a);     
}    

SEXP standardize(SEXP X_) {
  // Declarations
  int n = nrows(X_);
  int p = ncols(X_);
  SEXP XX_, c_, s_;
  PROTECT(XX_ = allocMatrix(REALSXP, n, p));
  PROTECT(c_ = allocVector(REALSXP, p));
  PROTECT(s_ = allocVector(REALSXP, p));
  double *X = REAL(X_);
  double *XX = REAL(XX_);
  double *c = REAL(c_);
  double *s = REAL(s_);

  for (int j=0; j<p; j++) {
    // Center
    c[j] = 0;
    for (int i=0; i<n; i++) {
      c[j] += X[j*n+i];
    }
    c[j] = c[j] / n;
    for (int i=0; i<n; i++) XX[j*n+i] = X[j*n+i] - c[j];

    // Scale
    s[j] = 0;
    for (int i=0; i<n; i++) {
      s[j] += pow(XX[j*n+i], 2);
    }
    s[j] = sqrt(s[j]/n);
    for (int i=0; i<n; i++) XX[j*n+i] = XX[j*n+i]/s[j];
  }

  // Return list
  SEXP res;
  PROTECT(res = allocVector(VECSXP, 3));
  SET_VECTOR_ELT(res, 0, XX_);
  SET_VECTOR_ELT(res, 1, c_);
  SET_VECTOR_ELT(res, 2, s_);
  UNPROTECT(4);
  return(res);
}

//From penalty file

double loglik (double *t2, int *ici, double *x, int ncov, int nin, double *wt, double *b)
{

    int i,j, j1;
	const int p=ncov,  n=nin;
	double likli=0, zb, s0;

    for (i = 0; i < n; i++)
	{
		if (ici[i] != 1) continue;
			//first part of logliklihood
			for (j = 0; j < p; j++)
				likli += b[j] * x[n*j+i];
   //second part of logliklihood
		s0=0;
		for (j = 0; j < n; j++)
		{
			if (t2[j] < t2[i] && ici[j] <= 1) continue;
			zb = 0.0;

				for (j1 = 0; j1 < p; j1 ++)
					zb += b[j1] *x[n*j1+j];
					
			if (t2[j] >= t2[i]) 
				s0 += exp(zb);
			else 
				s0 += exp(zb) * wt[i] / wt[j]; 

		}

		likli -= log(s0);	

	}
		return likli;
}





int checkConvergence(double *beta, double *beta_old, double eps, int l, int p) {
  int converged = 1;
  for (int j=0; j<p; j++) {
    if (fabs((beta[l*p+j]-beta_old[j])/beta_old[j]) > eps) {
      converged = 0;
      break;
    }
  }
  return(converged);
}

double MCP(double z, double l1, double l2, double gamma, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else if (fabs(z) <= gamma*l1*(1+l2)) return(s*(fabs(z)-l1)/(v*(1+l2-1/gamma)));
  else return(z/(v*(1+l2)));
}

double SCAD(double z, double l1, double l2, double gamma, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else if (fabs(z) <= (l1*(1+l2)+l1)) return(s*(fabs(z)-l1)/(v*(1+l2)));
  else if (fabs(z) <= gamma*l1*(1+l2)) return(s*(fabs(z)-gamma*l1/(gamma-1))/(v*(1-1/(gamma-1)+l2)));
  else return(z/(v*(1+l2)));
}

double lasso(double z, double l1, double l2, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else return(s*(fabs(z)-l1)/(v+l2));
}

double gMCP(double z, double l1, double l2, double gamma, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else if (fabs(z) <= gamma*l1*(1+l2)) return(s*(fabs(z)-l1)/(v*(1+l2-1/gamma)));
  else return(z/(v*(1+l2)));
}

double gSCAD(double z, double l1, double l2, double gamma, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else if (fabs(z) <= (l1*(1+l2)+l1)) return(s*(fabs(z)-l1)/(v*(1+l2)));
  else if (fabs(z) <= gamma*l1*(1+l2)) return(s*(fabs(z)-gamma*l1/(gamma-1))/(v*(1-1/(gamma-1)+l2)));
  else return(z/(v*(1+l2)));
}

double gLASSO(double z, double l1, double l2, double v) {
  double s=0;
  if (z > 0) s = 1;
  else if (z < 0) s = -1;
  if (fabs(z) <= l1) return(0);
  else return(s*(fabs(z)-l1)/(v+l2));
}

// Euclidean norm
double norm(double *x, int p) {
  double x_norm = 0;
  for (int j=0; j<p; j++) x_norm = x_norm + pow(x[j],2);
  x_norm = sqrt(x_norm);
  return(x_norm);
}

// Weighted cross product of y with jth column of x
double wcrossprod(double *X, double *y, double *w, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += X[nn+i]*y[i]*w[i];
  return(val);
}

// Weighted sum of squares of jth column of X
double wsqsum(double *X, double *w, int n, int j) {
  int nn = n*j;
  double val=0;
  for (int i=0;i<n;i++) val += w[i] * pow(X[nn+i], 2);
  return(val);
}


///////////////////////////////////////////////////////////////////////////////////////

SEXP cleanupB(double *a, double *eta, double *wye, double *st, double *w, SEXP beta, SEXP Dev, SEXP iter, SEXP residuals, SEXP score, SEXP hessian) {
  Free (a);
  Free(eta);
  Free(wye);
  Free(st);
  Free(w);
  SEXP res;
  PROTECT(res = allocVector(VECSXP, 6));
  SET_VECTOR_ELT(res, 0, beta);
  SET_VECTOR_ELT(res, 1, Dev);
  SET_VECTOR_ELT(res, 2, iter);
  SET_VECTOR_ELT(res, 3, residuals);
  SET_VECTOR_ELT(res, 4, score);
  SET_VECTOR_ELT(res, 5, hessian);
  UNPROTECT(7);
  return(res);
}


//////////////////////////////////////////////////////////////////////////////////
//start group cordinate descent
SEXP cdfit_psh(SEXP x_, SEXP t2_, SEXP ici_, SEXP wt_, SEXP penalty_, SEXP lambda, SEXP esp_, SEXP max_iter_, SEXP gamma_, SEXP multiplier, SEXP alpha_) {
     //Declaration
     int n = length(t2_);
     int p=length(x_)/n;
     int L=length(lambda);
     double nullDev;
     SEXP res, beta, Dev, iter, residuals, score, hessian;
     PROTECT(beta = allocVector(REALSXP, L*p));
     double *b = REAL(beta);
     for (int j=0; j<(L*p); j++) b[j] = 0;  
     PROTECT (score = allocVector(REALSXP, L*n));
     double *s=REAL(score);  
     for (int i=0; i<(L*n); i++) s[i] = 0;  
     PROTECT (hessian = allocVector(REALSXP, L*n));
     double *h=REAL(hessian);  
     for (int i=0; i<(L*n); i++) h[i] = 0;  
     PROTECT(residuals = allocVector(REALSXP, n));
     double *r = REAL(residuals);
     PROTECT(Dev = allocVector(REALSXP, L));
     for (int i=0; i<L; i++) REAL(Dev)[i] = 0;
     PROTECT(iter = allocVector(INTSXP, L));
     for (int i=0; i<L; i++) INTEGER(iter)[i] = 0;
     double *a = Calloc(p, double);    // Beta from previous iteration
     for (int j=0; j<p; j++) a[j] = 0;                   // Beta0 from previous iteration
     double *st = Calloc(n, double);
     for (int i=0; i<n; i++) st[i]=0;
     double *w = Calloc(n, double);
     for ( int i=0; i<n; i++) w[i]=0;
     double *x = REAL(x_);
     double *t2 = REAL(t2_);
     double *wt=REAL(wt_);
     int *ici=INTEGER(ici_);
     const char *penalty = CHAR(STRING_ELT(penalty_, 0));
     double *lam = REAL(lambda);
     double esp = REAL(esp_)[0];
     int max_iter = INTEGER(max_iter_)[0];
     double gamma = REAL(gamma_)[0];
     double *m = REAL(multiplier);
     double alpha = REAL(alpha_)[0];
     double *eta = Calloc(n, double);
     for (int i=0; i<n; i++) eta[i] = 0; 
     double *wye =Calloc(n, double);
     double xwr, xwx, u, v, l1, l2, shift, likli, s0, si;
     int converged;
     
//end of declaration;

//initialization
  nullDev=-2*loglik(t2, ici, x, p, n, wt, a);
  // Path
for (int l=0; l<L; l++) {
    
    if (l != 0) 
      for (int j=0; j<p; j++) a[j] = b[(l-1)*p+j];
           
  while (INTEGER(iter)[l] < max_iter) {
	   
       
       if (REAL(Dev)[l]-nullDev>0.99*nullDev) break;
       
       INTEGER(iter)[l]++;
       
           //calculate score and w
    likli=0;
    for (int i = 0; i < n; i++){
		st[i] = 0;
		w[i]= 0;
	}
    for (int i = 0; i < n; i++)
	{
		if (ici[i] != 1) continue;
		likli += eta[i];
		st[i]+=1;
		// score
		s0 = 0;
  		for (int j1 = 0; j1 < n; j1++)
		     wye[j1] = 0;
		for (int k = 0; k < n; k++)
		{	
            if (t2[k] < t2[i] && ici[k] <= 1) continue;
            
            if (t2[k] >= t2[i]) 
                wye[k]=exp(eta[k]);				
			else 
                wye[k] = exp(eta[k]) * wt[i] / wt[k];
          s0 += wye[k];
        }		
		for (int j2=0; j2<n; j2++){
            st[j2] += -wye[j2]/s0;
            w[j2]+=wye[j2]/s0-pow(wye[j2],2)/(s0*s0);
            }            
        likli -= log(s0);
  }
  
	  for ( int j3=0; j3<n; j3++){
          if (w[j3]==0) r[j3]=0; 
          else r[j3]=st[j3]/w[j3];
          }
          
           //calculate xwr and xwx
           
    for (int j=0; j<p; j++) {
          xwr = wcrossprod(x, r, w, n, j);
	      xwx = wsqsum(x, w, n, j);
          u=xwr/n+(xwx/n)*a[j];
          v=xwx/n;
            
            // Update b_j
	    l1 = lam[l] * m[j] * alpha;
	    l2 = lam[l] * m[j] * (1-alpha);
	    if (strcmp(penalty,"MCP")==0) b[l*p+j] = MCP(u, l1, l2, gamma, v);
	    if (strcmp(penalty,"SCAD")==0) b[l*p+j] = SCAD(u, l1, l2, gamma, v);
	    if (strcmp(penalty,"LASSO")==0) b[l*p+j] = lasso(u, l1, l2, v);
	    
	          // Update r
	      shift = b[l*p+j] - a[j];
	      if (shift !=0) {
		for (int i=0;i<n;i++) {
		  si = shift*x[j*n+i];
		  r[i] -= si;
		  eta[i] += si;
		}
	      }//end shift
           
       }//for j=0 to p              
	// Check for convergence
	converged = checkConvergence(b, a, esp, l, p);
	for (int i=0; i<p; i++) 
    a[i] = b[l*p+i];
    	
    REAL(Dev)[l]=loglik(t2, ici, x, p, n, wt, a)*(-2);
    for ( int i=0; i<n; i++){
        s[l*n+i]=st[i];
        h[l*n+i]=w[i];
    }          
    if (converged)  break;
  //for converge 

 }//for while loop 
 
}

  res = cleanupB(a, eta, wye, st, w, beta, Dev, iter, residuals, score, hessian);
  return(res);
} 




// Group descent update 
void gd_psh(double *b, double *x, double *r, double *w, double *eta, int g, int *K1, int n, int p, int l, const char *penalty, double lam1, double lam2, double gamma, double *a) {

  // Calculate z
  int K = K1[g+1] - K1[g];
  double *z = Calloc(K, double);
  double *v = Calloc(K, double);
  
  for (int j=K1[g]; j<K1[g+1]; j++){
  z[j-K1[g]] = wcrossprod(x, r, w, n, j)/n+wsqsum(x, w, n, j)/n*a[j];
  v[j-K1[g]] = wsqsum(x, w, n, j)/n;}
  double z_norm = norm(z,K);
  
// Update b
  double len=0;
  for (int j=K1[g]; j<K1[g+1]; j++) {
  if (strcmp(penalty, "gLASSO")==0) len = gLASSO(z_norm, lam1, lam2, v[j-K1[g]]);
  if (strcmp(penalty, "gMCP")==0) len = gMCP(z_norm, lam1, lam2, gamma, v[j-K1[g]]);
  if (strcmp(penalty, "gSCAD")==0) len = gSCAD(z_norm, lam1, lam2, gamma, v[j-K1[g]]);
  if ((len != 0) | (a[K1[g]] != 0)) {
    // If necessary, update b and r
      b[l*p+j] = len * z[j-K1[g]] / z_norm;
      double shift = b[l*p+j]-a[j];
      for (int i=0; i<n; i++) {
	double si = shift*x[j*n+i];
	r[i] -= si;
	eta[i] += si;
      }//for i=0 to n loop
    }//if loop
  }

  Free(z);
  Free(v);
}
//////////////////////////////////////////////////////////////////////////////////
//start group cordinate descent
SEXP gcdfit_psh(SEXP x_, SEXP t2_, SEXP ici_, SEXP wt_, SEXP K1_, SEXP penalty_, SEXP lambda, SEXP esp_, SEXP max_iter_, SEXP gamma_, SEXP multiplier, SEXP alpha_) {
     //Declaration
     int n = length(t2_);
     int p=length(x_)/n;
     int L=length(lambda);
     int J = length(K1_) - 1;
     int *K1 = INTEGER(K1_);
     double nullDev;
     SEXP res, beta, Dev, iter, residuals, score, hessian;
     PROTECT(beta = allocVector(REALSXP, L*p));
     double *b = REAL(beta);
     for (int j=0; j<(L*p); j++) b[j] = 0;  
     PROTECT (score = allocVector(REALSXP, L*n));
     double *s=REAL(score);  
     for (int i=0; i<(L*n); i++) s[i] = 0;  
     PROTECT (hessian = allocVector(REALSXP, L*n));
     double *h=REAL(hessian);  
     for (int i=0; i<(L*n); i++) h[i] = 0;  
     PROTECT(residuals = allocVector(REALSXP, n));
     double *r = REAL(residuals);
     PROTECT(Dev = allocVector(REALSXP, L));
     for (int i=0; i<L; i++) REAL(Dev)[i] = 0;
     PROTECT(iter = allocVector(INTSXP, L));
     for (int i=0; i<L; i++) INTEGER(iter)[i] = 0;
     double *a = Calloc(p, double);    // Beta from previous iteration
     for (int j=0; j<p; j++) a[j] = 0;                   // Beta0 from previous iteration
     double *st = Calloc(n, double);
     for (int i=0; i<n; i++) st[i]=0;
     double *w = Calloc(n, double);
     for ( int i=0; i<n; i++) w[i]=0;
     double *x = REAL(x_);
     double *t2 = REAL(t2_);
     double *wt=REAL(wt_);
     int *ici=INTEGER(ici_);
     const char *penalty = CHAR(STRING_ELT(penalty_, 0));
     double *lam = REAL(lambda);
     double esp = REAL(esp_)[0];
     int max_iter = INTEGER(max_iter_)[0];
     double gamma = REAL(gamma_)[0];
     double *m = REAL(multiplier);
     double alpha = REAL(alpha_)[0];
     double *eta = Calloc(n, double);
     for (int i=0; i<n; i++) eta[i] = 0; 
     double *wye =Calloc(n, double);
     double l1, l2, likli, s0;
     int converged;
     
//end of declaration;

//initialization
  nullDev=-2*loglik(t2, ici, x, p, n, wt, a);
  // Path
for (int l=0; l<L; l++) {
    
    if (l != 0) 
      for (int j=0; j<p; j++) a[j] = b[(l-1)*p+j];
           
  while (INTEGER(iter)[l] < max_iter) {
	   
       
       if (REAL(Dev)[l]-nullDev>0.99*nullDev) break;
       
       INTEGER(iter)[l]++;
       
    likli=0;
    for (int i = 0; i < n; i++){
		st[i] = 0;
		w[i]= 0;
	}
    for (int i = 0; i < n; i++)
	{
		if (ici[i] != 1) continue;
		likli += eta[i];
		st[i]+=1;
		// score
		s0 = 0;
  		for (int j1 = 0; j1 < n; j1++)
		     wye[j1] = 0;
		for (int k = 0; k < n; k++)
		{	
            if (t2[k] < t2[i] && ici[k] <= 1) continue;
            
            if (t2[k] >= t2[i]) 
                wye[k]=exp(eta[k]);				
			else 
                wye[k] = exp(eta[k]) * wt[i] / wt[k];
          s0 += wye[k];
        }		
		for (int j2=0; j2<n; j2++){
            st[j2] += -wye[j2]/s0;
            w[j2]+=wye[j2]/s0-pow(wye[j2],2)/(s0*s0);
            }            
        likli -= log(s0);
  }
  
	  for ( int j3=0; j3<n; j3++){
          if (w[j3]==0) r[j3]=0; 
          else r[j3]=st[j3]/w[j3];
          }
          
          
   	// Update penalized groups
	for (int g=0; g<J; g++) {
	  l1 = lam[l] * m[g] * alpha *sqrt(K1[g+1]-K1[g]);
	  l2 = lam[l] * m[g] * (1-alpha)*sqrt(K1[g+1]-K1[g]);	  
	  gd_psh(b, x, r, w, eta, g, K1, n, p, l, penalty, l1, l2, gamma, a);
	   
	}
              
	// Check for convergence
	converged = checkConvergence(b, a, esp, l, p);
	for (int i=0; i<p; i++) 
    a[i] = b[l*p+i];
    	
    REAL(Dev)[l]=loglik(t2, ici, x, p, n, wt, a)*(-2);
    for ( int i=0; i<n; i++){
        s[l*n+i]=st[i];
        h[l*n+i]=w[i];
    }          
    if (converged)  break;
  //for converge 

 }//for while loop 
 
}

  res = cleanupB(a, eta, wye, st, w, beta, Dev, iter, residuals, score, hessian);
  return(res);
} 
