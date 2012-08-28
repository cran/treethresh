#include <cstdlib> 
#include <new>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "utils.h"

using namespace std;
/*
inline void* RGarbageCollectedObject::operator new(size_t t) {
  return R_alloc(t,1);
}

inline void RGarbageCollectedObject::operator delete(void* p) {
}
*/
/////////////////////////////////////////////////////////////////////////////

SEXP getVectorElement(SEXP list, const  char *str) {
  SEXP elmt = R_NilValue, names = getAttrib(list, R_NamesSymbol);
  int i;
  for (i = 0; i < length(list); i++)
    if(strcmp(CHAR(STRING_ELT(names, i)), str) == 0) {
      elmt = VECTOR_ELT(list, i);
      break;
    }
  return elmt;
}

SEXP extractElement(SEXP args, const char* name) {
  int i, nargs;
  if (!isList(args)) return R_NilValue;
  nargs=length(args)-1;
  for (i=0; i<nargs; i++) {
    args=CDR(args);
    if (strcmp(CHAR(PRINTNAME(TAG(args))),name)==0) 
      return CAR(args);
  }
  return R_NilValue;
}

int getListInt(SEXP list, const char *name, int def) {
  SEXP elt=getVectorElement(list,name);
  SEXP buf;
  int foo;
  if (elt==R_NilValue)
    return def;
  PROTECT(buf=coerceVector(elt,INTSXP));
  foo=INTEGER(buf)[0];
  UNPROTECT(1);
  return foo;
}

double getListDouble(SEXP list, const char *name, double def) {
  SEXP elt=getVectorElement(list,name);
  SEXP buf;
  double foo;
  if (elt==R_NilValue)
    return def;
  PROTECT(buf=coerceVector(elt,REALSXP));
  foo= REAL(buf)[0];
  UNPROTECT(1);
  return foo;
}

int getListBoolean(SEXP list, const char *name, int def) {
  SEXP elt=getVectorElement(list,name);
  SEXP buf;
  int foo;
  if (elt==R_NilValue)
    return def;
  PROTECT(buf=coerceVector(elt,LGLSXP));
  foo=LOGICAL(elt)[0];
  UNPROTECT(1);
  return foo;
}

const char* getListString(SEXP list, const char *name, const char* def) {
  SEXP elt=getVectorElement(list,name);
  if (elt==R_NilValue) 
    return def;  
  return CHAR(STRING_ELT(elt,0));
}

int powi(int base, int exponent) {
  int i, ans;
  ans=1;
  for (i=0; i<exponent; i++) ans*=base;
  return ans;
}

double cubic_minimum(double* b) {
  double discriminant, sol1, sol2;
  if (b[0]==0)
    return -b[2]/(2*b[1]);
  discriminant=4*b[1]*b[1]-12*b[0]*b[2];
  if (fabs(discriminant)<=0) {
    if (b[0]>0)
      return R_PosInf;
    else
      return R_NegInf;
  }
  discriminant=sqrt(discriminant);
  sol1=(-2*b[1]+discriminant)/(6*b[0]);
  sol2=(-2*b[1]-discriminant)/(6*b[0]);
  if (6*b[0]*sol1+2*b[1]<0)
    return sol1;
  else
    return sol2;
}

extern "C" {
  void dgesv_(int *n, int *rhs, double *A, int *lda, int* work, double *b, int* ldb, int* info);
}

double quadratic_zero(double xl, double xm, double xr, double yl, double ym, double yr) {
  double A[9];
  double b[3];
  int p[3];
  double discriminant, sqrt_discriminant,solution;
  int three=3, one=1, info;
  A[0]=xl*xl; A[3]=xl; A[6]=1;
  A[1]=xm*xm; A[4]=xm; A[7]=1;
  A[2]=xr*xr; A[5]=xr; A[8]=1;
  b[0]=yl; b[1]=ym; b[2]=yr;
  dgesv_(&three, &one, A, &three, p, b, &three, &info);
  // If there is close to no quadratic coefficient, use a linear solution
  discriminant=b[1]*b[1]-4*b[0]*b[2];
  // If there is close to no quadratic coefficient (or the quadratic approximation has no root), use a linear solution
  if ((fabs(b[0])<1e-16) || (discriminant<0)) {
    if (fabs(yl-yr)<1e-16)
      return xm;
    else
      return xl+(xr-xl)*yl/(yl-yr);
  }
  sqrt_discriminant=sqrt(discriminant);
  solution=(-b[1]+sqrt_discriminant)/(2*b[0]);
  if ((solution>xl) && (solution<xr))
    return solution;
  return (-b[1]-sqrt_discriminant)/(2*b[0]);
}

double cubic_interpolation(double x1, double x2, double y1, double y2, double y1p, double y2p) {
  double A[16];
  double b[4];
  int p[4];
  int four=4, one=1, info;
  A[0]=x1*x1*x1; A[1]=x2*x2*x2; A[2]=3*x1*x1; A[3]=3*x2*x2; 
  A[4]=x1*x1;    A[5]=x2*x2;    A[6]=2*x1;    A[7]=2*x2;
  A[8]=x1;       A[9]=x2;       A[10]=1;      A[11]=1;
  A[12]=1;       A[13]=1;       A[14]=0;      A[15]=0;
  b[0]=y1;       b[1]=y2;       b[2]=y1p;     b[3]=y2p;
  dgesv_(&four, &one, A, &four, p, b, &four, &info);
  return cubic_minimum(b);
}

// Compute beta laplace

double beta_laplace(double x, double a) {
  double second_part,x_plus_a, x_minus_a;
  if (ISNAN(x))
    return NA_REAL;
  if (a>35) a=35;
  x=fabs(x);  
  x_plus_a=x+a;
  x_minus_a=x-a;
  if (x_plus_a>35)
    second_part=1/x;
  else
    second_part=pnorm(x_plus_a,0,1,0,0)/dnorm(x_plus_a,0,1,0);
  return 0.5*a*(pnorm(x_minus_a,0,1,1,0)/dnorm(x_minus_a,0,1,0)+second_part)-1; 
}

