#include <cstdlib> 
#include <R.h>
#include <Rinternals.h>
#include <new>

using namespace std;

class RGarbageCollectedObject {
public: 
  inline void* operator new(size_t t);
  inline void operator delete(void* p);
};

inline void* RGarbageCollectedObject::operator new(size_t t) {
  return R_alloc(t,1);
}

inline void RGarbageCollectedObject::operator delete(void* p) {
}

/////////////////////////////////////////////////////////////////////////////

SEXP getVectorElement(SEXP list, const char *str);

SEXP extractElement(SEXP args, const char* name);

int getListInt(SEXP list, const char *name, int def);

double getListDouble(SEXP list, const char *name, double def);

int getListBoolean(SEXP list, const char *name, int def);

const char* getListString(SEXP list, const char *name, const char* def);

int powi(int base, int exponent);

double quadratic_zero(double xl, double xm, double xr, double yl, double ym, double yr);
double cubic_minimum(double* b);

extern "C" {
  void dgesv_(int *n, int *rhs, double *A, int *lda, int* work, double *b, int* ldb, int* info);
}

double cubic_interpolation(double x1, double x2, double y1, double y2, double y1p, double y2p);

double beta_laplace(double x, double a);
