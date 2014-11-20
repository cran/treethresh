#include <cstdlib> 
#include <new>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "utils.h"

using namespace std;

/////////////////////////////////////////////////////////////////////////////

namespace treethresh {

class Options : public RGarbageCollectedObject {
public:
  int max_iter, max_depth, minimum_width, minimum_size;
  double a, tolerance, tolerance_grad, absolute_improvement, relative_improvement, absolute_criterion, beta_max, crit_signif, lr_signif, first_step, disparity_weight;
  Options(SEXP control);
};


Options::Options(SEXP control) {
  this->max_iter=getListInt(control,"max.iter",30);
  this->max_depth=getListInt(control,"max.depth",10);
  this->minimum_width=getListInt(control,"minimum.width",3);
  this->minimum_size=getListInt(control,"minimum.size",25);
  this->tolerance=getListDouble(control,"tolerance",1e-12);
  this->tolerance_grad=getListDouble(control,"tolerance.grad",1e-12);
  this->a=getListDouble(control,"a",0.5);
  this->absolute_improvement=getListDouble(control,"absolute.improvement",R_NegInf);
  this->relative_improvement=getListDouble(control,"relative.improvement",R_NegInf);  
  this->absolute_criterion=getListDouble(control,"absolute.criterion",0);
  this->beta_max=getListDouble(control,"beta.max",1e3);
  this->crit_signif=getListDouble(control,"crit.signif",0.9);
  this->lr_signif=getListDouble(control,"lr.signif",0.9);
  this->first_step=getListDouble(control,"first.step",0.2);
  this->disparity_weight=getListDouble(control,"disparity.weight",1);
}


Options* options;



/////////////////////////////////////////////////////////////////////////////

class Data: public RGarbageCollectedObject {
public:
  double *data, *beta;
  int dims, n;
  int* size;
  Data(SEXP args); 
  void compute_betas();
};

Data::Data(SEXP args) {
  SEXP data_sexp;
  data_sexp=extractElement(args,"data");
  n=length(data_sexp);
  data=REAL(data_sexp);
  dims=INTEGER(extractElement(args,"dims"))[0];
  size=INTEGER(extractElement(args,"size"));
  if (LOGICAL(extractElement(args,"use.beta"))[0])
    beta=REAL(data_sexp);
  else {
    beta=(double*)R_alloc(n,sizeof(double));
    compute_betas();
  }
}


void Data::compute_betas() {
  for (int pos=0; pos<n; pos++) {
    beta[pos]=beta_laplace(data[pos],options->a);
  }
}


/////////////////////////////////////////////////////////////////////////////

class Iterator : public RGarbageCollectedObject { 
public:
  Data* data;
  int *from, *to, *cur, *jumps;
  int pos, finished, n;
  Iterator(Data* data);
  inline int start();
  inline int next();
  inline double start_value();
  inline double next_value();
  inline double start_beta();
  inline double next_beta();
  inline int compute_n();
  Iterator* split_left(int dim, int pos);
  Iterator* split_right(int dim, int pos);
  //  void print();
};

Iterator::Iterator(Data* data) {
  this->data=data;
  from=(int*)R_alloc(data->dims,sizeof(int));
  to=(int*)R_alloc(data->dims,sizeof(int));
  cur=(int*)R_alloc(data->dims,sizeof(int));
  jumps=(int*)R_alloc(data->dims,sizeof(int));
  int n=1;
  for (int i=0; i<data->dims; i++) {
    jumps[i]=n;
    n*=data->size[i];
  }
}


inline int Iterator::start() {
  pos=0; finished=0;
  for (int i=0; i<data->dims; i++) {
    cur[i]=from[i];
    pos+=cur[i]*jumps[i];
  }
  return pos;
}

inline int Iterator::next() {
  for (int i=0; i<data->dims; i++) {
    if (cur[i]<to[i]-1) {
      cur[i]++;
      pos+=jumps[i];
      break;
    } else {
      if (i==data->dims-1) {
	finished=1;
        break;
      }
      pos-=(jumps[i]*(to[i]-from[i]-1));
      cur[i]=from[i];
    }
  }
  return pos;
}

inline double Iterator::start_value() {
  return data->data[start()];
}

inline double Iterator::next_value() {
  return data->data[next()];
}

inline double Iterator::start_beta() {
  return data->beta[start()];
}

inline double Iterator::next_beta() {
  return data->beta[next()];
}

Iterator* Iterator::split_left(int dim, int pos) {
  Iterator* it=new Iterator(data);
  for (int i=0; i<data->dims; i++) {
    it->from[i]=from[i];
    it->to[i]=to[i];
  }
  it->to[dim]=pos;
  return it;
}

Iterator* Iterator::split_right(int dim, int pos) {
  Iterator* it=new Iterator(data);
  for (int i=0; i<data->dims; i++) {
    it->from[i]=from[i];
    it->to[i]=to[i];
  }
  it->from[dim]=pos;
  return it;
}

int Iterator::compute_n() {
  int count=0;
  for (double z=start_beta(); !finished; z=next_beta())
    if (!(ISNAN(z)))
      count++;
  return count;
}

/*
void Iterator::print() {
  Rprintf("[");
  for (int i=0; i<data->dims; i++) {
    Rprintf(" %i-%i ",from[i],to[i]);
    if (i==data->dims-1)
      Rprintf("]\n");
    else
      Rprintf(";");
  }
}
*/
/////////////////////////////////////////////////////////////////////////////


double evaluate_llh(Iterator* it, double w) {
  double one_plus_w_beta, /*beta_by_one_plus_w_beta,*/ loglh=0;
  for (double beta=it->start_beta(); !it->finished; beta=it->next_beta()) {
    if (ISNAN(beta))
      continue;
    if (beta>options->beta_max)
      beta=options->beta_max;
    one_plus_w_beta=1+w*beta;
    //    beta_by_one_plus_w_beta=beta/one_plus_w_beta;
    loglh+=log(one_plus_w_beta);
  }
  return loglh;
}

double evaluate_score(Iterator* it, double w) {
  double one_plus_w_beta, beta_by_one_plus_w_beta, score=0;
  for (double beta=it->start_beta(); !it->finished; beta=it->next_beta()) {
    if (ISNAN(beta))
      continue;
    if (beta>options->beta_max)
      beta=options->beta_max;
    one_plus_w_beta=1+w*beta;
    beta_by_one_plus_w_beta=beta/one_plus_w_beta;
    score+=beta_by_one_plus_w_beta;
  }
  return score;
}


double compute_minimum_w(Iterator* it) {
  double a=options->a;
  double t=sqrt(2*log((double)it->data->n));
  double t_minus_a=t-a;
  return 1/((a*pnorm(t_minus_a,0,1,1,0))/dnorm(t_minus_a,0,1,0) - beta_laplace(t,options->a));
}

double search_solution(Iterator* it, double initial_guess, double minimum_w, int max_iter) {
  int iter, drop_left;
  double xl, xr, xm, xmn, xml, xmr, yl, yr, ym, ymn, yml, ymr;
  xl=minimum_w;
  yl=evaluate_score(it,xl);
  // Check whether we are already only in the negative
  if (yl<0)
    return xl;
  xm=initial_guess;
  ym=evaluate_score(it,xm);
  xr=1;
  yr=evaluate_score(it,xr);
  for (iter=0; iter<max_iter; iter++) {
    xmn=quadratic_zero(xl,xm,xr,yl,ym,yr);
    if (xmn>1) xmn=1;
    if (xmn<minimum_w) xmn=minimum_w;
    ymn=evaluate_score(it,xmn);
    if (xmn<xm) {
      xml=xmn; yml=ymn;
      xmr=xm;  ymr=ym;
    } else {
      xml=xm; yml=ym;
      xmr=xmn; ymr=ymn;
    }
    // We now have four values, we have to get red of the worst one
    // Case 1. ymr>0 (keep the last three. we need to keep the only negative one)
    if (ymr>0) {
      drop_left=1;
    } else {
     // Case 2. yml<0 (keep the first three. we need to keep the only positive)
      if (yml<0) {
	drop_left=0;
      } else {
	// Case 3. We have to positive and two negative ones. We can drop either or. We now base our decision on the range
	if (xmr-xl<xr-xml) {
	  // Drop the right one 
	  drop_left=0;
	} else {
	  // Drop the left one
	  drop_left=1;
	}
      }
    }
    if (drop_left) {
      xl=xml; yl=yml;
      xm=xmr; ym=ymr;
    } else {
      xm=xml; ym=yml;
      xr=xmr; yr=ymr;
    }
    if (fabs(ym)<options->tolerance_grad) 
      break;
  }
  return xm;
}

double compute_mle_w(Iterator* it, double initial_guess, double initial_step, int max_iter, double* loglh) {
  double w_est=search_solution(it,initial_guess,compute_minimum_w(it),max_iter);
  *loglh=evaluate_llh(it,w_est);
  return w_est;
}


/////////////////////////////////////////////////////////////////////////////

class Criterion : public RGarbageCollectedObject{
public:
  virtual void prepare_criterion(Iterator* full_parition, double current_w, double current_llh);
  virtual void prepare_search(Iterator* left_partition, Iterator* right_partition);
  virtual double compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff) = 0;    
};

void Criterion::prepare_criterion(Iterator* full_parition, double current_w, double current_llh) {
}

void Criterion::prepare_search(Iterator* left_partition, Iterator* right_partition) {
}



/////////////////////////////////////////////////////////////////////////////


class RCriterion : public Criterion {
  SEXP function, rho;
  double current_w, current_llh;
public:
  RCriterion(SEXP function, SEXP rho);
  void prepare_criterion(Iterator* full_parition, double current_w, double current_llh);
  double compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff);    
};

RCriterion::RCriterion(SEXP function, SEXP rho) {
  this->function=function;
  this->rho=rho;
}

void RCriterion::prepare_criterion(Iterator* full_parition, double current_w, double current_llh) {
  this->current_w=current_w;
  this->current_llh=current_llh;
}


double RCriterion::compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff) {
  SEXP left_data, right_data, left_beta, right_beta, call, ans, cur_w, cur_llh;
  PROTECT_INDEX ipx;
  int r_pos=0;
  double* data=left_partition->data->data;
  double* beta=left_partition->data->beta;
  double result;
  PROTECT(left_data=allocVector(REALSXP,left_partition->n));
  PROTECT(left_beta=allocVector(REALSXP,left_partition->n));
  PROTECT(right_data=allocVector(REALSXP,right_partition->n));
  PROTECT(right_beta=allocVector(REALSXP,right_partition->n));
  PROTECT(cur_w=allocVector(REALSXP,1));
  PROTECT(cur_llh=allocVector(REALSXP,1));
  for (int pos=left_partition->start(); !left_partition->finished; pos=left_partition->next()) {
    REAL(left_data)[r_pos]=data[pos];
    REAL(left_beta)[r_pos++]=beta[pos];
  }
  r_pos=0;
  for (int pos=right_partition->start(); !right_partition->finished; pos=right_partition->next()) {
    REAL(right_data)[r_pos]=data[pos];
    REAL(right_beta)[r_pos++]=beta[pos];

  }
  REAL(cur_w)[0]=current_w;
  REAL(cur_llh)[0]=current_llh;
  PROTECT(call = LCONS(function, LCONS(left_data, LCONS(left_beta, list4(right_data, right_beta,cur_w,cur_llh)))));



  PROTECT_WITH_INDEX(ans=eval(call,rho), &ipx);
  REPROTECT(ans = coerceVector(ans, REALSXP), ipx);
  result=REAL(ans)[0];
  UNPROTECT(8);
  return result;
}


/////////////////////////////////////////////////////////////////////////////

class LikelihoodCriterion : public Criterion {
  double left_w, left_diff, right_w, right_diff, old_llh;
public:
  LikelihoodCriterion();
  void prepare_criterion(Iterator* full_parition, double current_w, double current_llh);
  double compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff);    
};

LikelihoodCriterion::LikelihoodCriterion() {
  left_diff=options->first_step/2; 
  right_diff=options->first_step/2;
}


void LikelihoodCriterion::prepare_criterion(Iterator* full_partition, double current_w, double current_llh) {
  left_w=current_w;
  right_w=current_w;
  old_llh=current_llh;
}

double LikelihoodCriterion::compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff) {
  double left_llh, right_llh;
  double new_left_w=compute_mle_w(left_partition, left_w, 2*left_diff, options->max_iter, &left_llh);
  double new_right_w=compute_mle_w(right_partition, right_w, 2*right_diff, options->max_iter, &right_llh);
  left_diff=fabs(left_w-new_left_w);
  right_diff=fabs(right_w-new_right_w);
  left_w=new_left_w;
  right_w=new_right_w;
  return left_llh+right_llh - old_llh;
}


/////////////////////////////////////////////////////////////////////////////


class ScoreCriterion : public Criterion {
  double left_sum_score, right_sum_score, left_sum_ofisher, right_sum_ofisher;
  double *scores;
public:
  ScoreCriterion(Data* data);
  void prepare_criterion(Iterator* full_parition, double current_w, double current_llh);
  void prepare_search(Iterator* left_partition, Iterator* right_partition);
  double compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff);    
};

ScoreCriterion::ScoreCriterion(Data* data) {
  scores=(double*)R_alloc(data->n,sizeof(double));
}

void ScoreCriterion::prepare_criterion(Iterator* full_partition, double current_w, double current_llh) {
  double beta;
  for (int pos=full_partition->start(); !full_partition->finished; pos=full_partition->next()) {
    beta=full_partition->data->beta[pos];
    if (beta>options->beta_max)
      beta=options->beta_max;
    scores[pos]=beta/(1+current_w*beta);    
  }
}

void ScoreCriterion::prepare_search(Iterator* left_partition, Iterator* right_partition) {
  int pos;
  left_sum_score=0; 
  right_sum_score=0; 
  left_sum_ofisher=0;
  right_sum_ofisher=0;
  for (pos=left_partition->start(); !left_partition->finished; pos=left_partition->next()) {
    if (ISNAN(scores[pos]))
      continue;
    left_sum_score+=scores[pos];
    left_sum_ofisher+=(scores[pos]*scores[pos]);
  }
  for (pos=right_partition->start(); !right_partition->finished; pos=right_partition->next()) {
    if (ISNAN(scores[pos]))
      continue;
    right_sum_score+=scores[pos];
    right_sum_ofisher+=(scores[pos]*scores[pos]);
  }
}

double ScoreCriterion::compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff) {
  double diff_sum_score=0, diff_sum_ofisher=0;
  if (diff!=NULL) {
    for (int pos=diff->start(); !diff->finished; pos=diff->next()) {
      if (ISNAN(scores[pos]))
	continue;
      diff_sum_score+=scores[pos];
      diff_sum_ofisher+=(scores[pos]*scores[pos]);
    }
    left_sum_score+=diff_sum_score;
    right_sum_score-=diff_sum_score;
    left_sum_ofisher+=diff_sum_ofisher;
    right_sum_ofisher-=diff_sum_ofisher;
  }
  return ((left_sum_score*left_sum_score)/left_sum_ofisher + (right_sum_score*right_sum_score)/right_sum_ofisher); // /(left_partition->n+right_partition->n);
}

/////////////////////////////////////////////////////////////////////////////


class HeuristicCriterion : public Criterion {
  double left_sum_squares, right_sum_squares;
public:
  void prepare_search(Iterator* left_partition, Iterator* right_partition);
  double compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff);    
};

void HeuristicCriterion::prepare_search(Iterator* left_partition, Iterator* right_partition) {
  int pos;
  double* data=left_partition->data->data;
  left_sum_squares=0; 
  right_sum_squares=0; 
  for (pos=left_partition->start(); !left_partition->finished; pos=left_partition->next()) {
    if (ISNAN(data[pos]))
      continue;
    left_sum_squares+=(data[pos]*data[pos]);
  }
  for (pos=right_partition->start(); !right_partition->finished; pos=right_partition->next()) {
    if (ISNAN(data[pos]))
      continue;
    right_sum_squares+=(data[pos]*data[pos]);
  }
}

double HeuristicCriterion::compute_goodness(Iterator* left_partition, Iterator* right_partition, Iterator* diff) {
  double diff_sum_squares=0, p;
  double* data=left_partition->data->data;
  if (diff!=NULL) {
    for (int pos=diff->start(); !diff->finished; pos=diff->next()) {
      if (ISNAN(data[pos]))
	continue;
      diff_sum_squares+=(data[pos]*data[pos]);
    }
    left_sum_squares+=diff_sum_squares;
    right_sum_squares-=diff_sum_squares;
  }
  p=(double)left_partition->n/(double)(left_partition->n+right_partition->n);
  return fabs(left_sum_squares/left_partition->n - right_sum_squares/right_partition->n) + 4 * options->disparity_weight * (p-p*p);

}

///////////////////////////////////////////////////////////////////////////////


class TreeNode:public RGarbageCollectedObject {
public:
  TreeNode *left_child, *right_child, *parent;
  double criterion, neg_loglikelihood, w, alpha, cumulative_loglikelihood;
  int pruning_size, dim, pos, id;
  Iterator* it;
  TreeNode(TreeNode* parent, Iterator* it);
  void split(Criterion* sc, int depth);
  int compute_pruning_size();
  double compute_cumulative_loglikelihood();
  double search_best_prune(TreeNode** best);
  void prune(double alpha);
  void carry_out_pruning();
  //  void print(int depth);
  void issue_id();
  void issue_id(int* cur_id);
  void membership(double* membership);
  int num_nodes();
  void list_splits(double* matrix, int nrow);
  void list_splits(double* matrix, int nrow, int* counter);
};

TreeNode::TreeNode(TreeNode* parent, Iterator* it) {
  this->parent=parent;
  left_child=NULL;
  right_child=NULL;
  dim=-1;
  this->it=it;
  alpha=R_NegInf;
  if (parent==NULL) {
    double loglh;
    w=compute_mle_w(it,0.5,options->first_step,options->max_iter,&loglh);
    neg_loglikelihood=-loglh;
  }
}

int TreeNode::compute_pruning_size() {
  if ((left_child!=NULL) && (right_child!=NULL) && (alpha==R_NegInf))
    pruning_size=left_child->compute_pruning_size()+right_child->compute_pruning_size();
  else
    pruning_size=1;
  return pruning_size;
}

int TreeNode::num_nodes() {
  if ((left_child!=NULL) && (right_child!=NULL))
    return left_child->num_nodes()+right_child->num_nodes()+1;
  return 1;
}

void TreeNode::prune(double alpha) {
  if ((left_child!=NULL) && (right_child!=NULL) && (this->alpha==R_NegInf)) {
    this->alpha=alpha;
    left_child->prune(alpha);
    right_child->prune(alpha);
  }
}

double TreeNode::compute_cumulative_loglikelihood() {
  if ((left_child!=NULL) && (right_child!=NULL) && (alpha==R_NegInf)) 
    cumulative_loglikelihood=left_child->compute_cumulative_loglikelihood()+right_child->compute_cumulative_loglikelihood();
  else
    cumulative_loglikelihood=neg_loglikelihood;
  return cumulative_loglikelihood;  
}

double TreeNode::search_best_prune(TreeNode** best) {
  if ((left_child!=NULL) && (right_child!=NULL) && (pruning_size>1)) {
    TreeNode *best_left, *best_right;
    double crit_left=left_child->search_best_prune(&best_left);
    double crit_right=right_child->search_best_prune(&best_right);
    double crit=(neg_loglikelihood-cumulative_loglikelihood)/(double)(pruning_size-1);
    if ((crit_left<crit) && (crit_left<crit_right)) {
      *best=best_left;
      return crit_left;
    }
    if ((crit_right<crit) && (crit_right<crit_left)) {
      *best=best_right;
      return crit_right;
    }
    *best=this;
    return crit;
  } else {
    *best=this;
    return R_PosInf;
  }
}

void TreeNode::carry_out_pruning() {
  TreeNode* best;
  double alpha;
  if (left_child==NULL) {
    alpha=R_NaN;
    return;
  }
  while (this->alpha==R_NegInf) {
    R_CheckUserInterrupt();
    this->compute_cumulative_loglikelihood();
    this->compute_pruning_size();
    alpha=search_best_prune(&best);
    best->prune(alpha);
  }
}

/*
void TreeNode::print(int depth) {
  R_CheckUserInterrupt();
  if (dim==-1) {
    for (int i=0; i<depth; i++)
      Rprintf("| ");
    Rprintf("w = %f (alpha=%f)\n",w,alpha);
    return;
  }
  for (int i=0; i<depth; i++)
    Rprintf("| ");
  Rprintf("+ %i  < %i (alpha=%f, w=%f)\n",dim,pos,alpha,w);
  left_child->print(depth+1);
  for (int i=0; i<depth; i++)
    Rprintf("| ");
  Rprintf("+ %i >= %i (alpha=%f, w=%f)\n",dim,pos,alpha,w);
  right_child->print(depth+1);
}
*/

void TreeNode::issue_id(int* cur_id) {
  id=((*cur_id)++);
  if ((left_child!=NULL) && (right_child!=NULL)) {
    left_child->issue_id(cur_id);
    right_child->issue_id(cur_id);
  }
} 

void TreeNode::issue_id() {
  int cur_id=0;
  issue_id(&cur_id);
}


void TreeNode::split(Criterion* sc, int depth) {
  Iterator* left=new Iterator(it->data);
  Iterator* right=new Iterator(it->data);
  Iterator* diff=new Iterator(it->data);
  double crit, best_crit=R_NegInf;
  int best_dim=-1, best_pos=-1, accepted=0;
  R_CheckUserInterrupt();
  sc->prepare_criterion(it,w,-neg_loglikelihood);
  for (int dim=0; dim<it->data->dims; dim++) {
    for (int i=0; i<it->data->dims; i++) { 
      left->from[i]=it->from[i]; left->to[i]=it->to[i];
      right->from[i]=it->from[i]; right->to[i]=it->to[i];
      diff->from[i]=it->from[i]; diff->to[i]=it->to[i];
    }
    left->to[dim]=it->from[dim]+options->minimum_width;
    right->from[dim]=it->from[dim]+options->minimum_width;
    if (right->to[dim]-right->from[dim]<options->minimum_width)
      continue;
    left->n=left->compute_n();
    right->n=right->compute_n();
    sc->prepare_search(left,right);
    Iterator* cur_diff=NULL;
    for (int i=it->from[dim]+options->minimum_width-1; i<it->to[dim]-options->minimum_width; i++) {
      R_CheckUserInterrupt();
      crit=sc->compute_goodness(left,right,cur_diff);
      if (ISNAN(crit))
	crit=R_NegInf;
      if (crit>best_crit) {
	accepted=1;
	best_crit=crit;
	best_dim=dim;
	best_pos=i+1;
      }
      left->to[dim]=i+2;
      right->from[dim]=i+2;
      diff->from[dim]=i+1;
      diff->to[dim]=i+2;
      diff->n=diff->compute_n();
      left->n+=diff->n;
      right->n-=diff->n;
      cur_diff=diff;
    }
  }
  criterion=best_crit;
  if (parent!=NULL)
    if ((best_crit-parent->criterion<=parent->criterion*options->relative_improvement) && (best_crit-parent->criterion<=options->absolute_improvement))
      accepted=0;
  if (best_crit<=options->absolute_criterion)
    accepted=0;
  if (accepted && (depth<options->max_depth)) {
       for (int i=0; i<it->data->dims; i++) { 
	left->from[i]=it->from[i]; left->to[i]=it->to[i];
	right->from[i]=it->from[i]; right->to[i]=it->to[i];
	diff->from[i]=it->from[i]; diff->to[i]=it->to[i];
      }
      left->to[best_dim]=best_pos;
      right->from[best_dim]=best_pos;
      left->n=it->n/(it->to[best_dim]-it->from[best_dim])*(left->to[best_dim]-left->from[best_dim]);
      right->n=it->n/(it->to[best_dim]-it->from[best_dim])*(right->to[best_dim]-right->from[best_dim]);
      if ((left->n>=options->minimum_size) && (right->n>=options->minimum_size)) {
	double left_w, left_llh, right_w, right_llh;
	left_w=compute_mle_w(left,w,options->first_step,options->max_iter,&left_llh);
	right_w=compute_mle_w(right,w,options->first_step,options->max_iter,&right_llh);
	// Carry out a likelihood ratio test ...
       
        if (((options->lr_signif<=0) || (left_llh+right_llh+neg_loglikelihood>=0.5*qchisq(options->lr_signif,1,1,0))) && (fabs(left_w-right_w)>options->tolerance)) {
	  this->dim=best_dim;
	  this->pos=best_pos;
	  left_child=new TreeNode(this,left);
	  left_child->w=left_w;
	  left_child->neg_loglikelihood=-left_llh;
	  right_child=new TreeNode(this,right);
	  right_child->w=right_w;
	  right_child->neg_loglikelihood=-right_llh;
	  left_child->split(sc,depth+1);
	  right_child->split(sc,depth+1);
	}
      }
  }
}

void TreeNode::membership(double* membership) {
  if ((left_child!=NULL) && (right_child!=NULL)) {
    left_child->membership(membership);
    right_child->membership(membership);
  } else {
    for (pos=it->start(); !it->finished; pos=it->next()) 
      membership[pos]=id+1;
  }
}


void TreeNode::list_splits(double* matrix, int nrow) {
  int counter=0;
  list_splits(matrix,nrow, &counter);
}

void TreeNode::list_splits(double* matrix, int nrow, int* counter) {
  matrix[*counter]=id+1;
  if (parent!=NULL) {
    if (this==parent->left_child)
      matrix[*counter+1*nrow]=-(parent->id+1);
    else
      matrix[*counter+1*nrow]=parent->id+1;
  }
  else
    matrix[*counter+1*nrow]=NA_REAL;
  if ((left_child!=NULL) & (right_child!=NULL)) {
    matrix[*counter+2*nrow]=dim;
    matrix[*counter+3*nrow]=pos;
    matrix[*counter+4*nrow]=left_child->id+1;
    matrix[*counter+5*nrow]=right_child->id+1;
    matrix[*counter+6*nrow]=criterion;
  } else {
    for (int i=2; i<=6; i++)
      matrix[*counter+i*nrow]=NA_REAL;
  }
  matrix[*counter+7*nrow]=w;
  matrix[*counter+9*nrow]=-neg_loglikelihood;
  if (parent!=NULL)
    matrix[*counter+10*nrow]=parent->alpha;
  else
    matrix[*counter+10*nrow]=NA_REAL;
  (*counter)++;
  if ((left_child!=NULL) & (right_child!=NULL)) {
    left_child->list_splits(matrix,nrow,counter);
    right_child->list_splits(matrix,nrow,counter);
  }
}

}

//////////////////////////////////////////////////////////////////////////////

extern "C" {
  SEXP fit_tree(SEXP args) {
    using namespace treethresh;
    SEXP result, element, criterion_type;    
    Criterion* sc;
    options=new Options(extractElement(args,"control"));
    Data* data=new Data(args);
    Iterator* it=new Iterator(data);
    for (int i=0; i<data->dims; i++) { 
      it->from[i]=0; it->to[i]=data->size[i];
    }
    it->n=data->n;
    criterion_type=extractElement(args,"criterion");
    if (isString(criterion_type)) {
      if (strcmp(CHAR(STRING_ELT(criterion_type,0)),"score")==0)
	sc=new ScoreCriterion(data);
      else 
	if (strcmp(CHAR(STRING_ELT(criterion_type,0)),"likelihood")==0)
	  sc=new LikelihoodCriterion();
	else
	  sc=new HeuristicCriterion();
    } else 
      sc=new RCriterion(criterion_type,extractElement(args,"rho"));

    TreeNode* root=new TreeNode(NULL,it);
    root->split(sc,1);
    root->issue_id();
    root->carry_out_pruning();
    // Create output list
    PROTECT(result=allocVector(VECSXP,3));
    PROTECT(element=allocVector(REALSXP,data->n));
    root->membership(REAL(element));
    SET_VECTOR_ELT(result,0,element);
    UNPROTECT(1);
    int nrow=root->num_nodes();
    PROTECT(element=allocMatrix(REALSXP,nrow,12));
    root->list_splits(REAL(element),nrow);
    SET_VECTOR_ELT(result,1,element);
    UNPROTECT(1);
    PROTECT(element=allocVector(REALSXP,data->n));
    for (int i=0; i<data->n; i++)
      REAL(element)[i]=data->beta[i];
    SET_VECTOR_ELT(result,2,element);
    UNPROTECT(2);
    return result;
  }
}

