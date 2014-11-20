#include <cstdlib> 
#include <new>
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>
#include "utils.h"

#define round(x) fround(x,1.0)

using namespace std;

namespace blockthresh {

/////////////////////////////////////////////////////////////////////////////

class Options : public RGarbageCollectedObject {
public:
  int max_iter, max_depth, minimum_width, minimum_size, min_minimum_size, min_minimum_width;
  double a, tolerance_grad, tolerance,  absolute_improvement, relative_improvement, absolute_criterion, beta_max, rescale_quantile, lr_signif, first_step, min_size_scale_factor, min_width_scale_factor;
  Options(SEXP control);
};


Options::Options(SEXP control) {
  this->max_iter=getListInt(control,"max.iter",30);
  this->max_depth=getListInt(control,"max.depth",10);
  this->minimum_width=getListInt(control,"minimum.width",4);
  this->minimum_size=getListInt(control,"minimum.size",16);
  this->min_minimum_width=getListInt(control,"min.minimum.width",1);
  this->min_minimum_size=getListInt(control,"min.minimum.size",4);
  this->tolerance_grad=getListDouble(control,"tolerance.grad",1e-12);
  this->tolerance=getListDouble(control,"tolerance",1e-8);
  this->a=getListDouble(control,"a",0.5);
  this->absolute_improvement=getListDouble(control,"absolute.improvement",R_NegInf);
  this->relative_improvement=getListDouble(control,"relative.improvement",R_NegInf);  
  this->absolute_criterion=getListDouble(control,"absolute.criterion",0);
  this->beta_max=getListDouble(control,"beta.max",35);
  this->rescale_quantile=getListDouble(control,"rescale.quantile",0.5);
  this->lr_signif=getListDouble(control,"lr.signif",0.9);
  this->first_step=getListDouble(control,"first.step",0.2);
  this->min_size_scale_factor=getListDouble(control,"min.size.scale.factor",1);
  this->min_width_scale_factor=getListDouble(control,"min.width.scale.factor",1);
}


Options* options;

///////////////////////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////////////////////////////

class Data: public RGarbageCollectedObject {
public:
  double **data, **beta, *weights; // data and beta: top level is block, then "column-major" F77-like data
  int *min_width, *min_size; // for each block the minimum size of a partition and the minimum "width" of a partition 
  int dims, n_blocks, use_beta;  // dims is the dimension of the blocks (typically 2 for 2D images), b_blocks in the number of blocks (i.e. block of wavelet coefficients)
  int *size, *n; // SIZE WAS AN ARRAY ( OLD VERSION: 1 entry per dimension (matrices can be "rectangles", don't need to be squares)
                 //                     NEW VERSION: 1 entry per block ("squared" matrices / arrays)
  int full_size; // size (i.e. length along one dimension) of the biggest block 
  SEXP beta_list_sexp;
  Data(SEXP args); 
};

Data::Data(SEXP args) {
  SEXP data_sexp, beta_sexp, weights_sexp;
  use_beta=LOGICAL(extractElement(args,"use.beta"))[0];
  dims=INTEGER(extractElement(args,"dims"))[0];
  data_sexp=extractElement(args,"data");
  weights_sexp=extractElement(args,"weights");
  n_blocks=length(data_sexp);
  weights=REAL(weights_sexp);
  n=(int*)R_alloc(n_blocks,sizeof(int));
  size=(int*)R_alloc(n_blocks,sizeof(int));
  data=(double**)R_alloc(n_blocks,sizeof(double*));
  beta=(double**)R_alloc(n_blocks,sizeof(double*));
  min_width=(int*)R_alloc(n_blocks,sizeof(int));
  min_size=(int*)R_alloc(n_blocks,sizeof(int));
  full_size=0;
  if (!use_beta)
    beta_list_sexp=PROTECT(allocVector(VECSXP,n_blocks));
  for (int i=0; i<n_blocks; i++) {
    data[i]=REAL(VECTOR_ELT(data_sexp,i));
    n[i]=length(VECTOR_ELT(data_sexp,i));
    size[i]=(int)round(R_pow(n[i],1/(double)dims));
    if (size[i]>full_size) 
      full_size=size[i];
    if (use_beta)
      beta[i]=data[i];
    else {
      PROTECT(beta_sexp=allocVector(REALSXP,n[i]));
      beta[i]=REAL(beta_sexp);
      SET_VECTOR_ELT(beta_list_sexp,i,beta_sexp);
      UNPROTECT(1);
      for (int pos=0; pos<n[i]; pos++) {
	beta[i][pos]=beta_laplace(data[i][pos],options->a);
      }
    }
  }
  for (int i=0; i<n_blocks; i++) {
    min_size[i]=(int)round(options->minimum_size*R_pow((double)size[i]/(double)full_size,options->min_size_scale_factor*(double)dims));
    if (min_size[i]<options->min_minimum_size)
      min_size[i]=options->min_minimum_size;
    min_width[i]=(int)round(options->minimum_width*R_pow((double)size[i]/(double)full_size,options->min_width_scale_factor));
    if (min_width[i]<options->min_minimum_width)
      min_width[i]=options->min_minimum_width;
  }
}



class Iterator : public RGarbageCollectedObject { 
public:
  Data* data;
  int *from, *to, *cur, *jumps;
  int finished, block_id, empty, pos;
  Iterator(Data* data, int block_id);
  inline int start();
  inline int next();
  inline double start_value();
  inline double next_value();
  inline double start_beta();
  inline double next_beta();
  inline int compute_n();
  Iterator* split_left(int dim, int pos);
  Iterator* split_right(int dim, int pos);
  void print();
};

Iterator::Iterator(Data* data, int block_id) {
  this->data=data;
  this->block_id=block_id;
  from=(int*)R_alloc(data->dims,sizeof(int));
  to=(int*)R_alloc(data->dims,sizeof(int));
  cur=(int*)R_alloc(data->dims,sizeof(int));
  jumps=(int*)R_alloc(data->dims,sizeof(int));
  int n=1;
  for (int i=0; i<data->dims; i++) {
    jumps[i]=n;
    n*=data->size[block_id];
  }
  empty=0;
}

inline int Iterator::start() {
  pos=0; finished=empty;
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
  if (empty)
    return R_NaN;
  return data->data[block_id][start()];
}

inline double Iterator::next_value() {
  return data->data[block_id][next()];
}

inline double Iterator::start_beta() {
  if (empty)
    return R_NaN;
  return data->beta[block_id][start()];
}

inline double Iterator::next_beta() {
  return data->beta[block_id][next()];
}

int Iterator::compute_n() {
  int count=0;
  if (!empty)
    for (double z=start_beta(); !finished; z=next_beta())
      if (!(ISNAN(z)))
	count++;
  return count;
}

/////////////////////////////////////////////////////////////////////////////


class IteratorStack : public RGarbageCollectedObject {
public:
  Data* data;
  Iterator** iters;
  int *from, *to;
  int empty, df;
  double weighted_df;
  IteratorStack(Data* data);
  int split(IteratorStack* left, IteratorStack* right, IteratorStack* diff, int at_dim, int at_pos);
  int get_number_of_active_partitions();
  void print();
};

IteratorStack::IteratorStack(Data* data) {
  this->data=data;
  empty=0;
  iters=(Iterator**)R_alloc(data->n_blocks,sizeof(Iterator*));
  from=(int*)R_alloc(data->dims,sizeof(int));
  to=(int*)R_alloc(data->dims,sizeof(int));
  for (int i=0; i<data->n_blocks; i++) {
    iters[i]=new Iterator(data,i);
    for (int j=0; j<data->dims; j++) {
      iters[i]->from[j]=0; 
      iters[i]->to[j]=data->size[i];
    }
  }
}

IteratorStack* create_full_iterator_stack(Data* data) {
  IteratorStack* its=new IteratorStack(data);
  for (int j=0; j<data->dims; j++) {
    its->from[j]=0;
    its->to[j]=data->full_size;
    for (int i=0; i<data->n_blocks; i++) {
      its->iters[i]->from[j]=0;
      its->iters[i]->to[j]=data->size[i];
    }
  }
  return its;
}

int IteratorStack::get_number_of_active_partitions() {
  int count=0;
  for (int i=0; i<data->n_blocks; i++)
    if (!iters[i]->empty)
      count++;
  return count;
}

/*
void IteratorStack::print() {
    Rprintf("-------------------------------------------------------------------\n");
  for (int j=0; j<data->dims; j++) {
    Rprintf("Dimension %i:\n",j);
    Rprintf("FROM %3i ( ",from[j]);
    for (int block_id=0; block_id<data->n_blocks; block_id++) 
      Rprintf("%3i ",iters[block_id]->from[j]);
    Rprintf(")\nTO   %3i ( ",to[j]);
    for (int block_id=0; block_id<data->n_blocks; block_id++)
      Rprintf("%3i ",iters[block_id]->to[j]);
    Rprintf(")\n");
  }
  Rprintf("EMPTY%3i ( ",empty);
  for (int block_id=0; block_id<data->n_blocks; block_id++)
    Rprintf("%3i ",iters[block_id]->empty);
  Rprintf(")\n");
  Rprintf("-------------------------------------------------------------------\n");
}
*/

int IteratorStack::split(IteratorStack* left, IteratorStack* right, IteratorStack* diff, int at_dim, int at_pos) {
  int not_accepted, not_accepted_count=0, pos;
  for (int block_id=0; block_id<data->n_blocks; block_id++)
    diff->iters[block_id]->from[at_dim]=left->iters[block_id]->to[at_dim];
  for (int j=0; j<data->dims; j++) {
    diff->from[j]=from[j];
    diff->to[j]=to[j];
  }
  diff->from[at_dim]=left->to[at_dim];

  // For all levels 
  left->empty=1;
  right->empty=1;
  diff->empty=1;
  df=0;
  weighted_df=0;
  for (int i=0; i<data->dims; i++) {
    left->from[i]=from[i];
    left->to[i]=to[i];
    right->from[i]=from[i];
    right->to[i]=to[i];
  }
  left->to[at_dim]=at_pos;
  right->from[at_dim]=at_pos;
  for (int block_id=0; block_id<data->n_blocks; block_id++) {
    // Do the split
    for (int i=0; i<data->dims; i++) {
      left->iters[block_id]->empty=iters[block_id]->empty; right->iters[block_id]->empty=iters[block_id]->empty;
      left->iters[block_id]->from[i]=iters[block_id]->from[i]; right->iters[block_id]->from[i]=iters[block_id]->from[i];
      left->iters[block_id]->to[i]=iters[block_id]->to[i]; right->iters[block_id]->to[i]=iters[block_id]->to[i];
    }
    pos=(int)round((double)at_pos*(double)data->size[block_id]/(double)data->full_size);
    left->iters[block_id]->to[at_dim]=pos;
    right->iters[block_id]->from[at_dim]=pos;
    // Check whether the split is big enough ...
    not_accepted=0;
    if (pos-left->iters[block_id]->from[at_dim]<data->min_width[block_id])
      not_accepted=-1;
    if (right->iters[block_id]->to[at_dim]-pos<data->min_width[block_id])
      not_accepted=1;
    if (not_accepted==0) {
      if (left->iters[block_id]->compute_n()<data->min_size[block_id]) 
	not_accepted=-1;
      if (right->iters[block_id]->compute_n()<data->min_size[block_id]) 
	not_accepted=1;
    }
    // If not ...
    if (not_accepted!=0) {
      not_accepted_count++;
      if (not_accepted==-1) 
	for (int i=0; i<data->dims; i++) {
	  left->iters[block_id]->to[at_dim]=this->iters[block_id]->from[at_dim];
	  left->iters[block_id]->empty=1;
	  right->empty=0;
	  right->iters[block_id]->from[at_dim]=this->iters[block_id]->from[at_dim];
	}
      if (not_accepted==1) 
	for (int i=0; i<data->dims; i++) {
	  left->iters[block_id]->to[at_dim]=this->iters[block_id]->to[at_dim];
	  left->empty=0;
	  right->iters[block_id]->from[at_dim]=this->iters[block_id]->to[at_dim];
	  right->iters[block_id]->empty=1;
	  // CONSIDER right->iters[block_id]=NULL; is however rather tricky
	}
    } else {
      left->empty=0;
      right->empty=0;
      weighted_df+=data->weights[block_id];
      df++;
    }     
    diff->iters[block_id]->empty=0;
    for (int j=0; j<data->dims; j++) {
      if (j==at_dim) {
	diff->iters[block_id]->to[j]=left->iters[block_id]->to[j];

	if (diff->iters[block_id]->to[j]<=diff->iters[block_id]->from[j])
	  diff->iters[block_id]->empty=1;
      } else {
	diff->iters[block_id]->from[j]=left->iters[block_id]->from[j];
	diff->iters[block_id]->to[j]=left->iters[block_id]->to[j];
	if (diff->iters[block_id]->to[j]<=diff->iters[block_id]->from[j])
	  diff->iters[block_id]->empty=1;
      }
    }
    if (!diff->iters[block_id]->empty)
      diff->empty=0;
  }
  diff->to[at_dim]=left->to[at_dim];
  return df;
}


double compute_minimum_w(Iterator* it) {
  double a=options->a;
  double t=sqrt(2*log((double)it->data->n[it->block_id]));
  double t_minus_a=t-a;
  return 1/((a*pnorm(t_minus_a,0,1,1,0))/dnorm(t_minus_a,0,1,0) - beta_laplace(t,options->a));
}

double evaluate_llh(Iterator* it, double w) {
  double one_plus_w_beta,/* beta_by_one_plus_w_beta,*/ loglh=0;
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
  virtual void prepare_criterion(IteratorStack* full_stack, double* current_w, double* current_llh);
  virtual void prepare_search(IteratorStack* left_stack, IteratorStack* right_stack);
  virtual double compute_goodness(IteratorStack* left_stack, IteratorStack* right_stack, IteratorStack* diff_stack)=0;    
  virtual double rescale_criterion(double criterion, int n, int df);
};

void Criterion::prepare_criterion(IteratorStack* full_stack, double* current_w, double* current_llh) {
}

void Criterion::prepare_search(IteratorStack* left_stack, IteratorStack* right_stack) {
}

double Criterion::rescale_criterion(double criterion, int n, int df) {
  return criterion;
}


/////////////////////////////////////////////////////////////////////////////


class ScoreCriterion : public Criterion {
  double *left_sum_score, *right_sum_score, *left_sum_ofisher, *right_sum_ofisher;
  double **scores;
  Data* data;
public:
  ScoreCriterion(Data* data);
  void prepare_criterion(IteratorStack* full_stack, double* current_w, double* current_llh);
  void prepare_search(IteratorStack* left_stack, IteratorStack* right_stack);
  double compute_goodness(IteratorStack* left_stack, IteratorStack* right_stack, IteratorStack* diff_stack);    
  double rescale_criterion(double criterion, int n, int df);
};

ScoreCriterion::ScoreCriterion(Data* data) {
  this->data=data;
  scores=(double**)R_alloc(data->n_blocks,sizeof(double*));
  for (int i=0; i<data->n_blocks; i++) {
    scores[i]=(double*)R_alloc((int)round(R_pow((double)data->size[i],(double)data->dims)),sizeof(double));
  }
  left_sum_score=(double*)R_alloc(data->n_blocks,sizeof(double));
  left_sum_ofisher=(double*)R_alloc(data->n_blocks,sizeof(double));
  right_sum_score=(double*)R_alloc(data->n_blocks,sizeof(double));
  right_sum_ofisher=(double*)R_alloc(data->n_blocks,sizeof(double));
}


void ScoreCriterion::prepare_criterion(IteratorStack* full_stack, double *current_w, double *current_NEG_llh) {
  double beta;
  Iterator* full_partition;
  for (int i=0; i<data->n_blocks; i++) {
    full_partition=full_stack->iters[i];
    for (int pos=full_partition->start(); !full_partition->finished; pos=full_partition->next()) {
      beta=full_stack->data->beta[i][pos];
      if (beta>options->beta_max)
	beta=options->beta_max;
      scores[i][pos]=beta/(1+current_w[i]*beta);    
    }
  }
}

void ScoreCriterion::prepare_search(IteratorStack* left_stack, IteratorStack* right_stack) {
  int pos;
  Iterator *left_partition, *right_partition;
  for (int i=0; i<left_stack->data->n_blocks; i++) {
    left_sum_score[i]=0; 
    right_sum_score[i]=0; 
    left_sum_ofisher[i]=0;
    right_sum_ofisher[i]=0;
    left_partition=left_stack->iters[i];
    right_partition=right_stack->iters[i];
    for (pos=left_partition->start(); !left_partition->finished; pos=left_partition->next()) {
      if (ISNAN(scores[i][pos]))
	continue;
      left_sum_score[i]+=scores[i][pos];
      left_sum_ofisher[i]+=(scores[i][pos]*scores[i][pos]);
    }
    for (pos=right_partition->start(); !right_partition->finished; pos=right_partition->next()) {
      if (ISNAN(scores[i][pos]))
	continue;
      right_sum_score[i]+=scores[i][pos];
      right_sum_ofisher[i]+=(scores[i][pos]*scores[i][pos]);
    }
  }
}

double ScoreCriterion::compute_goodness(IteratorStack* left_stack, IteratorStack* right_stack, IteratorStack* diff_stack) {
  double diff_sum_score, diff_sum_ofisher, sum_crit=0;
  Iterator* diff;
  for (int i=0; i<left_stack->data->n_blocks; i++) {
    diff_sum_score=0;
    diff_sum_ofisher=0;
    if (diff_stack!=NULL) {
      diff=diff_stack->iters[i];
      for (int pos=diff->start(); !diff->finished; pos=diff->next()) {
	if (ISNAN(scores[i][pos]))
	  continue;
	diff_sum_score+=scores[i][pos];
	diff_sum_ofisher+=(scores[i][pos]*scores[i][pos]);
      } 
      left_sum_score[i]+=diff_sum_score;
      right_sum_score[i]-=diff_sum_score;
      left_sum_ofisher[i]+=diff_sum_ofisher;
      right_sum_ofisher[i]-=diff_sum_ofisher;
    } 
    if (!left_stack->iters[i]->empty && !right_stack->iters[i]->empty)
      sum_crit+=data->weights[i]*((left_sum_score[i]*left_sum_score[i])/left_sum_ofisher[i] + (right_sum_score[i]*right_sum_score[i])/right_sum_ofisher[i]);
  }
  return sum_crit;
}

double ScoreCriterion::rescale_criterion(double criterion, int n, int df) {
  double crit_value=qchisq(1-(1-options->rescale_quantile)/(double)n,df,1,0);
  return (criterion-crit_value)/crit_value;
}

class TreeNode: public RGarbageCollectedObject {
public:
  TreeNode *left_child, *right_child, *parent;
  double criterion, loglikelihood, *loglikelihoods, *w, alpha, cumulative_loglikelihood;
  int pruning_size, dim, pos, id, df;
  int* detail_pos;
  IteratorStack* its;
  TreeNode(TreeNode* parent, IteratorStack* its);
  void split(Criterion* sc, int depth);
  int compute_pruning_size();
  double compute_cumulative_loglikelihood();
  double search_best_prune(TreeNode** best);
  void prune(double alpha);
  void carry_out_pruning();
  void print(int depth);
  void issue_id();
  void issue_id(int* cur_id);
  void membership(int* membership, int block_id);
  int num_nodes();
  void list_splits(double* matrix, int* details, double* ws, int nrow);
  void list_splits(double* matrix, int* details, double* ws, int nrow, int* counter);
};

TreeNode::TreeNode(TreeNode* parent, IteratorStack* its) {
  this->parent=parent;
  left_child=NULL;
  right_child=NULL;
  dim=-1;
  this->its=its;
  alpha=R_NegInf;
  w=(double*)R_alloc(its->data->n_blocks,sizeof(double));
  detail_pos=(int*)R_alloc(its->data->n_blocks,sizeof(int));
  loglikelihoods=(double*)R_alloc(its->data->n_blocks,sizeof(double));
  if (parent==NULL) {
    double sum_loglh;
    sum_loglh=0;
    for (int i=0; i<its->data->n_blocks; i++) {
      w[i]=compute_mle_w(its->iters[i],0.5,options->first_step,options->max_iter,&loglikelihoods[i]);
      sum_loglh+=loglikelihoods[i];
    }
    loglikelihood=sum_loglh;
  }
}

int TreeNode::compute_pruning_size() {
  if ((left_child!=NULL) && (right_child!=NULL) && (alpha==R_NegInf))
    pruning_size=left_child->compute_pruning_size()+right_child->compute_pruning_size();
  else
    pruning_size=its->get_number_of_active_partitions();
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
    cumulative_loglikelihood=loglikelihood;
  return cumulative_loglikelihood;  
}

double TreeNode::search_best_prune(TreeNode** best) {
  if ((left_child!=NULL) && (right_child!=NULL) && (pruning_size>1)) {
    TreeNode *best_left, *best_right;
    double crit_left=left_child->search_best_prune(&best_left);
    double crit_right=right_child->search_best_prune(&best_right);
    double crit=(cumulative_loglikelihood-loglikelihood)/(double)(pruning_size-its->get_number_of_active_partitions());
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
  while (this->alpha==R_NegInf) 
    {
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
  // If the node is already empty there is nothing to do ...
  if (its->empty)
    return;
  if (depth>options->max_depth)
    return;
  // A first vmaxget
  void* mempos_1=vmaxget();
  IteratorStack* left=new IteratorStack(its->data);
  IteratorStack* right=new IteratorStack(its->data);
  // A second vmaxget
  void* mempos_2=vmaxget();
  IteratorStack* diff=new IteratorStack(its->data);
  double crit, best_crit=R_NegInf;
  int best_dim=-1, best_pos=-1, df, /* best_df=1,*/ accepted=0, first_time;
  R_CheckUserInterrupt();
  sc->prepare_criterion(its,w,loglikelihoods);
  for (int dim=0; dim<its->data->dims; dim++) {
    first_time=1;
    for (int i=its->from[dim]+1; i<its->to[dim]-1; i++) {
      R_CheckUserInterrupt();
      df=its->split(left,right,diff,dim,i);
      if (first_time) {
	sc->prepare_search(left,right);
	crit=sc->compute_goodness(left,right,(IteratorStack*)NULL);
	first_time=0;
      } else {
	crit=sc->rescale_criterion(sc->compute_goodness(left,right,diff),1,df);
      }
      if (ISNAN(crit))
	crit=R_NegInf;
     if ((crit>best_crit) & !left->empty & !right->empty) {
	accepted=1;
	best_crit=crit;
	best_dim=dim;
	//	best_df=df;
	best_pos=i;
      }
    }
  }
  criterion=best_crit;
  if (parent!=NULL)
    if ((best_crit-parent->criterion<=parent->criterion*options->relative_improvement) && (best_crit-parent->criterion<=options->absolute_improvement))
      accepted=0;
  if (best_crit<=options->absolute_criterion)
    accepted=0;
  if (accepted && (depth<options->max_depth)) {
    // vmaxset to second value
    vmaxset(mempos_2);
    df=its->split(left,right,diff,best_dim,best_pos);
   // A third vmaxget
    void* mempos_3=vmaxget();
    left_child=new TreeNode(this,left);
    right_child=new TreeNode(this,right);
    double sum_loglh=0;
    for (int i=0; i<its->data->n_blocks; i++) {
      left_child->w[i]=compute_mle_w(left->iters[i],0.5,options->first_step,options->max_iter,&left_child->loglikelihoods[i]);
      sum_loglh+=left_child->loglikelihoods[i];
    }
    left_child->loglikelihood=sum_loglh;
    sum_loglh=0;
    for (int i=0; i<its->data->n_blocks; i++) {
      right_child->w[i]=compute_mle_w(right->iters[i],0.5,options->first_step,options->max_iter,&right_child->loglikelihoods[i]);
      sum_loglh+=right_child->loglikelihoods[i];
    }
    right_child->loglikelihood=sum_loglh;
    // Carry out a likelihood ratio test ...
    if (((options->lr_signif<=0) || (left_child->loglikelihood+right_child->loglikelihood-loglikelihood>=0.5*qchisq(options->lr_signif,df,1,0))) && (left_child->loglikelihood+right_child->loglikelihood-loglikelihood>1e-10)) {
      this->dim=best_dim;
      this->pos=best_pos;
      this->df=df;
      for (int i=0; i<its->data->n_blocks; i++) {
	if (left->iters[i]->empty)
	  this->detail_pos[i]=its->iters[i]->from[best_dim];
	else if (right->iters[i]->empty)
	  this->detail_pos[i]=its->iters[i]->to[best_dim];
	else this->detail_pos[i]=left->iters[i]->to[best_dim];
      }  
      left_child->split(sc,depth+1);
      right_child->split(sc,depth+1);
    } else {
      left_child=NULL;
      right_child=NULL;
      // vmaxset to third value
      vmaxset(mempos_3);
    }
  } else {
    // vmaxset to first value
    vmaxset(mempos_1);
  }
}

void TreeNode::membership(int* membership, int block_id) {
  if ((left_child!=NULL) && (right_child!=NULL)) {
     left_child->membership(membership,block_id);
     right_child->membership(membership,block_id);
  } else {
	    for (pos=its->iters[block_id]->start(); !its->iters[block_id]->finished; pos=its->iters[block_id]->next()) {
	      membership[pos]=id+1;
	    }
  }
}


void TreeNode::list_splits(double* matrix, int* details, double* ws, int nrow) {
  int counter=0;
  list_splits(matrix,details,ws, nrow, &counter);
}

void TreeNode::list_splits(double* matrix, int* details, double *ws, int nrow, int* counter) {
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
    for (int i=0; i<its->data->n_blocks; i++) 
      details[*counter+i*nrow]=detail_pos[i];
    matrix[*counter+2*nrow]=dim+1;
    matrix[*counter+3*nrow]=pos;
    matrix[*counter+4*nrow]=left_child->id+1;
    matrix[*counter+5*nrow]=right_child->id+1;
    matrix[*counter+6*nrow]=criterion;
  } else {
    for (int i=2; i<=6; i++)
      matrix[*counter+i*nrow]=NA_REAL;
    for (int i=0; i<its->data->n_blocks; i++) 
      details[*counter+i*nrow]=NA_INTEGER;
  }
  for (int i=0; i<its->data->n_blocks; i++) 
    ws[*counter+i*nrow]=w[i];
  matrix[*counter+7*nrow]=-loglikelihood;
  if (parent!=NULL)
    matrix[*counter+8*nrow]=parent->alpha;
  else
    matrix[*counter+8*nrow]=NA_REAL;
  (*counter)++;
  if ((left_child!=NULL) & (right_child!=NULL)) {
    left_child->list_splits(matrix,details,ws,nrow,counter);
    right_child->list_splits(matrix,details,ws,nrow,counter);
  }
}

}

//////////////////////////////////////////////////////////////////////////////



extern "C" {
  SEXP fit_tree_wave(SEXP args) {
    using namespace blockthresh;
    SEXP result_sxp, list_sexp, element_sexp, matrix_sexp, details_sexp, ws_sexp;    
    options=new Options(extractElement(args,"control"));
    Data* data=new Data(args);
    IteratorStack* its=create_full_iterator_stack(data);
    Criterion* sc=new ScoreCriterion(data);
    TreeNode* root=new TreeNode(NULL,its);
    root->split(sc,1);
    root->issue_id();
    root->carry_out_pruning();
    PROTECT(result_sxp=allocVector(VECSXP,6));
    int nrow=root->num_nodes();
    PROTECT(matrix_sexp=allocMatrix(REALSXP,nrow,10));
    SET_VECTOR_ELT(result_sxp,0,matrix_sexp);
    PROTECT(details_sexp=allocMatrix(INTSXP,nrow,data->n_blocks));
    SET_VECTOR_ELT(result_sxp,1,details_sexp);
    PROTECT(ws_sexp=allocMatrix(REALSXP,nrow,data->n_blocks));
    SET_VECTOR_ELT(result_sxp,2,ws_sexp);
    root->list_splits(REAL(matrix_sexp),INTEGER(details_sexp),REAL(ws_sexp),nrow);
    if (!data->use_beta) {
      SET_VECTOR_ELT(result_sxp,5,data->beta_list_sexp);
    }
    PROTECT(list_sexp=allocVector(VECSXP,data->n_blocks));
    for (int i=0; i<data->n_blocks; i++) {
      PROTECT(element_sexp=allocVector(INTSXP,data->n[i]));
      root->membership(INTEGER(element_sexp),i);
      SET_VECTOR_ELT(list_sexp,i,element_sexp);
    }
    SET_VECTOR_ELT(result_sxp,4,list_sexp);
 
    UNPROTECT(5+data->n_blocks+!data->use_beta);

    return result_sxp;
  }
}

extern "C" {

  void prune_tree(double* matrix, int* nrow, int* kill, int* leaf, int* membership, double* C) {
    for (int row=*nrow-1; row>0; row--) {
      double cur_C=matrix[row+*nrow*9];
      int id=(int)round(fabs(matrix[row+*nrow*0]));
      membership[row]=id;
      if (cur_C<*C) { // Prune me!
        kill[row]=1;
	int parent_id=(int)round(fabs(matrix[row+*nrow*1]));
	for (int i=0; i<row; i++)
	  if (membership[i]==parent_id)
	    leaf[i]=1;
	for (int i=row; i<*nrow; i++)
	  if (membership[i]==id)
	    membership[i]=parent_id;
      } else {
	kill[row]=0;
      }
    }
  }


  void update_membership(int* old_membership, int* new_membership, int* n, int* old_types, int* new_types, int* n_types) {
    for (int i=0; i<*n; i++)
      new_membership[i]=old_membership[i];
    for (int j=0; j<*n_types; j++)
      if (old_types[j]!=new_types[j]) {
	for (int i=0; i<*n; i++)
	  if (old_membership[i]==old_types[j])
	    new_membership[i]=new_types[j];
      }
  }
}

