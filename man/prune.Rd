\name{prune}
\alias{prune}
\alias{prune.treethresh}
\alias{prune.wtthresh}
\keyword{tree}
\keyword{nonparametric}
\title{Prune a tree using cross-validation}
\description{
  Extracts an optimal subtree from a tree object of the classes
  \code{\link{treethresh}} or \code{\link{wtthresh}}. Contrary to
  \code{\link{subtree}} the values of the complexity parameter \eqn{C} does
  not need to be given, but is determined using cross-validation.
}
\usage{
prune(object, v=5, sd.mult=0.5, plot=TRUE)
prune.treethresh(object, v=5, sd.mult=0.5, plot=TRUE)
prune.wtthresh(object, v=5, sd.mult=0.5, plot=TRUE)
}
\arguments{
  \item{object}{An object of the class \code{\link{treethresh}} or
    \code{\link{wtthresh}} according to which thresholding is to be
    carried out.}
  \item{v}{The number of folds in the cross-validation used to determine
    the optimal subtree in the pruning step (see below for details).}
  \item{sd.mult}{The smallest subtree that is not \code{sd.mult} times
the standard error worse than the best loglikelihood will be chosen as
the optimal tree in the pruning step. (see below for details).}
\item{plot}{If \code{plot=TRUE} a plot of the relative predicted
  loglikelihood estimated in the cross-validation against the complexity
  parameter \eqn{C} is produced.}
\item{...}{additional arguments (see above for supported arguments).}
}  
\value{
    \code{prune} returns an object of the class
    \code{\link{treethresh}} or \code{\link{wtthresh}} that contains a
    tree pruned at value \eqn{C} (see the function \code{\link{prune}}
    for details on the pruning process).
}
\details{
The tree grown by \code{\link{treethresh}} or \code{\link{wtthresh}}
often yields too many partitions leading to an overfit. The resulting
tree has to be 'pruned', i.e. the branches corresponding to the least
important regions have to be 'snipped off'. 

As the TreeThresh model is a special case of a classification and
regression tree, there exists a sequence of nested subtrees (i.e. a
sequence of nested partitions) that maximises the regularised
loglikelihood
\deqn{\ell + \alpha \cdot \textrm{\# partitions}.}{l + alpha * #partitions.}

The parameter \eqn{\alpha}{alpha} controls the complexity of the
resulting partition. For \eqn{\alpha=0}{alpha=0} no pruning is carried
out. If a large enough \eqn{\alpha}{alpha} is chosen, only the root node
of the tree is retained, i.e. no partitioning is done. Denote this value
of \eqn{\alpha}{alpha} by \eqn{\\alpha_0}{alpha_0}. The complexity
parameter can thus be rescaled to
\deqn{C=\frac{\alpha}{\alpha_0}}{C = alpha / alpha_0}
yielding a complexity parameter ranging from \eqn{0} (no pruning) to
\eqn{1} (only retain the root node).

The optimal value of the complexity parameter \eqn{C} (or, equivalently, \eqn{\alpha}{alpha})
depends on the problem at hand and thus has to be chosen
carefully. \code{prune} estimates the optimal complexity parameter
\eqn{C} by a \eqn{v}-fold cross-validation. If \code{sd.mult=0} the
value of \eqn{C} that yields the highest predictive loglikelihood in the
cross-validation is used to prune the tree \code{object}. If
\code{sd.mult} is not \eqn{0} the largest \eqn{C} that is not
\code{sd.mult} standard errors worse than the best \eqn{C} is used.
}
\note{
\code{prune.treethresh} and \code{prune.wtthresh}
should rarely be directly called by the user. The more user-friendly S3 function
\code{prune} will take care of calling the right function.

For an example of the use of \code{prune}, see \code{\link{coefficients}}.}

\seealso{
\code{\link{treethresh}}, \code{\link{wtthresh}}, \code{\link{get.t}}, \code{\link{prune}}
}

