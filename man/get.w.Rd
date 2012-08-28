\name{get.w}
\alias{get.w}
\alias{get.w.treethresh}
\alias{get.w.wtthresh}
\keyword{tree}
\keyword{nonparametric}
\title{Extract estimated weights}
\description{
  Extracts from a tree object of the classes
  \code{\link{treethresh}} or \code{\link{wtthresh}} the estimated value
  of the weight \eqn{w} for each data point.
}
\usage{
\method{get.w}{treethresh}(object, C)
\method{get.w}{wtthresh}(object, C)
}
\arguments{
  \item{object}{An object of the class \code{\link{treethresh}} or
    \code{\link{wtthresh}} from which the weights are to be extracted.
  }
  \item{C}{If \code{C} is given then the tree is pruned with constant
    \eqn{C}{C} (see the help of \code{\link{prune}} for details on the
    pruning).}
}  
\value{
    \code{get.w} returns an estimated value of the weight \eqn{w} for
    each data point used to estimate \code{object}. If \code{object} is
    of the class
    \code{\link{treethresh}}, then \code{get.w} returns an array (or
    matrix) of exactly the same dimension as the data that was used to
    estimate \code{object}. If \code{object} is of the class
    \code{\link{wtthresh}} then it returns a list with elements of the
    same name and dimensions as the list that was used to estimate
    \code{object}.
  }

\seealso{
\code{\link{treethresh}}, \code{\link{wtthresh}}, \code{\link{get.t}}, \code{\link{prune}}
}

