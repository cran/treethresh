\name{get.t}
\alias{get.t}
\alias{get.t.treethresh}
\alias{get.t.wtthresh}
\keyword{tree}
\keyword{nonparametric}
\title{Extract estimated hard threshold}
\description{
  Extracts from a tree object of the classes
  \code{\link{treethresh}} or \code{\link{wtthresh}} the estimated value
  of the hard threshold \eqn{t} for each data point.
}
\usage{
get.t(object, C)
get.t.treethresh(object, C)
get.t.wtthresh(object, C)
}
\arguments{
  \item{object}{An object of the class \code{\link{treethresh}} or
    \code{\link{wtthresh}} from which the thresholds are to be computed.
  }
  \item{C}{If \code{C} is given, then the tree is pruned with constant
    \eqn{C}{C} (see the help of \code{\link{prune}} for details on the
    pruning process).}
}  
\value{
    \code{get.t} returns an estimated value of the hard threshold \eqn{t} for
    each data point used to estimate \code{object}. If \code{object} is
    of the class
    \code{\link{treethresh}}, then \code{get.t} returns an array (or
    matrix) of exactly the same dimension as the data that was used to
    estimate \code{object}. If \code{object} is of the class
    \code{\link{wtthresh}} then it returns a list with elements of the
    same name and dimensions as the list that was used to estimate
    \code{object}.
  }
\note{
\code{get.t.treethresh} and \code{get.t.wtthresh}
should rarely be directly called by the user. The more user-friendly S3 function
\code{get.t} will take care of calling the right function.
}

\seealso{
\code{\link{treethresh}}, \code{\link{wtthresh}}, \code{\link{get.w}}, \code{\link{prune}}
}

