\name{thresh}
\alias{thresh}
\alias{thresh.treethresh}
\alias{thresh.wtthresh}
\keyword{tree}
\keyword{nonparametric}
\title{Perform thresholding according to a tree}
\description{
  Thresholds according to a tree object of the classes
  \code{\link{treethresh}} or \code{\link{wtthresh}}.}
\usage{
\method{thresh}{treethresh}(object, data, C, postmed=TRUE)
\method{thresh}{wtthresh}(object, data, C, postmed=TRUE)
}
\arguments{
  \item{object}{An object of the class \code{\link{treethresh}} or
    \code{\link{wtthresh}} according to which thresholding is to be
    carried out.}
  \item{data}{The data to be thresholded. If \code{object} is of the class
    \code{\link{treethresh}}, then \code{object} has to be an array (or
    matrix) of exactly the same dimension as the data that was used to
    estimate \code{object}. If \code{object} is of the class
    \code{\link{wtthresh}} it has to be a list with elements of the
    same name and dimensions as the list that was used to estimate
    \code{object}. If no data object is given and if \code{object} is not
    estimated directly from the \eqn{\beta}{beta} then the data used to
    estimate \code{object} is used.}
  \item{C}{If \code{C} is given then the tree is pruned with constant
    \eqn{C}{C} (see the help of \code{\link{prune}} for details on the
    pruning).}
  \item{postmed}{If \code{postmed=TRUE} then the thresholding is done by
    returning the posterior median \eqn{\mu_i|X_i}{mu_i|X_i}. If
    \code{postmed=FALSE} then hard thresholding is performed.}
}  
\value{
  \code{thresh} returns the thresholded \code{data}.
}
\note{
For an example of the use of \code{thresh}, see
\code{\link{coefficients}}.
}

\seealso{
\code{\link{treethresh}}, \code{\link{wtthresh}}, \code{\link{get.t}}, \code{\link{prune}}
}

