\name{subtree}
\alias{subtree}
\alias{subtree.treethresh}
\alias{subtree.wtthresh}
\keyword{tree}
\keyword{nonparametric}
\title{Extract subtree by pruning according to a specified complexity parameter}
\description{
  Extracts a subtree from a tree object of the classes
  \code{\link{treethresh}} or \code{\link{wtthresh}} by pruning
  according to a specified value of the complexity parameter \eqn{C}.
}
\usage{
\method{subtree}{treethresh}(object, C)
\method{subtree}{wtthresh}(object, C)
}
\arguments{
  \item{object}{An object of the class \code{\link{treethresh}} or
    \code{\link{wtthresh}} from which a subtree is to be extracted.
  }
  \item{C}{The value of the complexity parameter \eqn{C} to be used for the pruning.}
}  
\value{
    \code{subtree} returns an object of the class
    \code{\link{treethresh}} or \code{\link{wtthresh}} that contains a
    tree pruned at value \eqn{C} (see the function \code{\link{prune}}
    for details on the pruning process).
  }
\note{
Use the function \code{\link{prune}} to carry out the pruning without
having to specify the complexity parameter \eqn{C}.
}

\seealso{
\code{\link{treethresh}}, \code{\link{wtthresh}}, \code{\link{prune}}
}

