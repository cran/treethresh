\name{estimate.sdev}
\alias{estimate.sdev}
\alias{estimate.sdev.wd}
\alias{estimate.sdev.imwd}
\keyword{tree}
\keyword{nonparametric}
\title{Estimate noise level from wavelet coefficients}
\description{
\code{estimate.sdev} estimates by analysing the the wavelet coefficients
at the finest level(s) of an \code{\link[wavethresh]{wd}} or
\code{\link[wavethresh]{imwd}} object.
}

\usage{
\method{estimate.sdev}{wd}(object, dev=mad)
\method{estimate.sdev}{imwd}(object, dev=mad)
}
\arguments{
  \item{object}{For \code{extract.coefficients} the
    \link[wavethresh]{wd} or \link[wavethresh]{imwd} object for which to
    estimate the standard error.}
  \item{dev}{The function used to estimate the standard error. Typical
    examples are \code{\link[stats]{sd}}, or the more
    robust \code{\link[stats]{mad}} (default).}
\item{...}{additional arguments (see above for supported arguments).}
}
\value{
  \code{estimate.sdev} returns an estimate of the standard error.
}
\note{
For an example of the use of \code{estimate.sdev}, see \code{\link{coefficients}}.
}
