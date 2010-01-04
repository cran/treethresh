\name{wavelet.treethresh}
\alias{wavelet.treethresh}
\keyword{tree}
\keyword{nonparametric}
\title{Threshold wavelet coefficients}
\description{
  \code{wavelet.threshold} is a more user-friendly function for
  thresholding wavelet coefficients stored in an \code{\link[wavethresh]{wd}} or \code{\link[wavethresh]{imwd}} object. It
  combines the functions \code{\link{extract.coefficients}},
  \code{\link{estimate.sdev}}  (rescales the coefficients accordingly),
  \code{\link{treethresh}} or \code{\link{wtthresh}},
  \code{\link{prune}}, \code{\link{thresh}}, and
  \code{\link{insert.coefficients}}
}
\usage{
wavelet.treethresh(object, sdev=NA, dev=mad, start.level=5,
                   levelwise=FALSE, v=5, sd.mult=0.5, postmed=TRUE, ...) 
}
\arguments{
  \item{object}{An object of the class \code{\link[wavethresh]{wd}} or
    \code{\link{wtthresh}} to be smoothed by thresholding.
  }
  \item{sdev}{The standard error of the noise (if known), otherwise
\code{NA}.}
  \item{dev}{The function used to estimate the standard error. Typical
    examples are \code{\link[stats]{sd}} (teh default), or the more
    robust \code{\link[stats]{mad}}. Not used if \code{sdev} is given.}
  \item{start.level}{The level in the wavelet transform from which to commence thresholding the wavelet
  coefficients. This level and all those finer levels will be thresholded.}
  \item{levelwise}{Indicates whether the thresholding should be carried
out independently for each level and type of coefficients as in the
function \code{\link{treethresh}} (if \code{levelwise=TRUE}), or whether
a common partition is to be found jointly for all coefficients as
in the function \code{\link{wtthresh}} (if \code{levelwise=FALSE}).}
  \item{v}{The number of folds in the cross-validation used to determine
the optimal subtree in the pruning step (see the function
\code{\link{prune}} for details).  \code{NA} if no pruning is to be carried out.}
  \item{sd.mult}{The smallest subtree that is not \code{sd.mult} times
the standard error worse than the best loglikelihood will be chosen as
the optimal tree in the pruning step. (see the function
\code{\link{prune}} for details).}
  \item{postmed}{Controls whether thresholding is to be carried out by
using the posterior median of the coefficient \eqn{\mu|y}{mu|y}
(\code{postmed=true}) or by using a hard threshold
(\code{postmed=FALSE}). (See the function \code{\link{thresh}} for more details.)}
  \item{...}{arguments passed to to \code{\link{wtthresh}} (if
\code{levelwise=FALSE}) or \code{\link{treethresh}} (if
\code{levelwise=TRUE}).}
}
\value{
Returns an object of the class \code{\link[wavethresh]{wd}} or
    \code{\link{wtthresh}}, where the coefficients have
    been thresholded using the TreeThresh algorithm.
}
\note{\code{wavelet.treethresh} combines the functions \code{\link{extract.coefficients}},
  \code{\link{estimate.sdev}} (and the appropriate scaling), \code{\link{treethresh}} /
  \code{\link{wtthresh}}, \code{\link{prune}},
  \code{\link{insert.coefficients}} into a single (hopefully) more
  user-friendly function.
}  
\examples{
## The following examples shows how an example image can be
## thresholded step by step. All the steps are combined in the more
## user-friendly function wavelet.treethresh

## (01) Load the example image
data(tiles)

## (02) Display the image
par(mai=rep(0,4)) ; image(tiles,col=grey(0:255/255))

## (03) Add noise to the image
corrupted <- tiles + rnorm(length(tiles))

## (04) Display the corrupted image
par(mai=rep(0,4)) ; image(corrupted,col=grey(0:255/255))

## (05) Compute the wavelet transform
corrupted.wt <- imwd(corrupted)

save.image(file="~/holygrail.RData")

## (06) Perform the thresholding
denoised.wt <- wavelet.treethresh(corrupted.wt)

## (07) Compute inverse wavelet transform
denoised <- imwr(denoised.wt)

## (08) Display denoised image
par(mai=rep(0,4)) ; image(denoised,col=grey(0:255/255))

## (09) Compute l2 loss
sum((denoised-tiles)^2)

## The call to wavelet.treethresh is equivalent to steps (06) to (13)
## of the example in the help section "coefficients".
}
\seealso{
  \code{\link{extract.coefficients}}, \code{\link{estimate.sdev}},
  \code{\link{treethresh}}, \code{\link{wtthresh}}, \code{\link{prune}},
  \code{\link{thresh}}, \code{\link{insert.coefficients}}
}

