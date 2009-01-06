\name{coefficients}
\alias{extract.coefficients}
\alias{insert.coefficients}
\alias{extract.coefficients.wd}
\alias{insert.coefficients.wd}
\alias{extract.coefficients.imwd}
\alias{insert.coefficients.imwd}
\keyword{utilities}
\title{Extracting and updating coefficients stored in wd or imwd objects}
\description{
\code{extract.coefficients} extracts wavelet coefficient vectors (in case of
\code{\link[wavethresh]{wd}}) and coefficient matrices (in case of
\code{\link[wavethresh]{imwd}}), so that these can be thresholded by
\code{\link{treethresh}} or
\code{\link{wtthresh}}. \code{update.coefficients} re-inserts these
vector or matrices into the \code{\link[wavethresh]{wd}} or \code{\link[wavethresh]{imwd}}
objects, such that the inverse transform can be computed using the
thresholded coefficients.
}

\usage{
extract.coefficients(object, start.level=5)
insert.coefficients(object, update)
extract.coefficients.wd(object, start.level=5)
insert.coefficients.wd(object, update)
extract.coefficients.imwd(object, start.level=5)
insert.coefficients.imwd(object, update)
}
\arguments{
  \item{object}{For \code{extract.coefficients} the
    \link[wavethresh]{wd} or \link[wavethresh]{imwd} object to extract the coefficients
    from. For \code{insert.coefficients} the
    \link[wavethresh]{wd} or \link[wavethresh]{imwd} object to be updated.}
  \item{start.level}{The coarsest level of coefficients to be extracted
    (\code{extract.coefficients} only)}
  \item{update}{A list with the matrices that should be copied into the
    \link[wavethresh]{wd} or \link[wavethresh]{imwd} object. (\code{update.coefficients} only)}
\item{...}{additional arguments (see above for supported arguments).}
}
\value{
  \code{extract.coefficients} returns the coefficient matrices to
  be extracted. \code{update.coefficients} returns the updated
  \code{\link[wavethresh]{wd}} or \code{\link[wavethresh]{imwd}} object.

  \code{insert.coefficients} returns the updated
  \code{\link[wavethresh]{wd}} or \code{\link[wavethresh]{imwd}} object
  into which the coefficients from \code{update} have been inserted.
  
}
\note{
\code{extract.coefficients.wd} and \code{extract.coefficients.imwd}
should rarely be directly called by the user. The more user-friendly S3 function
\code{extract.coefficients} will take care of calling the right
function (idem for \code{insert.coefficients}).
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
par(mai=rep(0,4)) ; image(corrupted, col=grey(0:255/255))

## (05) Compute the wavelet transform
corrupted.wt <- imwd(corrupted)

## (06) Estimate the standard error
dev <- estimate.sdev(corrupted.wt)

## (07) Extract the coefficient matrices to be thresholded
coefs <- extract.coefficients(corrupted.wt)

## (08) Rescale the coefficients using the estimated standard error
## (should be around 1)
for (nm in names(coefs))
  coefs[[nm]] <- coefs[[nm]] / dev

## (09) Compute the tree
coefs.tree <- wtthresh(coefs)

## (10) Prune the tree
coefs.pruned.tree <- prune.wtthresh(coefs.tree)

## (11) Threshold according to the pruned tree
coefs.threshed <- thresh(coefs.pruned.tree)

## (12) Undo the rescaling
for (nm in names(coefs))
  coefs.threshed[[nm]] <- coefs.threshed[[nm]] * dev

## (13) Update coefficients
denoised.wt <- insert.coefficients(corrupted.wt, coefs.threshed)

## (14) Compute inverse wavelet transform
denoised <- imwr(denoised.wt)

## (15) Display denoised image
par(mai=rep(0,4)) ; image(denoised, col=grey(0:255/255))

## (16) Compute l2 loss
sum((denoised-tiles)^2)

## Equivalently we could have called
## denoised.wt <- wavelet.treethresh(corrupted.wt)
## instead of steps (06) - (13)
}

\seealso{
\code{\link{treethresh}}, \code{\link{wtthresh}}, \code{\link{wavelet.treethresh}}
}

