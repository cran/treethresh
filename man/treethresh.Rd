\name{treethresh}
\alias{treethresh}
\keyword{tree}
\keyword{nonparametric}
\title{Compute optimal thresholding partition for a sequence, matrix, or
  array of data.}
\description{
This function carries out the tree-based thresholding algorithm described in
section 3 of Evers and Heaton (2009).
}
\usage{
treethresh(data, beta, criterion="score", control=list(),
           rho=sys.frame(sys.parent()))
}
\arguments{
  \item{data}{An array (or an object coercible to an array, i.e. a
    vector or matrix) containing the data. The
data is assumed to have noise of unit variance, thus the data needs to
be rescaled a priori (e.g. in the case of wavelet coefficients using function \code{\link{estimate.sdev}}).
}
\item{beta}{Instead of using the original \code{data}, one can call
\code{wtthresh} using the \eqn{\beta_i}{beta_i} instead of the observed
data. These can be computed using \code{\link[EbayesThresh]{beta.laplace}}.
}
\item{criterion}{The criterion to be used. Can be \code{"score"}
  (default) for using the score test, \code{"likelihood"} for using the
  likelihood ratio test (slower), \code{"heuristic"} for using a
  heuristic criterion based on the original data, or a user-specified
  function that computes the goodness of a split. This function should
  take four arguments (which should be self-explanatory),
  \code{left_data}, \code{left_betas}, \code{right_data}, and \code{right_betas}. }
\item{control}{A list that allows the user to tweak the behaviour of
\code{treethresh}. It can contain the following elements:
\describe{
\item{max.depth}{The maximum depth of the tree. Defaults to \code{10}.}
\item{minimum.width}{The minimum width of a region of the partitions. This setting avoids creating too small regions. Defaults to \code{3}.}
\item{minimum.size}{The minimum size of a region of the partitions. This setting avoids creating too small  regions. Defaults to \eqn{5^d}, where \eqn{d} is the dimension of the arras.}
\item{lr.signif}{If the p-value of the corresponding likelihood
  ratio test is larger than \code{1-lr.signif} a split will be
  discarded. Defaults to \code{0.5}.}
\item{absolute.improvement}{The minimum absolute improvement of the
  above criterion necessary such that a split is retained. Defaults to \code{-Inf}, i.e. deactivated.}
\item{relative.improvement}{The minimum relative improvement of the
  above criterion necessary such that a split is retained. Defaults to
  \code{-Inf}, i.e. deactivated.}
\item{absolute.criterion}{The minimum value of the above criterion necessary such that a split is retained. Defaults to
  \code{0}, i.e. deactivated.}
\item{a}{The parameter \eqn{a} of the Laplace distribution
\eqn{\gamma(\mu)\propto \exp(-a|\mu|) }{gamma(mu) = const * exp(-a*mu)}
corresponding to the signal. Defaults to \code{0.5}.}
\item{beta.max}{The maximum value of \eqn{\beta}{beta}. Defaults to \code{1e5}.}
\item{max.iter}{The maximum number of iterations when computing the
  estimate of the weight \eqn{w} in a certain region. Defaults to \code{30}. }
\item{tolerance.grad}{The estimate of the weight \eqn{w} in a certain
  region is considered having converged, if the gradient of the
  likelihood is less than \code{tolerance.grad}. Defaults to \code{1e-8}.}
\item{tolerance}{The estimate of the weight \eqn{w} in a certain
  region is considered having converged, if the estimates of the weight
  \eqn{w} change less than \code{tolerance}. Defaults to \code{1e-6}.}
}}
\item{rho}{The environment used to evaluate the user-speficied criterion
  function if one is supplied). (You want to change
this argument only in very rare circumstances).}
}
\value{
  \code{treethresh} returns an object of the class \code{c("treethresh")},
  which is a list containing the following elements:
  \item{splits}{A table describing the detailed structure of the fitted tree
  together with the local loglikelihoods required for the pruning.}
  \item{membership}{An array of the same dimension as \code{data} or
    \code{beta} indicating to which region each entry of the array of data belongs.}
  \item{beta}{The values of \eqn{\beta}{beta} for each observation / coefficient.}  
  \item{data}{The data used.}
  \item{criterion}{The criterion used to decide on splits (see argument \code{criterion}).}
  \item{control}{The control list of tuning options used (see argument \code{control}).}
}
\references{
Evers, L. and Heaton T. (2009) Locally adaptive tree-based thresholding.
}
\examples{
# (1) Create a vector with the probabilities of a signal being present
w.true <- c(rep(0.1,400),rep(0.7,300),rep(0.1,300))

# (2) Generate the signal
mu <- numeric(length(w.true))
non.zero.entry <- runif(length(mu))<w.true
num.non.zero.entries <- sum(non.zero.entry)
mu[non.zero.entry] <- rexp(num.non.zero.entries,rate=0.5)*sample(c(-1,1),num.non.zero.entries,replace=TRUE)

# (3) Split graphics device
par(mfrow=c(2,2))

# (3) Draw the true signal (signal present in red)
plot(mu,col=non.zero.entry+1)
title("True signal")

# (4) Add noise to the signal
x <- mu + rnorm(length(mu))

# (5) Plot the noisy signal (signal present in red)
plot(x,col=non.zero.entry+1)
title("Noisy signal")

# (6) Carry out the tree-based thresholding
tt <- treethresh(x)

# (7) Prune the tree
tt.pruned <- prune(tt)

# (8) Threshold the signal according to the pruned tree
mu.hat <- thresh(tt.pruned)

# (9) Print the denoised signal
plot(mu.hat,col=non.zero.entry+1)
title("Denoised signal")

# (10) Add solid lines for splits (lines removed by the pruing are dashed)
abline(v=tt$split[,"pos"],lty=2)
abline(v=tt.pruned$split[,"pos"],lty=1)
}
