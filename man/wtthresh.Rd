\name{wtthresh}
\alias{wtthresh}
\keyword{tree}
\keyword{nonparametric}
\title{Compute optimal thresholding partition for a sequence of linked arrays}
\description{
This function carries out the joint thresholding algorithm described in
section 5 of Evers and Heaton (2009). Though the function, in principle,
can work any sequence of arrays, it is designed to work with blocks of
wavelet coefficients. These can be extracted from an
\code{\link[wavethresh]{wd}} or \code{\link[wavethresh]{imwd}} object
  using the function \code{extract.coefficients}.
  }
\usage{
wtthresh(data, beta, weights, control=list())
}
\arguments{
\item{data}{A list containing the arrays to be thresholded. The elements
of the array have to be arrays of the same number of dimensions. The
arrays can be of different sizes, however the ratio of their side
lengths has to be the same. Can be a list of wavelet coefficients
extracted using \code{\link[=coefficients]{extract.coefficients}}. The
data is assumed to have noise of unit variance, thus the data needs to
be rescaled a priori (e.g. in the case of wavelet coefficients using function \code{\link{estimate.sdev}}).
}
\item{beta}{Instead of using the original \code{data}, one can call
\code{wtthresh} using the \eqn{\beta_i}{beta_i} instead of the observed
data. These can be computed using \code{\link[EbayesThresh]{beta.laplace}}.
}
\item{weights}{The different elements of the list can be weighted. This
allows for giving greater weight to certain arrays. By default, no
weights are used.}
\item{control}{A list that allows the user to tweak the behaviour of
\code{wtthresh}. It can contain the following elements:
\describe{
\item{max.depth}{The maximum depth of the tree. Defaults to \code{10}.}
\item{minimum.width}{The minimum width of a region of the partitioning
  of the largest array. This setting avoids creating too small regions. Defaults to \code{4}.}
\item{min.width.scale.factor}{If the minimum width of the largest array
  of size \eqn{l_0^d} is \code{minimum.width}, then the minimum width
  for an array of the size \eqn{l^2} is \eqn{\texttt{minimum.width}
    \cdot \left(\frac{l}{l_0}\right)^{d\cdot
      \texttt{minimum.width.scale.factor}}}{minimum.width * (l / l_0) ^
    (d * min.width.scale.factor)}. Defaults to \code{1}.}
\item{min.minimum.width}{Do not allow the minimum width of a region to
  become less
  than \code{min.minimum.width} at any level. Defaults to \code{1}.}
\item{minimum.size}{The minimum size of a region of the partitioning
  of the largest array. This setting avoids creating too small
  regions. Defaults to \eqn{8^d}, where \eqn{d} is the dimension of the arrays.}
\item{min.size.scale.factor}{If the minimum size of the largest array
  of size \eqn{l_0^d} is \code{minimum.size}, then the minimum size
  for an array of the size \eqn{l^2} is \eqn{\texttt{minimum.size}
    \cdot
    \left(\frac{l}{l_0}\right)^{\texttt{minimum.size.scale.factor}}
  }{minimum.size * (l / l_0) ^ (min.size.scale.factor)}. Defaults to \code{1}.}
\item{min.minimum.size}{Do not allow the minimum size of a region to
  become less than \code{min.minimum.size} at any level. Defaults to \eqn{4^d}, where \eqn{d} is the dimension of the arrays.}

\item{rescale.quantile}{In order to correct for different depths of the
  splits (due to minimum size and width requirements) the score statistic \eqn{s} is rescaled: \eqn{(s-q_{\alpha}(df)) / q_{\alpha}(df)}{(s-q(df,alpha)) /
    q(df,alpha)}, where \eqn{q_{\alpha}(df)}{q(df,alpha)} is the
  \eqn{\alpha}{alpha} quantile of a \eqn{\chi^2}{chi square}
  distribution with \eqn{df} degrees of freedom, and \eqn{\alpha}{alpha}
  is set to \code{rescale.quantile}. Defaults to \code{0.5}.}
\item{lr.signif}{If the p-value of the corresponding likelihood
  ratio test is larger than \code{1-lr.signif} a split will be
  discarded. Defaults to \code{0.5}.}
\item{absolute.improvement}{The minimum absolute improvement of the
  above criterion necessary such that a split is retained. Defaults to \code{-Inf}, i.e. deactivated.}
\item{relative.improvement}{The minimum relative improvement of the
  above criterion necessary such that a split is retained. Defaults to \code{-Inf}, i.e. deactivated.}
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
}
\value{
  \code{wtthresh} returns an object of the class \code{c("wtthresh")},
  which is a list containing the following elements:
  \item{splits}{A table describing the structure of the fitted tree
  together with the local loglikelihoods required for the pruning.}
  \item{details}{A table giving the details about where the split was
  carried out for each of the arrays (i.e. for each block of coefficients).}
  \item{w}{The weights \eqn{w} of the mixture component corresponding to
  the signal for each region as described by the corresponding row of \code{splits}.}
  \item{t}{The corresponding hard threshold \eqn{t} for each region as described by the corresponding row of \code{splits}.}
  \item{membership}{A list of the same length as \code{data} indicating
  to which region each entry of the arrays of data belongs.}
  \item{beta}{The values of \eqn{\beta}{beta} for each coefficient (as a list).}  
  \item{data}{The data used (as a list).}
  \item{weights}{The weights used for each array of observations/coefficients.}
  \item{control}{The control list of tuning options used. (see argument \code{control}).}
}
\references{
Evers, L. and Heaton T. (2009) Locally adaptive tree-based thresholding.
}
\note{
For an example of the use of \code{wtthresh}, see \code{\link{coefficients}}.
}
