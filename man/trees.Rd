\name{trees}
\docType{data}
\alias{trees}
\title{Example image}
\description{
  Greyscale image of 512 x 512 pixels showing tree close to Swellendam,
  Western Cape, South Africa. The image was used as an example in Evers
  and Heaton (2009).
}
\usage{tiles}
\examples{
data(trees)
image(trees, col=grey(seq(0,1,length.out=64)))
}
\seealso{\code{\link{row263}}}
\references{
Evers, L. and Heaton T. (2009) Locally adaptive tree-based tresholding.
}
\keyword{datasets}
