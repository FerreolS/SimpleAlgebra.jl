
struct LinOpSDFT{T,I,O} <:  AbstractLinOp{T,I,O}
	index # index along wich dimension are computed the FFT
	Notindex # ~index
	N # Number of element
	ndms # number of dimensions
	unitary
end