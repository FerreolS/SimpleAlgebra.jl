struct LinOpConv{I,D<:LinOpDiag,FT<:LinOpDFT} <:  AbstractLinOp{I,I}
	M::D 	# Transfer function (LinOpDiag)
	F::FT	# Fourier operator
end

function  LinOpConv(mtf::AbstractArray)
	M = LinOpDiag(mtf) 
	sz = size(mtf)
	F = LinOpDFT(ComplexF64,sz)

    # Build operator.
    D = typeof(M)
    FT = typeof(F)
	I = inputspace(F)
	return LinOpConv{I,D,FT}(M,F)
end


function  LinOpConv(::Val{:psf},psf::AbstractArray{T}) where{T<:Number}
	sz = size(psf)
	F = LinOpDFT(T,sz)
	M =  LinOpDiag(F*psf) 

    # Build operator.
    D = typeof(M)
    FT = typeof(F)
	I = inputspace(F)
	return LinOpConv{I,D,FT}(M,F)
end

function apply(A::LinOpConv, x)
	return A.F'*(A.M*(A.F*x))
	#return apply_adjoint(A.F, apply(A.M, apply(A.F,x)))
end


function apply_adjoint(A::LinOpConv, x)
	return A.F'*( A.M' *( A.F * x))
	#return apply_adjoint(A.F, apply_adjoint(A.M, apply(A.F,x)))
end

expand(A::LinOpConv) = A.F' * A.M * A.F