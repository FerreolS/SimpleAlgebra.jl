struct LinOpConv{I,D<:LinOpDiag,FT<:LinOpDFT} <:  AbstractLinOp{I,I}
    inputspace::I
	M::D 	# Transfer function (LinOpDiag)
	F::FT	# Fourier operator
	LinOpConv(insp::I, M::D,F::FT) where {I<:AbstractDomain,D<:LinOpDiag,FT<:LinOpDFT} = new{I,D,FT}(insp,M,F)
end

@functor LinOpConv

outputspace(A::LinOpConv) = A.inputspace

function  LinOpConv(mtf::AbstractArray{T}) where {T}
	M = LinOpDiag(mtf) 
	sz = size(mtf)
	F = LinOpDFT(T,sz)

    # Build operator.
	insp = inputspace(F)
	return LinOpConv(insp,M,F)
end


function  LinOpConv(::Val{:psf},psf::AbstractArray{T}; centered = true) where{T<:Number}
	sz = size(psf)
	F = LinOpDFT(T,sz)
	if !centered 
		psf = fftshift(psf)
	end
	M =  LinOpDiag(F*psf) 
	insp = inputspace(F)
	return LinOpConv(insp,M,F)
end

function apply_(A::LinOpConv, x)
	return A.F'*(A.M*(A.F*x))
	#return apply_adjoint_(A.F, apply_(A.M, apply_(A.F,x)))
end


function apply_adjoint_(A::LinOpConv, x)
	return A.F'*( A.M' *( A.F * x))
	#return apply_adjoint_(A.F, apply_adjoint_(A.M, apply_(A.F,x)))
end

expand(A::LinOpConv) = A.F' * A.M * A.F

function compose(A::LinOpAdjoint{I,O,P}, B::LinOpConv)  where {I,O,P<:LinOpConv}
    if A===B
		modulus = abs2.(B.M.diag)
		return LinOpConv(inputspace(B),LinOpDiag(modulus),B.F)
	end
	return LinOpComposition(A,B)
end

function add(A::(LinOpConv{IA,D,F} where {D,F}),
					B::(LinOpConv{IB,D,F} where {D,F})) where 
						{N,TA,TB,IA<:CoordinateSpace{TA,N},IB<:CoordinateSpace{TB,N}}
	T = promote_type(TA,TB)
	F = adapt(SimpleAlgebraEltypeAdaptor{T}(),A.F)
	inpsp = CoordinateSpace(T,inputspace(A))
	return LinOpConv(inpsp, A.M + B.M ,F)
end


function add(A::(LinOpConv{IA,D,F} where {D,F}), ::LinOpIdentity)  where 
						{N,TA,IA<:CoordinateSpace{TA,N}}
	return LinOpConv(inputspace(A), A.M .+ oneunit(TA) ,A.F)
end

function add(A::(LinOpConv{IA,D,F} where {D,F}), B::(LinOpScale{IB,OB,T} where {IB,T}))  where 
						{N,TA,TB,IA<:CoordinateSpace{TA,N},OB<:CoordinateSpace{TB,N}}
	T = promote_type(TA,TB)
	F = adapt(SimpleAlgebraEltypeAdaptor{T}(),A.F)
	inpsp = CoordinateSpace(T,inputspace(A))
	return LinOpConv(inpsp, A.M .+ T(B.scale) ,A.F)
end