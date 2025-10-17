struct LinOpConv{I,D<:LinOpDiag,FT<:LinOpDFT} <:  AbstractLinOp{I,I}
    inputspace::I
	M::D 	# Transfer function (LinOpDiag)
	F::FT	# Fourier operator
	LinOpConv(insp::I, M::D,F::FT) where {I<:AbstractDomain,D<:LinOpDiag,FT<:LinOpDFT} = new{I,D,FT}(insp,M,F)
end


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
	M =  LinOpDiag(outputtype(F),F*psf) 
	insp = inputspace(F)
	return LinOpConv(insp,M,F)
end

if true
	function apply_(A::LinOpConv, x)
		return length(inputspace(A)) .\ (A.F'*(A.M*(A.F*x)))
		#return apply_adjoint_(A.F, apply_(A.M, apply_(A.F,x)))
	end
else
	
	function apply_(A::LinOpConv, x::T) where T
		wrk1 = Scratchspace(T)
		wrk2 = Scratchspace(T)
		tmp1 = apply!(wrk1,A.F,x)
		tmp2 = apply!(wrk2,A.M,tmp1)
		tmp1 = apply_adjoint!(wrk1,A.F,tmp2)
		return length(inputspace(A)) .\ tmp1
	end
end


function apply_adjoint_(A::LinOpConv, x)
	return length(inputspace(A)) .\ (A.F'*( A.M' *( A.F * x)))
	#return apply_adjoint_(A.F, apply_adjoint_(A.M, apply_(A.F,x)))
end


function apply_inverse_(A::LinOpConv, x)
	return A.F'*( inv(A.M) *( A.F * x))
	#return apply_inverse_(A.F, apply_inverse_(A.M, apply_inverse_(A.F,x)))
end 

expand(A::LinOpConv) = A.F' * A.M * A.F

# FIXME 2 Solutions
AdjointLinOp(A::LinOpConv{I,D,FT}) where{I,D,FT} = LinOpConv(inputspace(A),A.M',A.F)

function compose(A::AdjointLinOp{I,O,P}, B::LinOpConv{O,D,FT})  where {I,O,D,FT,P<:LinOpConv}
    if A.parent===B
		modulus = abs2.(B.M.diag)
		return LinOpConv(inputspace(B),LinOpDiag(modulus),B.F)
	end
	return LinOpConv(inputspace(B),A.parent.M' * B.M,B.F)
end

function compose(A::InverseMap{I,O,P}, B::LinOpConv{O,DB,FT})  where {I,O,DA,DB,FT,P<:LinOpConv{O,DA,FT}}
    if A.parent===B
		return LinOpIdentity(inputspace(B))
	end
	return LinOpConv(inputspace(B),A.parent.M \ B.M,B.F)
end

function compose(A::LinOpConv{I,DA,FT},B::LinOpConv{I,DB,FT})  where {I,DA,DB,FT}
	return LinOpConv(inputspace(B), A.M' * B.M ,B.F)
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
	return LinOpConv(inputspace(A), A.M + oneunit(TA) ,A.F)
end

function add(A::(LinOpConv{IA,D,F} where {D,F}), B::(LinOpScale{IB,OB,T} where {IB,T}))  where 
						{N,TA,TB,IA<:CoordinateSpace{TA,N},OB<:CoordinateSpace{TB,N}}
	T = promote_type(TA,TB)
	F = adapt(SimpleAlgebraEltypeAdaptor{T}(),A.F)
	inpsp = CoordinateSpace(T,inputspace(A))
	return LinOpConv(inpsp, A.M + T(B.scale) ,A.F)
end


function add(scalar::Number, A::LinOpConv) 
	iszero(scalar) && return A
    add(LinOpScale(inputspace(A),outputspace(A), outputtype(A)(scalar)),A)
end


inverse(A::LinOpConv) = LinOpConv(inputspace(A), inv(A.M),A.F) 