
abstract type AbstractLinOp{I,O}  <: AbstractMap{I,O}  end

apply_adjoint(A::AbstractLinOp,x) = apply_jacobian(A,zeros(eltype(x),inputsize(A)),x)

Base.adjoint(A::AbstractLinOp) = LinOpAdjoint(A)





struct LinOpIdentity{I} <:  AbstractLinOp{I,I} end

LinOpIdentity(sz::NTuple) = LinOpIdentity{CoordinateSpace{sz}}()

apply(::LinOpIdentity{I}, x) where {I} = x

apply_adjoint(::LinOpIdentity{I}, x) where {I} =  x
	
makeHtH(A::LinOpIdentity{I}) where {I} = A

compose(A::AbstractMap{I,O}, ::LinOpIdentity{I}) where {I,O} = A
compose(::LinOpIdentity{I},A::AbstractMap{I,O}) where {I,O} = A



struct LinOpScale{I,T<:Number} <:  AbstractLinOp{I,I} 
	scale::T
end

function LinOpScale(::Type{T}, sz::NTuple, scale::T1) where {T<:Number,T1<:Number}  
	scale = convert(T, scale)
	return LinOpScale(sz,scale)
end

function LinOpScale(sz::NTuple, scale::T) where {T<:Number}
	scale==oneunit(scale) && return LinOpIdentity(sz)
    LinOpScale{CoordinateSpace{sz},T}(scale)
end

apply(A::LinOpScale{I,T}, x) where {I,T} = A.scale * x

apply_adjoint(A::LinOpScale{I,T}, x) where {I,T} =  conj(A.scale) * x
	
makeHtH(A::LinOpScale{I,T}) where {I,T} = A

compose(A::AbstractMap{I,O}, B::LinOpScale{I,T}) where {I,O,T} = compose(B,A)
compose(::LinOpIdentity{I}, B ::LinOpScale{I, T}) where {I, T} = B

compose(A::LinOpScale{I,T}, B::LinOpScale{I,T1}) where {I,T<:Number,T1<:Number}  = LinOpScale( size(I), A.scale * B.scale)



include("./LinOpDiag.jl")
include("./LinOpAdjoint.jl")
