struct LinOpDiag{I,D<:AbstractArray} <:  AbstractLinOp{I,I}
	diag::D
end

LinOpDiag(::Type{T}, sz::NTuple{N,Int},diag::T1) where {T<:Number,T1<:Number, N} = LinOpScale(T,sz,diag)
LinOpDiag(sz::NTuple,diag::T) where {T<:Number} = LinOpScale(sz,diag)

function LinOpDiag(::Type{T}, diag::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	diag = convert.(T, diag)
	return LinOpDiag(diag)
end

function LinOpDiag(diag::D) where {T<:Number,D<:AbstractArray{T}}  
	sz = size(diag)
	return LinOpDiag{CoordinateSpace{sz},D}(diag)
end

function LinOpDiag(sz::NTuple{N,Int},diag::D) where {T<:Number,D<:AbstractArray{T},N}  
	@assert sz==size(diag) "the diagonal should be of size $sz"
	return LinOpDiag{sz,D}(diag)
end

apply(A::LinOpDiag{I,D}, v) where {I,D} = @. v * A.diag

apply_adjoint(A::LinOpDiag{I,D}, v) where {I,D} = @. v * conj(A.diag)
	
makeHtH(obj::LinOpDiag{I,D}) where {I,D} = LinOpDiag{I,D}(@. abs2(obj.diag))
	
compose(A::LinOpDiag{I,D1}, B::LinOpDiag{I,D2}) where {I,D1,D2} = LinOpDiag(@. A.diag * B.diag)

compose(A::LinOpScale{I,T}, B::LinOpDiag{I,D}) where {I,T<:Number,D}  = LinOpDiag( size(I), A.scale * B.diag)

# FIXME issue here should be generated only when apply_adjoint is implemented
function ChainRulesCore.rrule( ::typeof(apply),A::AbstractLinOp, v)
    ∂Y(Δy) = (NoTangent(),NoTangent(), apply_adjoint(A,Δy))
    return apply(A,v), ∂Y
end