struct LinOpDiag{I,D<:Union{AbstractArray,Number}} <:  AbstractLinOp{I,I}
	diag::D
end

function LinOpDiag(::Type{T}, sz::NTuple,diag::T1) where {T<:Number,T1<:Number}  
	return LinOpDiag{CoordinateSpace{sz},T}(diag)
end

function LinOpDiag(::Type{T}, diag::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	diag = convert.(T, diag)
	return LinOpDiag(diag)
end

function LinOpDiag(diag::D) where {T<:Number,D<:AbstractArray{T}}  
	sz = size(diag)
	return LinOpDiag{CoordinateSpace{sz},D}(diag)
end

function LinOpDiag(sz::NTuple,diag::D) where {T<:Number,D<:AbstractArray{T}}  
	@assert sz==size(diag) "the diagonal should be of size $sz"
	return LinOpDiag{sz,D}(diag)
end

function apply(A::LinOpDiag{I,D}, v) where {I,D}
	return v .* A.diag
end

apply_adjoint(A::LinOpDiag, v) =  v .* conj.(A.diag)
	

makeHtH(obj::LinOpDiag{I,D}) where {I,D} = LinOpDiag{I,D}(abs2.(obj.diag))
	

function compose(A::LinOpDiag{I,D1}, B::LinOpDiag{I,D2}) where {I,D1,D2}
	#T = promote_type(T1,T2)
	#diag = convert.(T,A.diag)  .* convert.(T,B.diag)
	diag = A.diag  .* B.diag

	#diag = (.+)(promote.(A.diag,B.diag))
	return LinOpDiag{I,typeof(diag)}(diag)
end


# function ChainRulesCore.rrule( ::typeof(apply),A::AbstractLinOp, v)
#     ∂Y(Δy) = (NoTangent(),NoTangent(), apply_adjoint(A,Δy))
#     return apply(A,v), ∂Y
# end