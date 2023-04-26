struct LinOpDiag{I,D<:Union{AbstractArray,Number}} <:  AbstractLinOp{I,I}
	diag::D
end

LinOpDiag(sz::NTuple,diag::T) where {T<:Number}  =  LinOpDiag(sz,diag)
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

LinOpDiag(obj::LinOpDiag{I,D}) where {I,D} = LinOpDiag(obj.diag)

function apply(A::LinOpDiag{I,D}, v) where {I,D}
	return v .* A.diag
end

function apply_adjoint(A::LinOpDiag{I,D}, v) where {I,D}
	if T<:Real
		return v .* A.diag
	else
		return v .* conj.(A.diag)
	end
end

makeHtH(obj::LinOpDiag{I,D}) where {I,D} = LinOpDiag{I,D}(abs2.(obj.diag))
	

function compose(A::LinOpDiag{I,D1}, B::LinOpDiag{I,D2}) where {I,D1,D2}
	#T = promote_type(T1,T2)
	#diag = convert.(T,A.diag)  .* convert.(T,B.diag)
	diag = A.diag  .* B.diag

	#diag = (.+)(promote.(A.diag,B.diag))
	return LinOpDiag{I,typeof(diag)}(diag)
end