struct LinOpDiag{T,I,D<:Union{AbstractArray,T}} <:  AbstractLinOp{T,I,I}
	diag::D
end

LinOpDiag(sz::NTuple,diag::T) where {T<:Number}  =  LinOpDiag(T,sz,diag)
function LinOpDiag(::Type{T}, sz::NTuple,diag::T1) where {T<:Number,T1<:Number}  
	return LinOpDiag{T,sz,T}(diag)
end

function LinOpDiag(::Type{T}, diag::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	diag = convert.(T, diag)
	return LinOpDiag(T,diag)
end
function LinOpDiag(::Type{T}, diag::D) where {T<:Number,D<:AbstractArray{T}}  
	sz = size(diag)
	return LinOpDiag{T,sz,D}(diag)
end
function LinOpDiag(::Type{T}, sz::NTuple,diag::D) where {T<:Number,D<:AbstractArray{T}}  
	@assert sz==size(diag) "the diagonal should be of size $sz"
	return LinOpDiag{T,sz,D}(diag)
end

LinOpDiag(diag::D) where {T<:Number,D<:AbstractArray{T}}  =  LinOpDiag(T, diag)

LinOpDiag{T}(obj::LinOpDiag{T2,I,D}) where {T,T2,I,D} = LinOpDiag(T,I,convert.(T,obj.diag))

function apply(A::LinOpDiag{T,I,D}, v) where {T,I,D}
	return v .* A.diag
end

function apply_adjoint(A::LinOpDiag{T,I,D}, v) where {T,I,D}
	if T<:Real
		return v .* A.diag
	else
		return v .* conj.(A.diag)
	end
end

makeHtH(obj::LinOpDiag{T,I,D}) where {T,I,D} = LinOpDiag{T,I,D}(abs2.(obj.diag))
	

function compose(A::LinOpDiag{T,I,D1}, B::LinOpDiag{T,I,D2}) where {T,I,D1,D2}
	#T = promote_type(T1,T2)
	#diag = convert.(T,A.diag)  .* convert.(T,B.diag)
	diag = A.diag  .* B.diag

	#diag = (.+)(promote.(A.diag,B.diag))
	return LinOpDiag{T,I,typeof(diag)}(diag)
end