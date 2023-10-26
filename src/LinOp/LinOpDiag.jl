
### IDENTITY
struct LinOpIdentity{I} <:  AbstractLinOp{I,I} end

LinOpIdentity(sz::NTuple{N,Int}) where {N} = LinOpIdentity{CoordinateSpace{sz}}()

apply_(::LinOpIdentity{I}, x) where {I} = x

apply_adjoint_(::LinOpIdentity{I}, x) where {I} =  x
Base.adjoint(A::LinOpIdentity) = A	
makeHtH(A::LinOpIdentity{I}) where {I} = A

compose(A::AbstractMap{I,O}, ::LinOpIdentity{I}) where {I,O} = A
compose(::LinOpIdentity{I},A::AbstractMap{I,O}) where {I,O} = A
compose(A::SimpleAlgebra.LinOpIdentity{I}, ::SimpleAlgebra.LinOpIdentity{I}) where I =  A


### SCALING 

struct LinOpScale{I,T<:Number} <:  AbstractLinOp{I,I} 
	scale::T
end

function LinOpScale(::Type{T}, sz::NTuple{N,Int}, scale::T1) where {T<:Number,T1<:Number,N}  
	scale = convert(T, scale)
	return LinOpScale(sz,scale)
end

function LinOpScale(sz::NTuple{N,Int}, scale::T) where {T<:Number,N}
	scale==oneunit(scale) && return LinOpIdentity(sz)
    LinOpScale{CoordinateSpace{sz},T}(scale)
end

apply_(A::LinOpScale{I,T}, x) where {I,T} = A.scale * x

apply_adjoint_(A::LinOpScale{I,T}, x) where {I,T} =  conj(A.scale) * x
	
makeHtH(A::LinOpScale{I,T}) where {I,T} = A

compose(A::AbstractLinOp{I,O}, B::LinOpScale{I,T}) where {I,O,T} = compose(B,A)
Base.:*(scalar::T, A::AbstractMap{I,O}) where {I, O,T<:Number} = compose(LinOpScale( size(O), scalar), A)
compose(A::LinOpScale{I,T}, B::LinOpScale{I,T1}) where {I,T<:Number,T1<:Number}  = LinOpScale( size(I), A.scale * B.scale)


### DIAGONAL (element-wise multiplication) 

struct LinOpDiag{I,D<:AbstractArray} <:  AbstractLinOp{I,I}
	diag::D
	LinOpDiag{I,D}(diag::D) where {I,T,D<:AbstractArray{T}}  = new{I,D}(diag)
end

function LinOpDiag(diag::D) where {T<:Number,D<:AbstractArray{T}}  
	sz = size(diag)
	return LinOpDiag{CoordinateSpace{sz},D}(diag)
end

LinOpDiag{I,D1}(diag::D2) where {I,T1,T2,D1<:AbstractArray{T1},D2<:AbstractArray{T2}}  = LinOpDiag{I,D2}(diag)



LinOpDiag(::Type{T}, sz::NTuple{N,Int},diag::T1) where {T<:Number,T1<:Number, N} = LinOpScale(T,sz,diag)
LinOpDiag(sz::NTuple,diag::T) where {T<:Number} = LinOpScale(sz,diag)

function LinOpDiag(::Type{T}, diag::D) where {T<:Number,T1<:Number,D<:AbstractArray{T1}}  
	diag = convert.(T, diag)
	return LinOpDiag(diag)
end

function LinOpDiag(sz::NTuple{N,Int},diag::D) where {T<:Number,D<:AbstractArray{T},N}  
	@assert sz==size(diag) "the diagonal should be of size $sz"
	return LinOpDiag{sz,D}(diag)
end

apply_(A::LinOpDiag{I,D}, v) where {I,D} = @. v * A.diag

apply_adjoint_(A::LinOpDiag{I,D}, v) where {I,D} = @. v * conj(A.diag)
	
makeHtH(obj::LinOpDiag{I,D}) where {I,D} = LinOpDiag{I,D}(@. abs2(obj.diag))
	
compose(A::LinOpDiag{I,D1}, B::LinOpDiag{I,D2}) where {I,D1,D2} = LinOpDiag(@. A.diag * B.diag)

compose(A::LinOpScale{I,T}, B::LinOpDiag{I,D}) where {I,T<:Number,D}  = LinOpDiag( size(I), A.scale * B.diag)
