
### IDENTITY
struct LinOpIdentity{I} <:  AbstractLinOp{I,I} 
	inputspace::I
end

@functor LinOpIdentity

outputspace(A::LinOpIdentity) = A.inputspace


LinOpIdentity(sz::NTuple) = LinOpIdentity(CoordinateSpace(sz))
LinOpIdentity(sz::Int) = LinOpIdentity(Tuple(sz))
LinOpIdentity(inputspace::I) where {I<:AbstractDomain} = LinOpIdentity{I}(inputspace) 

apply_(::LinOpIdentity{I}, x) where {I} = x

apply_adjoint_(::LinOpIdentity{I}, x) where {I} =  x
Base.adjoint(A::LinOpIdentity) = A	
makeHtH(A::LinOpIdentity{I}) where {I} = A

# FIXME should we be restrictive about the size?
compose(A::AbstractMap, ::LinOpIdentity)  = A
compose(A::AbstractLinOp, ::LinOpIdentity)  = A
compose(::LinOpIdentity,A::AbstractMap) = A
compose(A::LinOpIdentity, ::LinOpIdentity)  =  A


### SCALING 

struct LinOpScale{I<:CoordinateSpace,O<:CoordinateSpace,T} <:  AbstractLinOp{I,O} 
	inputspace::I
	outputspace::O
	scale::T
	LinOpScale(inputspace::I,outputspace::O, scale::T) where {I<:CoordinateSpace,O<:CoordinateSpace,T} =  new{I,O,T}(inputspace,outputspace,scale)
end

@functor LinOpScale

function LinOpScale(::Type{TI}, sz::NTuple{N,Int}, scale::T1) where {TI,T1,N}  
	if T1==TI
		scale==oneunit(scale) && return LinOpIdentity(sz)
	end
	inputspace = CoordinateSpace(TI,sz)
	TO = isconcretetype(TI) ? typeof(oneunit(TI) * oneunit(T1)) : TI
	outputspace = CoordinateSpace(TO,sz)
	return LinOpScale(inputspace,outputspace, scale)
end
LinOpScale(::Type{T},sz::Int, scale) where {T} = LinOpScale(T,Tuple(sz),scale)

function LinOpScale(sz::NTuple{N,Int}, scale::T) where {T<:Number,N}
	scale==oneunit(scale) && return LinOpIdentity(sz)
	inputspace = CoordinateSpace(Number,sz)
	outputspace = CoordinateSpace(Number,sz)
    LinOpScale(inputspace,outputspace, scale)
end
LinOpScale(sz::Int, scale) = LinOpScale(Tuple(sz),scale)
LinOpScale(scale) = LinOpScale((), scale)

apply_(A::LinOpScale, x)  = A.scale * x

apply_adjoint_(A::LinOpScale, x) =  conj(A.scale) * x

function makeHtH(A::LinOpScale{I,O,T}) where {I,O,T}
	TI = eltype(I)
    LinOpScale(TI, inputsize(A), abs2(A.scale))
end

compose(A::AbstractLinOp{I,O}, B::LinOpScale{I,T}) where {I,O,T} = compose(LinOpScale( outputsize(A), B.scale),A)
Base.:*(scalar::T, A::AbstractMap{I,O}) where {I, O,T<:Number} = compose(LinOpScale( outputsize(A), scalar), A)
compose(A::LinOpScale{I,T}, B::LinOpScale{I,T1}) where {I,T<:Number,T1<:Number}  = LinOpScale( inputsize(A), A.scale * B.scale)


### DIAGONAL (element-wise multiplication) 

struct LinOpDiag{I<:CoordinateSpace,O<:CoordinateSpace,D<:AbstractArray} <:  AbstractLinOp{I,O}
	inputspace::I
	outputspace::O
	diag::D
	LinOpDiag(inputspace::I,outputspace::O, diag::D) where {I<:CoordinateSpace,O<:CoordinateSpace,D<:AbstractArray} =  new{I,O,D}(inputspace,outputspace,diag)
end

@functor LinOpDiag

function LinOpDiag(::Type{TI}, diag::AbstractArray{T1}) where {TI,T1}  
	sz = size(diag)
	inputspace = CoordinateSpace(TI,sz)
	TO = isconcretetype(TI) ? typeof(oneunit(TI) * oneunit(T1)) : TI
	outputspace = CoordinateSpace(TO,sz)
	return LinOpDiag(inputspace,outputspace, diag)
end

LinOpDiag(scale::Number)  = LinOpScale(scale)




function LinOpDiag(diag::D) where {D<:AbstractArray}  
	sz = size(diag)
	inputspace = CoordinateSpace(sz)
	outputspace = CoordinateSpace(sz)
	return LinOpDiag(inputspace,outputspace, diag)
end

LinOpDiag(::Type{T}, sz::NTuple{N,Int},diag::T1) where {T<:Number,T1<:Number, N} = LinOpScale(T,sz,diag)
LinOpDiag(sz::NTuple,diag::T) where {T<:Number} = LinOpScale(sz,diag)

function LinOpDiag(sz::NTuple{N,Int},diag::D) where {T<:Number,D<:AbstractArray{T},N}  
	@assert sz==size(diag) "the diagonal should be of size $sz"
	return LinOpDiag(diag)
end

apply_(A::LinOpDiag, v)  = @. v * A.diag

apply_adjoint_(A::LinOpDiag, v)  = @. v * conj(A.diag)
	
function makeHtH(obj::LinOpDiag{I,O,T}) where {I,O,T}
	TI = eltype(I)
    LinOpDiag(TI, @. abs2(obj.diag))
end
	
compose(A::LinOpDiag{I,I,D1}, B::LinOpDiag{I,I,D2}) where {I,D1,D2} = LinOpDiag(@. A.diag * B.diag)

compose(A::LinOpScale{I,I,T}, B::LinOpDiag{I,I,D}) where {I,T<:Number,D}  = LinOpDiag( size(I), A.scale * B.diag)
