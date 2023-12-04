
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
AdjointLinOp(A::LinOpIdentity) = A	

# FIXME should we be restrictive about the size?
compose(::LinOpIdentity,A::LinOpIdentity)  = A
compose(A::AbstractMap, ::LinOpIdentity)  = A
compose(A::AbstractLinOp, ::LinOpIdentity)  = A
compose(::LinOpIdentity,A::AbstractMap) = A
compose(::LinOpIdentity,A::AbstractLinOp) = A

function add(A::LinOpIdentity, B::LinOpIdentity)  
	sp =inputspace(A)
	T = promote_type(eltype(sp),eltype(inputspace(B)))
	return LinOpScale(sp,sp, T.(2))
end

inverse(A::LinOpIdentity) = A
### SCALING 

struct LinOpScale{I<:CoordinateSpace,O<:CoordinateSpace,T} <:  AbstractLinOp{I,O} 
	inputspace::I
	outputspace::O
	scale::T
	LinOpScale(inputspace::I,outputspace::O, scale::T) where {I<:CoordinateSpace,O<:CoordinateSpace,T<:Number} =  new{I,O,T}(inputspace,outputspace,scale)
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

AdjointLinOp(A::LinOpScale{I,O,T}) where {T<:Real,I,O} = A
AdjointLinOp(A::LinOpScale) =   LinOpScale(outputspace(A),inputspace(A), conj(A.scale))

function compose(left::LinOpScale{I,O,Tl},right::LinOpScale{O,P,Tr}) where {I,O,P,Tr,Tl}
	insp = inputspace(right) 
	T = (isconcretetype(Tr) || isconcretetype(Tl)) ? typeof(oneunit(Tl) * oneunit(Tr)) : Tr
	outsp = CoordinateSpace(T,outputspace(left) )
    LinOpScale(insp, outsp,left.scale .* right.scale)
end

Base.:*(scalar::T, A::AbstractMap{I,O}) where {I, O,T<:Number} = compose(LinOpScale( outputsize(A), scalar), A)

compose(A::AbstractLinOp{I,O}, B::LinOpScale{I,T}) where {I,O,T} = compose(LinOpScale( outputsize(A), B.scale),A)
compose(A::LinOpScale{I,T}, B::LinOpScale{I,T1}) where {I,T<:Number,T1<:Number}  = LinOpScale( inputsize(A), A.scale * B.scale)

compose(::LinOpIdentity,B::LinOpScale)  = B
compose(B::LinOpScale,::LinOpIdentity)  = B

function add(A::LinOpScale{I,O,T} , ::LinOpIdentity) where {I,O,T}
	return LinOpScale(inputspace(A),outputspace(A), A.scale + oneunit(T) )
end
add( B::LinOpIdentity,A::LinOpScale)  = add(A,B)
add(A::LinOpScale , B::LinOpScale)  = LinOpScale(inputspace(A),outputspace(A), A.scale + B.scale )

inverse(A::LinOpScale) = LinOpScale(outputspace(A),inputspace(A), 1/A.scale )


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
apply_inverse_(A::LinOpDiag, v)  = @. A.diag \ v 

AdjointLinOp(A::LinOpDiag{I,O,D}) where {T<:Real,I,O,D<:AbstractArray{T}} = A
AdjointLinOp(A::LinOpDiag) =  LinOpDiag(outputspace(A),inputspace(A), conj.(A.diag))


	
compose(A::LinOpDiag, B::LinOpDiag)  = LinOpDiag(inputspace(A),outputspace(B),@. A.diag * B.diag)
compose(A::LinOpScale, B::LinOpDiag)  = LinOpDiag(inputspace(A),outputspace(B),@. A.scale * B.diag)
compose(A::LinOpDiag, B::LinOpScale)  = LinOpDiag(inputspace(A),outputspace(B),@. A.diag * B.scale)
compose(::LinOpIdentity,B::LinOpDiag)  = B
compose(B::LinOpDiag,::LinOpIdentity)  = B


add(A::LinOpDiag{I,O,D1}, B::LinOpDiag{I,O,D2})   where {I,O,D1,D2} = LinOpDiag(inputspace(A),outputspace(A),@. A.diag + B.diag)
add(A::LinOpDiag{I,O,D1}, B::LinOpScale{I,O,D2})   where {I,O,D1,D2} = LinOpDiag(inputspace(A),outputspace(A),@. A.diag + B.scale)
add(A::LinOpScale, B::LinOpDiag)  = add(B,A)
add(A::LinOpDiag{I,O,D1}, ::LinOpIdentity{I})   where {N,T,I,O,D1<:AbstractArray{T,N}} = LinOpDiag(inputspace(A),outputspace(A),A.diag .+ oneunit(T))
add(A::LinOpIdentity,B::LinOpDiag)  = add(B,A)

add(A::LinOpDiag, B::Number) = LinOpDiag(inputspace(A),outputspace(A),A.diag .+ B)
add(B::Number,A::LinOpDiag) = add(A,B)
add(A::LinOpDiag{I,O,D1}, B::AbstractArray{T,N})   where {T,N,I,O,D1<:AbstractArray{T,N}} = LinOpDiag(inputspace(A),outputspace(A),@. A.diag + B)
add(B::AbstractArray,A::LinOpDiag) = add(A,B)

Base.:\(A::LinOpDiag{I,O,D1}, B::LinOpDiag{I,O,D2})   where {I,O,D1,D2} = LinOpDiag(inputspace(A),outputspace(A),@. A.diag \ B.diag)
Base.:/(A::LinOpDiag{I,O,D1}, B::LinOpDiag{I,O,D2})   where {I,O,D1,D2} = LinOpDiag(inputspace(A),outputspace(A),@. A.diag / B.diag)

inverse(A::LinOpDiag{I,O,D}) where {T,I,O,D<:AbstractArray{T}} = LinOpDiag(outputspace(A),inputspace(A), one(T)./(A.diag) )
