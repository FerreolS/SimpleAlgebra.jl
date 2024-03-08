struct LinOpSelect{N,I<:CoordinateSpace,O<:CoordinateSpace,D<:Union{(AbstractArray{T,N} where T),Number}} <:  AbstractLinOp{I,O} 
	inputspace::I
	outputspace::O
	index::Vector{CartesianIndex{N}}
	zeroarray::D
	LinOpSelect(inputspace::I,outputspace::O, list::Vector{CartesianIndex{N}},zeroarray::D) where {TI,T,N,I<:CoordinateSpace{TI,N},O<:CoordinateSpace,D<:Union{AbstractArray{T,N},T}} =  new{N,I,O,D}(inputspace,outputspace,list,zeroarray)
	#LinOpSelect(::Type{T},inputspace::I,outputspace::O, list::Vector{CartesianIndex{N}},zeroarray::D) where {T<:Number,N,I<:AbstractCoordinateSpace{N},O<:CoordinateSpace,D<:Union{AbstractArray{T,N},T}} =  new{T,N,I,O,D}(inputspace,outputspace,list)
end

function LinOpSelect(::Type{T},sz::NTuple{N,Int},list::Vector{CartesianIndex{N}}) where {N,T}
	inputspace  = CoordinateSpace(T,sz)
	outputspace = CoordinateSpace(T,length(list))
	if isconcretetype(T)
		zeroarray = zeros(T,sz)
	else
		zeroarray = T(0)
	end
	LinOpSelect(inputspace,outputspace, list,zeroarray)
end

LinOpSelect(sz::NTuple{N,Int}, list::Vector{CartesianIndex{N}}) where {N} = LinOpSelect(Number,sz,list)

function LinOpSelect(::Type{T},selected::BitArray) where T
	sz = size(selected)
	list = findall(selected)
	LinOpSelect(T,sz,list)
end


LinOpSelect(selected::BitArray) =LinOpSelect(Number,selected::BitArray) 

@functor LinOpSelect

apply_(A::LinOpSelect, x)  =  x[A.index] #view(x,A.index) 

function apply_adjoint_(A::LinOpSelect{N,I,O,D}, x) where {N,I,O,D<:Number}
	tmp = zeros(eltype(x),inputspace(A))	
	ChainRulesCore.@ignore_derivatives	view(tmp,A.index) .= x
	return tmp
end

function apply_adjoint_(A::LinOpSelect{N,I,O,D}, x) where {N,I,O,D<:AbstractArray}
	T = eltype(O)
	tmp = A.zeroarray 
	ChainRulesCore.@ignore_derivatives	view(tmp,A.index) .= T.(x)
	return tmp
end

function compose(left::B, right::C)  where {N,I,O,D,C<:LinOpSelect{N,I,O,D},B<:AdjointLinOp{O,I,C}} 
    if left.parent===right
		return LinOpIdentity(inputspace(left))
	end
	return CompositionMap(left,right)
end

function compose(left::C, right::B)  where {N,I,O,D,C<:LinOpSelect{N,I,O,D},B<:AdjointLinOp{O,I,C}} 
    if left===right.parent
		sp =inputspace(left)
		diag = zeros(sp)
		view(diag,left.index) .= one(eltype(diag))
		return LinOpDiag(sp,sp, diag)
	end
	return CompositionMap(left,right)
end

function ChainRulesCore.rrule( ::typeof(apply_),A::LinOpSelect, v)
    LinOpSelect_pullback(Δy) = (NoTangent(),NoTangent(), apply_adjoint_(A, Δy))
    return  apply_(A,v), LinOpSelect_pullback
end

function ChainRulesCore.rrule( ::typeof(apply_adjoint_),A::LinOpSelect, v)
    LinOpSelect_pullback(Δy) = (NoTangent(),NoTangent(), apply_(A, Δy))
    return  apply_adjoint_(A,v), LinOpSelect_pullback
end 