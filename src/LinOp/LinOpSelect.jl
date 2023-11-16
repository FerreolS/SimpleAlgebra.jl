struct LinOpSelect{I<:CoordinateSpace,O<:CoordinateSpace,T,N} <:  AbstractLinOp{I,O} 
	inputspace::I
	outputspace::O
	index::Vector{CartesianIndex{N}}
	LinOpSelect(inputspace::I,outputspace::O, list::T,N::Int) where {I<:CoordinateSpace,O<:CoordinateSpace,T} =  new{I,O,T,N}(inputspace,outputspace,list)
	LinOpSelect(inputspace::I,outputspace::O, list::T) where {I<:CoordinateSpace,O<:CoordinateSpace,T} =  new{I,O,T,ndims(inputspace)}(inputspace,outputspace,list)

end

function LinOpSelect(::Type{T},sz::NTuple{N,Int}, list::Vector{CartesianIndex{N}}) where {N,T}
	inputspace  = CoordinateSpace(T,sz)
	outputspace = CoordinateSpace(T,length(list))
	LinOpSelect(inputspace,outputspace, list)
end

LinOpSelect(sz::NTuple{N,Int}, list::Vector{CartesianIndex{N}}) where {N} = LinOpSelect(Number,sz,list)
function LinOpSelect(::Type{T},selected::BitArray) where T
	sz = size(selected)
	list = findall(selected)
	LinOpSelect(T,sz,list)
end
LinOpSelect(selected::BitArray) =LinOpSelect(Number,selected::BitArray) 

@functor LinOpSelect

apply_(A::LinOpSelect, x)  = x[A.index] #view(x,A.index) 

function apply_adjoint_(A::LinOpSelect, x)
	tmp = zeros(eltype(x),inputspace(A))
	view(tmp,A.index) .= x
	return tmp
end

function makeHtH(A::LinOpSelect) 
	sz = inputsize(A)
	return LinOpIdentity(sz)
end