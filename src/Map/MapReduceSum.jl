struct MapReduceSum{I<:CoordinateSpace,O<:CoordinateSpace,N,F<:Function} <:  AbstractMap{I,O} 
	inputspace::I
	outputspace::O
	index::NTuple{N,Int}
	f::F
	MapReduceSum(inputspace::I,outputspace::O,index::NTuple{N},f::F) where {I<:CoordinateSpace,O<:CoordinateSpace,N,F<:Function} = 
			new{I,O,N,F}(inputspace, outputspace, index,f)
end

@functor MapReduceSum

MapReduceSum(sz::NTuple{N,Int}) 			where N 	=  MapReduceSum(Number,sz)
MapReduceSum(::Type{T},sz::NTuple{N,Int})  	where {T,N} = MapReduceSum(CoordinateSpace(T,sz), identity)
MapReduceSum(::Type{T},sz::NTuple{N,Int},f::Function) where {T,N} = MapReduceSum(CoordinateSpace(T,sz), f)
MapReduceSum(sz::NTuple{N,Int},f::Function) where {N} = MapReduceSum(CoordinateSpace(Number,sz), f)
function MapReduceSum(inputspace::CoordinateSpace{T,N},f::Function) where {T,N}
	outputspace = Scalar{T}()
	index = Tuple(1:N)
	return MapReduceSum(inputspace, outputspace, index,f)
end 


MapReduceSum(sz::NTuple{N,Int},index) where {N} =  MapReduceSum(Number,sz,index)
MapReduceSum(::Type{T},sz::NTuple{N,Int},index) where {T,N} = MapReduceSum(CoordinateSpace(T,sz),identity,index)
MapReduceSum(inputspace::CoordinateSpace,index) = MapReduceSum(inputspace,identity,index)
MapReduceSum(sz::NTuple{N,Int},f::Function,index) where {T,N} = MapReduceSum(CoordinateSpace(Number,sz), f,index)
function MapReduceSum(inputspace::CoordinateSpace{T,N},f::Function,index) where {T,N}
	index =  Tuple(unique(index))
	szin = size(inputspace)
	szout = szin[[i ∉ index for i ∈ 1:N]]
	outputspace = CoordinateSpace(T,szout)
	return MapReduceSum(inputspace, outputspace, index,f)
end 

apply_(A::MapReduceSum{I,O,N,F}, x) where {N,I,F, O} =  reshape(sum(A.f, x, dims=A.index),outputsize(A))
apply_(A::MapReduceSum{I,O,N,F}, x) where {T,N,I,F, O<:Scalar{T}}  =  sum(A.f, x)
