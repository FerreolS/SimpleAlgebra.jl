

### SUM ###

struct MapSum{I,O,D1<:AbstractMap{I,O},D2<:Union{Number,AbstractArray,AbstractMap{I,O}}} <:  AbstractMap{I,O}
	inputspace::I
	outputspace::O
	left::D1
	right::D2
end


function MapSum(A::D1, B::D2) where {I,O, D1<:AbstractMap{I,O},  D2<:AbstractMap{I,O}} 
	# FIXME should use some kind of promotion
	insp  = inputspace(A)
	outsp = outputspace(A)
	return MapSum(insp,outsp,A,B)
end

function MapSum(A::D1, v::D2) where {N,I,O<:AbstractCoordinateSpace{N},T, D1<:AbstractMap{I,O},  D2<:AbstractArray{T,N}} 
	@assert  v ∈ outputspace(A) "The size of the added array must be of the size of the output: $(outputsize(A))"
	return MapSum(inputspace(A),outputspace(A),A,v)
end
function MapSum(A::D1, a::T) where {I,O,T<:Number, D1<:AbstractMap{I,O}} 
	@assert T<:eltype(O) " The scalar must be of type  <:$(eltype(O))"
	if iszero(a)
		return A
	end
	return MapSum(inputspace(A),outputspace(A),A,a)
end


@functor MapSum

add(A::AbstractMap, B::AbstractMap) = MapSum(A,B)
add(A::AbstractMap, B::Number)  =  MapSum(A,B)
add(A::AbstractMap, B::AbstractArray) = MapSum(A,B)

#MapSum(A::AbstractMap,B) = throw(SimpleAlgebraFailure("Dimension or type mismatch in MapSum"))

apply_(A::MapSum{I,O, D1,D2} ,x) where {I,O,D1<:AbstractMap{I,O},  D2<:AbstractMap{I,O}}= apply(A.left,x) .+ apply(A.right,x)
apply_(A::MapSum{I,O, D1,D2} ,x) where {I,O,D1<:AbstractMap{I,O}, D2<:Union{Number,AbstractArray}} = apply(A.left,x) .+ A.right


### Inverse ###

function inverse(A::M) where {M<:AbstractMap}
	if hasmethod(apply_inverse_,(M,Any))
		return MapInverse(A)
	end
    throw(SimpleAlgebraFailure("Unimplemented inverse for $(typeof(A))"))
end

struct MapInverse{I,O,D<:AbstractMap} <:  AbstractMap{I,O}
	parent::D
	MapInverse(A::AbstractMap{O,I}) where {I,O}   = new{I,O,typeof(A)}(A)
end

@functor MapInverse

MapInverse(A::MapInverse) = A.parent

inputspace(A::MapInverse)  = outputspace(A.parent)
outputspace(A::MapInverse) = inputspace(A.parent)


apply_(A::MapInverse, v) = apply_inverse_(A.parent,v)
apply_inverse_(A::MapInverse, v) = apply_(A.parent,v)


### COMPOSITION

compose(A::AbstractMap,B::AbstractMap) = MapComposition(A, B)


struct MapComposition{I,O,Dleft<:AbstractMap,Dright<:AbstractMap} <:  AbstractMap{I,O}
	left::Dleft
	right::Dright
	function MapComposition(A::Dleft, B::Dright) where {I1,O1, I2, O2,Dleft<:AbstractMap{I1,O1},  Dright<:AbstractMap{I2,O2}} 
		    return new{I2,O1, Dleft,Dright}(A,B)
	end
end

@functor MapComposition


inputspace(A::MapComposition)  = inputspace(A.right)
outputspace(A::MapComposition) = outputspace(A.left)


apply_(A::MapComposition, v) = apply(A.left,apply(A.right,v))
apply_jacobian_(A::MapComposition, v) = apply_jacobian_(A.right,apply_jacobian_(A.left,v))

inverse(A::MapComposition) = inverse(A.right) * inverse(A.left)