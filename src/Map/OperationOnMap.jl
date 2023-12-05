

### SUM ###

struct SumMap{I,O,D1<:AbstractMap,D2<:Union{Number,AbstractArray,AbstractMap}} <:  AbstractMap{I,O}
	inputspace::I
	outputspace::O
	left::D1
	right::D2
end


function SumMap(A::D1, B::D2) where {I,O, D1<:AbstractMap{I,O},  D2<:AbstractMap{I,O}} 
	# FIXME should use some kind of promotion
	insp  = inputspace(A)
	outsp = outputspace(A)
	return SumMap(insp,outsp,A,B)
end


function SumMap(A::D1, B::D2) where {I,OL,OR, D1<:AbstractMap{I,OL},  D2<:AbstractMap{I,OR}} 
	# FIXME should use some kind of promotion
	insp  = inputspace(A)
	outsp = promote_space(outputspace(A), outputspace(B))
	return SumMap(insp,outsp,A,B)
end

function SumMap(A::D1, v::D2) where {N,I,O<:AbstractCoordinateSpace{N},T, D1<:AbstractMap{I,O},  D2<:AbstractArray{T,N}} 
	@assert  v ∈ outputspace(A) "The size of the added array must be of the size of the output: $(outputsize(A))"
	return SumMap(inputspace(A),outputspace(A),A,v)
end
function SumMap(A::D1, a::T) where {I,O,T<:Number, D1<:AbstractMap{I,O}} 
	@assert T<:eltype(O) " The scalar must be of type  <:$(eltype(O))"
	if iszero(a)
		return A
	end
	return SumMap(inputspace(A),outputspace(A),A,a)
end


@functor SumMap

add(A::AbstractMap, B::AbstractMap) = SumMap(A,B)
add(A::AbstractMap, B::Number)  =  SumMap(A,B)
add(A::AbstractMap, B::AbstractArray) = SumMap(A,B)

#MapSum(A::AbstractMap,B) = throw(SimpleAlgebraFailure("Dimension or type mismatch in MapSum"))

apply_(A::SumMap ,x)  = A.left(x) .+ A.right(x)
apply_(A::SumMap{I,O, D1,D2} ,x) where {I,O,D1<:AbstractMap{I,O}, D2<:Union{Number,AbstractArray}} = A.left*x .+ A.right


### Inverse ###

function inverse(A::M) where {M<:AbstractMap}
	if hasmethod(apply_inverse_,(M,Any))
		return InverseMap(A)
	end
    throw(SimpleAlgebraFailure("Unimplemented inverse for $(typeof(A))"))
end

struct InverseMap{I,O,D<:AbstractMap} <:  AbstractMap{I,O}
	parent::D
	InverseMap(A::AbstractMap{O,I}) where {I,O}   = new{I,O,typeof(A)}(A)
end

@functor InverseMap

InverseMap(A::InverseMap) = A.parent

inputspace(A::InverseMap)  = outputspace(A.parent)
outputspace(A::InverseMap) = inputspace(A.parent)


apply_(A::InverseMap, v) = apply_inverse_(A.parent,v)
apply_inverse_(A::InverseMap, v) = apply_(A.parent,v)


### COMPOSITION

compose(A::AbstractMap,B::AbstractMap) = CompositionMap(A, B)


struct CompositionMap{I,O,Dleft<:AbstractMap,Dright<:AbstractMap} <:  AbstractMap{I,O}
	left::Dleft
	right::Dright
	function CompositionMap(A::Dleft, B::Dright) where {I1,O1, I2, O2,Dleft<:AbstractMap{I1,O1},  Dright<:AbstractMap{I2,O2}} 
		    return new{I2,O1, Dleft,Dright}(A,B)
	end
end

@functor CompositionMap


inputspace(A::CompositionMap)  = inputspace(A.right)
outputspace(A::CompositionMap) = outputspace(A.left)


apply_(A::CompositionMap, v) = apply(A.left,apply(A.right,v))
apply_jacobian_(A::CompositionMap, v) = apply_jacobian_(A.right,apply_jacobian_(A.left,v))

inverse(A::CompositionMap) = inverse(A.right) * inverse(A.left)