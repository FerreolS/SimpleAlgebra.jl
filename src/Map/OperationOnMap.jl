

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

Base.:+(A::AbstractMap, B::AbstractMap)  = add(A,B)
Base.:+(A::AbstractMap, B::T) where {T<:Union{Number,AbstractArray}}= add(A,B)
Base.:+(B::T,A::AbstractMap) where {T<:Union{Number,AbstractArray}} = add(A,B)

apply_(A::MapSum{I,O, D1,D2} ,x) where {I,O,D1<:AbstractMap{I,O},  D2<:AbstractMap{I,O}}= apply(A.left,x) .+ apply(A.right,x)
apply_(A::MapSum{I,O, D1,D2} ,x) where {I,O,D1<:AbstractMap{I,O}, D2<:Union{Number,AbstractArray}} = apply(A.left,x) .+ A.right
