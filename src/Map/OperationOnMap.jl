

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

function SumMap(A::D1, v::D2) where {I,O<:CoordinateSpace, D1<:AbstractMap{I,O},  D2<:AbstractArray} 
	v ∈ outputspace(A)  || throw(SimpleAlgebraFailure("The size of the added array must be of the size of the output: $(outputsize(A))"))
	return SumMap(inputspace(A),outputspace(A),A,v)
end
function SumMap(A::D1, a::T) where {I,O,T<:Number, D1<:AbstractMap{I,O}} 
	T<:eltype(O)  || throw(SimpleAlgebraFailure(" The scalar must be of type  <:$(eltype(O))"))
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
apply_(A::SumMap{I,O, D1,D2} ,x) where {I,O,D1<:AbstractMap, D2<:Union{Number,AbstractArray}} = A.left*x .+ A.right


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
#apply_jacobian_(A::CompositionMap, v, x) = apply_jacobian_(A.right,apply_jacobian_(A.left,v,x),A.left*x)

inverse(A::CompositionMap) = inverse(A.right) * inverse(A.left)

struct StackMap{I,O,N,S<:NTuple{N,AbstractMap},II,OI} <:  AbstractMap{I,O}
	inputspace::I
	outputspace::O
	terms::S
	inputindex::II
	outputindex::OI
end	
@functor StackMap



function apply_(A::StackMap, x)
	y = similar(x,outputspace(A))
	apply_!(y,A::StackMap, x)
end

function apply_!(y,A::StackMap, x) 
	fill!(y, 0)
    for (I, O, M) in zip(A.inputindex,A.outputindex, A.terms)
		y[O] .+= reshape(M*reshape(x[I],inputsize(M)),size(y[O]))
	end 
	return y
end



function apply_jacobian_(A::StackMap, v, x)
	y = similar(x,inputspace(A))
	fill!(y, 0)
    for (I, O, M) in zip(A.inputindex,A.outputindex, A.terms)
		y[I] .+= apply_jacobian(M,reshape(v[I],inputsize(M)),reshape(x[O],outputsize(M)))
	end 
	return y
end

function StackMap(terms::NTuple{N,AbstractMap}; indims=1, outdims=1, reversed=false) where N
	ind2c = Base.dims2cat(indims)
	insize = Base.cat_shape(ind2c,inputsize.(terms))
	intype = narrow_type( inputtype.(terms)...)
	inspace = CoordinateSpace(intype,insize)

	outd2c = Base.dims2cat(outdims)
	#outsize = Base.cat_shape(outd2c,outputsize.(terms))
	outsize = Base.cat_size_shape(outd2c,outputspace.(terms)...)
	outtype = narrow_type( outputtype.(terms)...)
	outspace = CoordinateSpace(outtype,outsize)

	inputindex = Vector{CartesianIndices}(undef,N)
	outputindex = Vector{CartesianIndices}(undef,N)

	inoff = CartesianIndex(zeros(Int,ndims(inspace))...)
	outoff = CartesianIndex(zeros(Int,ndims(outspace))...)
	for  (index, op) ∈ enumerate(terms) 
		inputindex[index] = inoff .+ CartesianIndices(extend_axes(axes(inputspace(op)),ndims(inspace)))
		inoff = CartesianIndex(last(inputindex[index]).I .* ind2c)

		outputindex[index] = outoff .+ CartesianIndices(extend_axes(axes(outputspace(op)),ndims(outspace)))
		outoff = CartesianIndex(last(outputindex[index]).I .* outd2c)
	end 
	if reversed
		for i ∈ eachindex(outputindex)
			outputindex[i] = CartesianIndices(reverse(outputindex[i].indices))
		end
		outsize = Base.cat_size_shape(outd2c,outputspace.(terms)...)
		outspace = CoordinateSpace(outtype,reverse(outsize))
	end
	return StackMap(inspace,outspace,terms,inputindex,outputindex)
end
