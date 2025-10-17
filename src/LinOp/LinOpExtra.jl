struct LinOp1DPolynomial{I,O,X<:Union{AbstractArray{<:AbstractFloat},AbstractFloat},P} <: AbstractLinOp{I,O}
	inputspace::I
	outputspace::O
	x::X
	precond::P
	LinOp1DPolynomial(inputspace::I,outputspace::O,x::X,p::P)  where {I<:CoordinateSpace,O<:CoordinateSpace,X<:Union{AbstractArray,Number},P} = 
		new{I,O,X,P}(inputspace,outputspace,x,p)
end


LinOp1DPolynomial(x,order) = LinOp1DPolynomial(Number,x,order)

function LinOp1DPolynomial(::Type{TI},x,degree) where TI
	inputspace = CoordinateSpace(TI,(degree+1,))
	TO = isconcretetype(TI) ? typeof(oneunit(TI) * oneunit(eltype(x))) : TI
	outputspace = CoordinateSpace(TO,size(x))
	precond = [ sqrt(length(x) /sum(x.^(2*n)) ) for n ∈ 0:degree]
	LinOp1DPolynomial(inputspace,outputspace,x,precond)
end

function apply_(A::LinOp1DPolynomial, coefs)
	coefs .*= A.precond
	degree = length(A.inputspace) - 1
	y = similar(A.x)
	backend = get_backend(coefs)
	LinOp1DPolynomialKernel_(backend)(y,A.x,coefs,degree, ndrange=size(A.x))
	synchronize(backend)
	return y
	#return mapreduce( n -> coefs[n+1] .* (A.x).^n,+,0:degree) # no GPU compatible
	#return reduce(hcat,[ (A.x).^n  for n ∈ 0:degree])*coefs
end

@kernel function LinOp1DPolynomialKernel_(Y, X,coefs,degree) 
	I = @index(Global, Cartesian)
	Y[I] = coefs[1]
	for n = 1:degree
		Y[I] += X[I].^n * coefs[n+1]
	end
end


function apply_adjoint_(A::LinOp1DPolynomial{I,O,X}, x) where {I,O,X}
	T =  isconcretetype(eltype(I)) ? eltype(I) : eltype(x)
	degree = length(A.inputspace) - 1
	y= similar(x,T,(degree+1,))
	backend = get_backend(x)
	LinOp1DPolynomialKernel_adjoint_(backend)(y,A.x,x, ndrange=(degree+1,))
	synchronize(backend)
	return y .* A.precond
end


@kernel function LinOp1DPolynomialKernel_adjoint_(Y, X,input) 
	(I,) = @index(Global, NTuple)
	Y[I] = sum(X.^(I-1) .* input)
end
#= 

function ChainRulesCore.rrule( ::typeof(apply_),A::LinOp1DPolynomial, v)
	LinOp1DPolynomial_pullback(Δy) = (NoTangent(),NoTangent(), apply_adjoint_(A, Δy))
    return  apply_(A,v), LinOp1DPolynomial_pullback
end
 =#