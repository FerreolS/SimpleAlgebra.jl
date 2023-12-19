

struct LinOpGrad{I,O} <:  AbstractLinOp{I,O}
    inputspace::I
	outputspace::O
	function LinOpGrad(inputspace::I) where {I<:AbstractDomain}
		N = ndims(inputspace)
		outputspace = CoordinateSpace(eltype(inputspace), (size(inputspace)...,N))
    	return new{I,typeof(outputspace),}(inputspace,outputspace)
	end
end 

LinOpGrad(sz::NTuple) 						= LinOpGrad(CoordinateSpace(sz))
LinOpGrad(sz::Int)							= LinOpGrad(Tuple(sz))
LinOpGrad(::Type{TI}, sz::Int) where TI 	= LinOpGrad(TI,Tuple(sz))
LinOpGrad(::Type{TI}, sz::NTuple) where TI 	= LinOpGrad(CoordinateSpace(TI,sz))
LinOpGrad(::Type{TI}, inputspace::CoordinateSpace) where TI = LinOpGrad(CoordinateSpace(TI,inputspace))

apply_(::LinOpGrad, x)  = compute_grad(x)

apply_adjoint_(::LinOpGrad, x) =  compute_grad_adjoint(x)

function compute_grad(x::AbstractArray{T,N}) where {T,N}
    sz = size(x)
	Y = similar(x,sz...,N)
	compute_grad!(Y,x)
	return Y
end


function compute_grad_adjoint(x::AbstractArray{T,N}) where {T,N}
	sz = size(x)
	Y = similar(x,sz[1:end-1])
	compute_grad_adjoint!(Y,x)
	return Y
end





@generated function compute_grad!(Y::AbstractArray{T,M},X::AbstractArray{T,N}) where {M,N,T}
	M != N+1 && throw(SimpleAlgebraFailure("LinOpGrad output must have one more dimensions than the input"))
	code =Expr(:block)
	push!(code.args,:(fill!(Y,zero(T))))
	for d = 1:N
		indices = generate_indices(N, d, 1)
		Id = CartesianIndex(indices...)
		push!(code.args, :( difN(get_backend(X))(Y,X,$Id,$d,ndrange = size(X) .- tuple($indices...))))
	end
	push!(code.args,:(synchronize(get_backend(X))))
	push!(code.args,:(return Y))
	return code 
end

@generated function compute_grad_adjoint!(Y::AbstractArray{T,N},X::AbstractArray{T,M}) where {M,N,T}
	N != M-1 && throw(SimpleAlgebraFailure("LinOpGrad output must have one more dimensions than the input"))
	code =Expr(:block)
	push!(code.args,:(fill!(Y,zero(T))))
	for d = 1:N
		indices = generate_indices(N, d, 1)
		Id = CartesianIndex(indices...)
		push!(code.args, :( difN_adjoint(get_backend(X))(Y,X,$Id,$d,ndrange = size(Y) .- tuple($indices...))))
	end 
	push!(code.args,:(synchronize(get_backend(X))))
	push!(code.args,:(return Y))
   return code

end

function generate_indices(num_dims, d, offset)
	indices = zeros(Int,num_dims)
	indices[d] = offset
	return indices
end



@kernel function difN_adjoint(Y, X,idx,d) 
	I = @index(Global, Cartesian)
	Y[I] += X[I,d]
	Y[I + idx] -= X[I,d]
end


@kernel function difN(Y, X,idx,d) 
	I = @index(Global, Cartesian)
	Y[I,d] = X[I] - X[I + idx]
end


function ChainRulesCore.rrule( ::typeof(apply_),A::LinOpGrad, v)
	LinOpGrad_pullback(Δy) = (NoTangent(),NoTangent(), apply_adjoint_(A, Δy))
    return  apply_(A,v), LinOpGrad_pullback
end






@generated function compute_gradient!(Y::AbstractArray{T,M},X::AbstractArray{T,N}) where {M,N,T}
	M != N+1 && throw(SimpleAlgebraFailure("LinOpGrad output must have one more dimensions than the input"))
	code =Expr(:block)
	push!(code.args,:(fill!(Y,zero(T))))
	for d = 1:N
		left = (i==d ? :(1:size(X,$i)-1) : :(Colon()) for i=1:N)
		right = (i==d ? :(2:size(X,$i)) : :(Colon()) for i=1:N)
		push!(code.args, :( @inbounds Y[$(left...),$d] .= X[$(left...)] .- X[$(right...)]))
	end
	return code 
end

# FIXME  use generated even with LoopVectorization
function compute_gradient!(Y::AbstractArray{T,2},X::AbstractArray{T,1}) where {T} 
	Y[end] = zero(T)
	t1,= size(X)
		@turbo check_empty=true warn_check_args=false for i1 = 1:(t1-1)
			Y[i1, 1] = X[i1] - X[i1 + 1]
		end
end


function compute_gradient!(Y::AbstractArray{T,3},X::AbstractArray{T,2}) where {T} 
	fill!(Y,zero(T))
	t1,t2 = size(X)
		@tturbo check_empty=true warn_check_args=false for i2 = 1:t2-1
			for i1 = 1:(t1-1)
				Y[i1, i2, 1] = X[i1, i2] - X[i1 + 1, i2]
				Y[i1, i2, 2] = X[i1, i2] - X[i1, i2 + 1]
			end
			Y[t1, i2, 2] = X[t1, i2] - X[t1, i2 + 1]
		end
		@turbo warn_check_args=false for i1 = 1:(t1-1)
			Y[i1, t2, 1] = X[i1, t2] - X[i1 + 1, t2]
		end
end

function compute_gradient!(Y::AbstractArray{T,4},X::AbstractArray{T,3}) where {T} 
	fill!(Y,zero(T))
	t1,t2,t3 = size(X)
	@tturbo check_empty=true warn_check_args=false for i3 = 1:t3-1
		for i2 = 1:t2-1
			for i1 = 1:(t1-1)
				Y[i1, i2,i3, 1] = X[i1, i2,i3] - X[i1 + 1, i2,i3]
				Y[i1, i2,i3, 2] = X[i1, i2,i3] - X[i1, i2 + 1,i3]
				Y[i1, i2,i3, 3] = X[i1, i2,i3] - X[i1,i2, i3 + 1]
			end
			Y[t1, i2,i3, 2] = X[t1, i2,i3] - X[t1, i2 + 1,i3]
			Y[t1, i2,i3, 3] = X[t1, i2,i3] - X[t1, i2 ,i3 + 1]
		end
		for i1 = 1:(t1-1)
			Y[i1, t2,i3, 1] = X[i1, t2,i3] - X[i1 + 1, t2,i3]
			Y[i1, t2,i3, 3] = X[i1, t2,i3] - X[i1, t2 ,i3 + 1]
		end
		Y[t1, t2,i3, 3] = X[t1, t2,i3] - X[t1, t2 ,i3 + 1]
	end
	@turbo warn_check_args=false for i2 = 1:t2-1
		for i1 = 1:(t1-1)
			Y[i1, i2, t3, 1] = X[i1, i2, t3] - X[i1 + 1, i2, t3]
			Y[i1, i2, t3, 2] = X[i1, i2, t3] - X[i1, i2 + 1, t3]
		end
		Y[t1, i2,t3, 2] = X[t1, i2,t3] - X[t1, i2 + 1,t3]
	end
	@turbo warn_check_args=false for i1 = 1:(t1-1)
		Y[i1, t2, t3, 1] = X[i1, t2, t3] - X[i1 + 1, t2, t3]
	end
	
end



@generated function compute_gradient_adjoint!(Y::AbstractArray{T,N},X::AbstractArray{T,M}) where {M,N,T}
	N != M-1 && throw(SimpleAlgebraFailure("LinOpGrad output must have one more dimensions than the input"))
	code =Expr(:block)
	push!(code.args,:(fill!(Y,zero(T))))
	for d = 1:N
		left = (i==d ? :(1: (size(X,$i)-1)) : :(Colon()) for i=1:N)
		right = (i==d ? :(2: (size(X,$i))) : :(Colon()) for i=1:N)
		push!(code.args, :( @inbounds  Y[$(left...)] .+= X[$(left...),$d];))
		push!(code.args, :( @inbounds   Y[$(right...)] .-= X[$(left...),$d];))
	end 
   return code
end

# FIXME  use generated even with LoopVectorization
function compute_gradient_adjoint!(Y::AbstractArray{T,1},X::AbstractArray{T,2}) where {T} 
	fill!(Y,zero(T))
	t1,= size(Y)
		 for i1 = 1:(t1-1)
			Y[i1] += X[i1,1] 
			Y[i1 + 1] -= X[i1,1]
		end
end


function compute_gradient_adjoint!(Y::AbstractArray{T,2},X::AbstractArray{T,3}) where {T} 
	fill!(Y,zero(T))
	t1,t2 = size(X)
	@turbo warn_check_args=false check_empty=true  for i2 = 1:(t2-1)
			for i1 = 1:(t1-1)
				Y[i1,i2] += X[i1,i2,1] 
				Y[i1 + 1,i2] -= X[i1,i2,1]
				Y[i1,i2] += X[i1,i2,2] 
				Y[i1,i2 + 1] -= X[i1,i2,2]
			end
			Y[t1,i2] += X[t1,i2,2] 
			Y[t1,i2 + 1] -= X[t1,i2,2]

		end
		for i1 = 1:(t1-1)
			Y[i1,t2] += X[i1,t2,1] 
			Y[i1 + 1,t2] -= X[i1,t2,1]
		end
			
end




#= 



function compute_gradientKA!(N) 
	code =Expr(:block)
	push!(code.args,:(fill!(Y,zero(eltype(Y)))))
	for d = 1:N
		push!(code.args, (difX(N,d,1)))
	end
	push!(code.args,:(synchronize(get_backend(X))))
	push!(code.args,:(return Y))
	return code 
end

function  difX(N, d, offset) 
	code =Expr(:block)
	indices =generate_indices(N, d, offset) 
	Id = CartesianIndex(indices...)
	push!(code.args,:(@kernel function ($(Symbol("dif$d")))(Y, X) 
		I = @index(Global, Cartesian)
		Y[I,$d] = X[I] - X[I + $Id]
	end))
	# workgroup size
	#wrk = zeros(Int,N)
	#wrk[d] = 512
	push!(code.args,:(($(Symbol("dif$d")))(get_backend(X))(Y,X,ndrange=(size(X) .- $(tuple(indices...))))))
	return code
end


function  difX_adjoint(N, d, offset) 
	code =Expr(:block)
	indices =generate_indices(N, d, offset) 
	Id = CartesianIndex(indices...)
	push!(code.args,:(@kernel function ($(Symbol("dif$(d)_adjoint")))(Y, X) 
		I = @index(Global, Cartesian)
		Y[I] += X[I,$d]
		Y[I + $Id] -= X[I,$d]
	end))
	# workgroup size
	#wrk = zeros(Int,N)
	#wrk[d] = 512
	push!(code.args,:(($(Symbol("dif$(d)_adjoint")))(get_backend(X))(Y,X,ndrange=(size(Y) .- $(tuple(indices...))))))
	return code
end


function compute_gradient_adjointKA!(N)
	code =Expr(:block)
	push!(code.args,:(fill!(Y,zero(eltype(Y)))))
	for d = 1:N
		push!(code.args, (difX_adjoint(N,d,1)))
	end
	push!(code.args,:(synchronize(get_backend(X))))
	push!(code.args,:(return Y))
	return code 
end



"""
Adapted from https://github.com/roflmaostc/DeconvOptim.jl/blob/master/src/regularizer.jl

    generate_indices(num_dims, dim, offset)

Generates a list of symbols which can be used to generate Tullio expressions
via metaprogramming.
`num_dims` is the total number of dimensions.
`d` is the dimension where there is a offset `offset` in the index.
.
"""
function generate_indices(num_dims, d, offset)
    # create initial symbol
    ind = :i
    # create the array of symbols for each dimension

    inds1 = map(1:num_dims) do di
        # map over numbers and append the number of the position to symbol i
        i = Symbol(ind, di)
        # at the dimension where we want to do the step, add $offset
        (di == d) && (offset!=0) ? :($i + $offset) : i
    end
    return inds1
end

function select_indices(num_dims, d, value)
    # create initial symbol
    ind = :i
    # create the array of symbols for each dimension

    inds1 = map(1:num_dims) do di
        # map over numbers and append the number of the position to symbol i
        i = Symbol(ind, di)
        # at the dimension where we want to do the step, add $offset
        (di == d)  ? :($value) : i
    end
    return inds1
end


function generate_gradient_tullio(num_dims)
	out =[]
	for d = 1:num_dims
	 	idx1 = generate_indices(num_dims, d, +1)
		idx2 = generate_indices(num_dims, d,0)
		push!(out, :(@tullio  Y[$(idx2...),$d] = X[$(idx2...)] - X[$(idx1...)];))
	end
	push!(out, :( return Y;))
	return out
    return @eval (Y,X) -> ($(out...))
end


function generate_gradient_adjoint_tullio(num_dims)
	out =[]
	for d = 1:num_dims
	 	idx1 = generate_indices(num_dims, d, +1)
		idx2 = generate_indices(num_dims, d,0)
		push!(out, :(@tullio    Y[$(idx1...)] += X[$(idx1...),$d] - X[$(idx2...),$d];))
		lfirst = select_indices(num_dims,d,1)
		llast = select_indices(num_dims,d,:end)
		push!(out, :(@tullio  Y[$(lfirst...)] += X[$(lfirst...),$d];))
		push!(out, :(@tullio  Y[$(llast...  )] += -X[$(llast...  ),$d];))
	end
	push!(out, :( return Y;))
	#return out
   	return @eval (Y,X) -> ($(out...))
end



function generate_gradient(num_dims)
	out =[]
	for d = 1:num_dims

		left = (i==d ? :(1:size(X,$i)-1) : :(Colon()) for i=1:num_dims)
		right = (i==d ? :(2:size(X,$i)) : :(Colon()) for i=1:num_dims)
		push!(out, :( @inbounds Y[$(left...),$d] .= X[$(left...)] .- X[$(right...)];))
	end 
	push!(out, :( return Y;))
#	return out
   return @eval (Y,X) -> ($(out...))
end


function generate_gradient_adjoint(num_dims)
	out =[]
	for d = 1:num_dims

		left = (i==d ? :(1: (size(X,$i)-1)) : :(Colon()) for i=1:num_dims)
		right = (i==d ? :(2: (size(X,$i))) : :(Colon()) for i=1:num_dims)
		push!(out, :(@inbounds  Y[$(left...)] .+= X[$(left...),$d];))
		push!(out, :( @inbounds  Y[$(right...)] .-= X[$(left...),$d];))
	end 
	push!(out, :( return Y;))
#	return out
   return @eval (Y,X) -> ($(out...))
end


function apply_grad(f,x::AbstractArray{T,N}) where {T,N}
	sz = size(x)
	Y = similar(x,sz...,N)
	fill!(Y,T(0))
	@invokelatest f(Y,x)
	return Y
end

function apply_grad_adjoint(f,x::AbstractArray{T,N}) where {T,N}
	sz = size(x)
	Y = similar(x,sz[1:end-1])
	fill!(Y,T(0))
	@invokelatest f(Y,x)
	return Y
end




function gradient_generated3(Y::AbstractArray{T,4},X::AbstractArray{T,3}) where {T} 
	t1,t2,t3 = size(X)
	@inbounds 	@fastmath for i3 = 1:t3 # no @turbo because of &&
		for i2 = 1:t2
			for i1 = 1:t1
				(i1 != t1) && (Y[i1, i2,i3, 1] = X[i1, i2,i3] - X[i1 + 1, i2,i3])
				(i2 != t2) && (Y[i1, i2,i3, 2] = X[i1, i2,i3] - X[i1, i2 + 1,i3])
				(i3 != t3) && (Y[i1, i2,i3, 3] = X[i1, i2,i3] - X[i1,i2, i3 + 1])
			end
		end
		
	end
end
 =#