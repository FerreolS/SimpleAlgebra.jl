
using Tullio

"""
Adapted from https://github.com/roflmaostc/DeconvOptim.jl/blob/master/src/regularizer.jl

    generate_indices(num_dims, dim, offset)

Generates a list of symbols which can be used to generate Tullio expressions
via metaprogramming.
`num_dims` is the total number of dimensions.
`dim` is the dimension where there is a offset `offset` in the index.
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

#= 

function generate_gradient_adjoint(num_dims)
	out =[]
	for d = 1:num_dims
	 	idx1 = generate_indices(num_dims, d, +1)
		idx2 = generate_indices(num_dims, d,0)
		push!(out, :(@tullio    Y[$(idx1...)] += X[$(idx1...),$d] - X[$(idx2...),$d];))
		lfirst = (i==d ? 1 : :(Colon()) for i=1:num_dims)
		lend = (i==d ? :(size(X,$i)) : :(Colon()) for i=1:num_dims)
		push!(out, :(  Y[$(lfirst...)] .+= X[$(lfirst...),$d]))
		push!(out, :(  Y[$(lend...)] .-= X[$(lend...),$d]))
	end
   	return @eval (Y,X) -> ($(out...))
end =#


function generate_gradient(num_dims)
	out =[]
	for d = 1:num_dims

		left = (i==d ? :(1:size(X,$i)-1) : :(Colon()) for i=1:num_dims)
		right = (i==d ? :(2:size(X,$i)) : :(Colon()) for i=1:num_dims)
		push!(out, :(Y[$(left...),$d] .= X[$(left...)] .- X[$(right...)];))
	end 
	push!(out, :( return Y;))
   return @eval (Y,X) -> ($(out...))
end


function generate_gradient_adjoint(num_dims)
	out =[]
	for d = 1:num_dims

		left = (i==d ? :(1: (size(X,$i)-1)) : :(Colon()) for i=1:num_dims)
		right = (i==d ? :(2: (size(X,$i))) : :(Colon()) for i=1:num_dims)
		push!(out, :( Y[$(left...)] .+= X[$(left...),$d];))
		push!(out, :(  Y[$(right...)] .-= X[$(left...),$d];))
	end 
	push!(out, :( return Y;))
   return @eval (Y,X) -> ($(out...))
end


function apply_gradient(f,x::AbstractArray{T,N}) where {T,N}
	sz = size(x)
	Y = similar(x,sz...,N)
	fill!(Y,T(0))
	f(Y,x)
	return Y
end

function apply_gradient_adjoint(f,x::AbstractArray{T,N}) where {T,N}
	sz = size(x)
	Y = similar(x,sz[1:end-1])
	fill!(Y,T(0))
	f(Y,x)
	return Y
end
