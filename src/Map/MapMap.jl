struct MapMap{I,O,F} <:  AbstractMap{I,O} 
	inputspace::I
	outputspace::O
	func::F
	MapMap(inputspace::I,outputspace::O,func::F)  where {I<:CoordinateSpace,O<:CoordinateSpace,F<:Union{Function,AbstractMap}} = 
		new{I,O,F}(inputspace,outputspace,func)
end

MapMap(func, numel::Int) = MapMap(func, (numel,))
MapMap(func, sz::NTuple) = MapMap(Number,func, sz)
MapMap(::Type{TI},func,sz::NTuple) where TI  = MapMap(CoordinateSpace(TI,sz),func)
MapMap(inputspace::CoordinateSpace,func::Function) = MapMap(inputspace,inputspace,func) 
function MapMap(inspace::CoordinateSpace,map::AbstractMap)
	in = cat(inputspace(map),inspace)
	out = cat(outputspace(map),inspace)
    MapMap(in,out,map)
end 


@functor MapMap

function apply_(A::MapMap, x)  
	y = similar(x, outputspace(A))
	return computeMapMap!(y,A.func, x)
end
function apply_!(y,A::MapMap, x)  
	return computeMapMap!(y,A.func, x)
end

function computeMapMap!(y,map::AbstractMap, x)  
	ndrange = size(x)[(ndims(inputspace(map))+1):end]
	backend =get_backend(x)
	MapMap_kernel(backend,64)(y,x,map, ndrange=ndrange)
	synchronize(backend)
	return y
end

function computeMapMap!(y,map::Function, x)  
	ndrange = size(x)[2:end]
	backend =get_backend(x)
	MapMap_kernel(backend)(y,x,map.apply_!, ndrange=ndrange)
	synchronize(backend)
	return y
end

@kernel function MapMap_kernel(Y, X,F) 
	I = @index(Global, Cartesian)
	Y[..,I] .=  F(X[..,I])
end

function apply_jacobian_(A::MapMap, v, x)
	y = similar(x, inputspace(A))
	computeMapMapjacobian!(y,A.func, v,x)
end

function computeMapMapjacobian!(y,map::AbstractMap, v, x)  
	ndrange = size(x)[(ndims(inputspace(map))+1):end]
	backend =get_backend(x)
	MapMapJacobian_kernel(backend)(y,v, x,map.apply_jacobian_, ndrange=ndrange)
	synchronize(backend)
end


@kernel function MapMap_kernel(Y, V, X,F) 
	I = @index(Global, Cartesian)
	Y[..,I] .=  F(V[..,I],X[..,I])
end


#stack( G.(eachslice(x,dims=(2,3))))