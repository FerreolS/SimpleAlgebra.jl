struct MapGaussian1D{I,O,D<:AbstractArray,N} <:  AbstractMap{I,O} 
	inputspace::I
	outputspace::O
	abscisse::D
	MapGaussian1D(inputspace::I,outputspace::O,x::D)  where {I<:CoordinateSpace,O<:CoordinateSpace,D<:AbstractArray} = 
		new{I,O,D,length(inputspace)}(inputspace,outputspace,x)
end

@functor MapGaussian1D

MapGaussian1D(abscisse)  = MapGaussian1D(abscisse,3)
MapGaussian1D(abscisse,nbparameter)  = MapGaussian1D(Number, abscisse, nbparameter)
function MapGaussian1D(::Type{TI},x::D,nbparameter::Int) where {TI,D<:AbstractArray}  
	inputspace = CoordinateSpace(TI,(nbparameter,))
	outputspace = CoordinateSpace(TI,size(x))
	MapGaussian1D(inputspace,outputspace,x)
end



apply_(A::MapGaussian1D, x) = Gaussian1D(A,x...)
apply_!(y,A::MapGaussian1D, x) = Gaussian1D!(y,A,x...)
# ... is not differentiable 	see https://github.com/FluxML/Zygote.jl/issues/599
# The next lines patche this issue waiting for the fix
apply_(A::MapGaussian1D{I,O,D,1} , x) where {I,O,D} = Gaussian1D(A,x[1])
apply_!(y,A::MapGaussian1D{I,O,D,1} , x) where {I,O,D} = Gaussian1D!(y,A,x[1])
apply_(A::MapGaussian1D{I,O,D,2} , x) where {I,O,D} = Gaussian1D(A,x[1],x[2])
apply_!(y,A::MapGaussian1D{I,O,D,2} , x) where {I,O,D} = Gaussian1D!(y,A,x[1],x[2])
apply_(A::MapGaussian1D{I,O,D,3} , x) where {I,O,D} = Gaussian1D(A,x[1],x[2],x[3])
apply_!(y,A::MapGaussian1D{I,O,D,3} , x) where {I,O,D} = Gaussian1D!(y,A,x[1],x[2],x[3])

# FIXME error    
apply_jacobian_(A::MapGaussian1D, x,v) = JGaussian1D(A.abscisse,x...,v)
function JGaussian1D( x, fwhm::T, center::T, amplitude,Δp) where T
	a = T(2/ (2 * sqrt(2 * log(2.))))
	halfprecision =  inv((a*fwhm)^2)	
    r = (x .- center)
	g = @. exp(-(r)^2 * halfprecision)
	m = (@. 2 * amplitude * halfprecision * r * g)
	fwhm_pullback  = -2/a^2 / fwhm^3 .* (sum((@. -amplitude * r^2 * g).* Δp))
	center_pullback = (sum( m .*Δp))
	amplitude_pullback = (sum(g.* Δp))
	[fwhm_pullback, center_pullback,amplitude_pullback]
end


function JGaussian1D(x, fwhm::T, center::T,Δp) where T
	amplitude = T(1)
	a = T(2/ (2 * sqrt(2 * log(2.))))
	halfprecision =  inv((a*fwhm)^2)	
    r = (x .- center)
	g = @. exp(-(r)^2 * halfprecision)
	m = (@. 2 * amplitude * halfprecision * r * g)
	fwhm_pullback  = -2/a^2 / fwhm^3 .* (sum((@. -amplitude * r^2 * g).* Δp))
	center_pullback = (sum( m .*Δp))
	[fwhm_pullback, center_pullback]
end
 


function JGaussian1D(x, fwhm::T,Δp) where T
	amplitude = T(1)
	center = T(0)
	a = T(2/ (2 * sqrt(2 * log(2.))))
	halfprecision =  inv((a*fwhm)^2)	
    r = (x .- center)
	g = @. exp(-(r)^2 * halfprecision)
	return fwhm_pullback  = -2/a^2 / fwhm^3 .* (sum((@. -amplitude * r^2 * g).* Δp))
	
end


function compute_Gaussian1D!(y,x,halfprecision::T, center::T, amplitude) where T	
	return @. y = amplitude * exp(-(x - center)^2 * halfprecision)
end

function compute_Gaussian1D(x,fwhm::T, center::T, amplitude) where T	
	halfprecision =  inv((T(2/ (2 * sqrt(2 * log(2.))))*fwhm)^2)
	return   @. amplitude * exp(-(x - center)^2 * halfprecision)
end

Gaussian1D(A::MapGaussian1D, fwhm::T, center::T) where T<:Number  = Gaussian1D(A,fwhm,center,1)
Gaussian1D(A::MapGaussian1D, fwhm::T) where T<:Number = Gaussian1D(A,fwhm,T(0),1)
function Gaussian1D(A::MapGaussian1D, fwhm::T, center::T, amplitude) where T
	halfprecision =  inv((T(2/ (2 * sqrt(2 * log(2.))))*fwhm)^2)	
	y = similar(A.abscisse,T)
	return compute_Gaussian1D!(y,A.abscisse,halfprecision, center, amplitude)

end

Gaussian1D!(y,A::MapGaussian1D, fwhm::T, center::T) where T<:Number  = Gaussian1D!(y,A,fwhm,center,1)
Gaussian1D!(y,A::MapGaussian1D, fwhm::T) where T<:Number = Gaussian1D!(y,A,fwhm,T(0),1)
function Gaussian1D!(y,A::MapGaussian1D, fwhm::T, center::T, amplitude) where T
	halfprecision = inv((T(2/ (2 * sqrt(2 * log(2.))))*fwhm)^2)
	return 	compute_Gaussian1D!(y,A.abscisse,halfprecision, center, amplitude)
end


function ChainRulesCore.rrule( ::typeof(compute_Gaussian1D!),y,x,halfprecision, center, amplitude)
	r = (x .- center)
	g = @. exp(-(r)^2 * halfprecision)
	@. y = amplitude*g 
	m = @thunk(@. 2 * amplitude * halfprecision * r * g)
	x_pullback(Δp)  = m.*Δp
	halfprecision_pullback(Δp)  = (sum(@thunk(@. -amplitude * r^2 * g).* Δp))
	center_pullback(Δp) = (sum( m .*Δp))
	amplitude_pullback(Δp) = (sum(g.* Δp))

    return y,Δp->(NoTangent(),NoTangent(),x_pullback(Δp), halfprecision_pullback(Δp), center_pullback(Δp), amplitude_pullback(Δp))
end



function ChainRulesCore.rrule( ::typeof(apply),A::MapGaussian1D{I,O,D,1} , x::AbstractVector{T}) where {T,I,O,D}
	fwhm=x[1] 
	center = T(0) 
	amplitude = T(1)
	y,fwhm_pullback,_, _ = innerrrule(A.abscisse,fwhm,center,amplitude)
    return y,Δp->(NoTangent(),NoTangent(), fwhm_pullback(Δp))
end

function ChainRulesCore.rrule( ::typeof(apply),A::MapGaussian1D{I,O,D,2} , x::AbstractVector{T}) where {T,I,O,D}
	fwhm, center =x
	amplitude = T(1)
	y,fwhm_pullback,center_pullback, _ = innerrrule(A.abscisse,fwhm,center,amplitude)
	all_pullback(Δp) = [fwhm_pullback(Δp),center_pullback(Δp)]
    return y,Δp->(NoTangent(),NoTangent(), all_pullback(Δp))
end

function ChainRulesCore.rrule( ::typeof(apply),A::MapGaussian1D{I,O,D,3} , x::AbstractVector{T}) where {T,I,O,D}
	fwhm, center,	amplitude = x
	y,fwhm_pullback,center_pullback, amplitude_pullback = innerrrule(A.abscisse,fwhm,center,amplitude)
	all_pullback(Δp) = [fwhm_pullback(Δp),center_pullback(Δp),amplitude_pullback(Δp)]

    return y,Δp->(NoTangent(),NoTangent(), all_pullback(Δp))
end

function innerrrule(x,fwhm::T,center,amplitude) where T
	a = T(2/ (2 * sqrt(2 * log(2.))))
	halfprecision =  inv((a*fwhm)^2)	
	r = (x .- center)
	g = @. exp(-(r)^2 * halfprecision)
	y = @.  amplitude*g 
	m = @thunk(@. 2 * amplitude * halfprecision * r * g)
	x_pullback(Δp)  = m.*Δp
	fwhm_pullback(Δp)  = -2/a^2 / fwhm^3 .* (sum(@thunk(@. -amplitude * r^2 * g).* Δp))
	center_pullback(Δp) = (sum( m .*Δp))
	amplitude_pullback(Δp) = (sum(g.* Δp))
	return y,fwhm_pullback,center_pullback, amplitude_pullback
end

#= 
function compute_Gaussian1D(x,halfprecision, center,amplitude)
	#x = -100:0.01:100
	#center=0
	#amplitude=1
	#halfprecision = 1
	r = (x .- center)
	g = @. exp(-(r)^2 * halfprecision)
	return y = @. amplitude*g 
end


function ChainRulesCore.rrule( ::typeof(compute_Gaussian1D),x, halfprecision, center, amplitude)

	# x = -100:0.01:100
	#center=0
	#amplitude=1
	#halfprecision = 1
	r = (x .- center)
	g = @. exp(-(r)^2 * halfprecision)
	y = @. amplitude*g 

	#function pullback_Gaussian1D(Δp)
		
	m = @thunk(@. 2 * amplitude * halfprecision * r * g)
	x_pullback(Δp)  = -m.*Δp
		halfprecision_pullback(Δp)  = (sum(-amplitude .* r.^2 .* g.* Δp))
		center_pullback(Δp) = (sum( 2 .* amplitude .* halfprecision .* r .* g.*Δp))
		amplitude_pullback(Δp) = (sum(g.* Δp))
    #	return (NoTangent(), center_pullback, amplitude_pullback)
	#end 
	#f(Δp) =  (NoTangent(),sum(g.* Δp))

    return y,Δp->(NoTangent(), x_pullback(Δp), halfprecision_pullback(Δp), center_pullback(Δp), amplitude_pullback(Δp))
end =#