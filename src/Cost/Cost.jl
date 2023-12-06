
abstract type AbstractCost{I}  <: AbstractMap{I,Scalar{Real}}  end



																
struct CostComposition{I,Left<:AbstractCost,Right<:AbstractMap} <:  AbstractCost{I}
	left::Left
	right::Right
	function CostComposition(A::Dleft, B::Dright) where {Ileft, I2, O2,Dleft<:AbstractCost{Ileft},  Dright<:AbstractMap{I2,O2}} 
		    return new{I2, Dleft,Dright}(A,B)
	end
end
@functor CostComposition

outputspace(::AbstractCost) = Scalar{Real}()
inputspace(A::CostComposition)  = inputspace(A.right)

#= compose(A::AbstractCost{Ileft}, B::AbstractMap{Iright,Oright}) where {N,Tleft,
																Tright<:Tleft,
																Oright<:CoordinateSpace{Tright,N},
																Ileft<:CoordinateSpace{Tleft,N},
																Iright} = CostComposition(A, B) 
 =#																
#= compose(A::AbstractCost{IA}, B::AbstractMap{IB,OB}) where 
			{N,TA,TB,IA<:CoordinateSpace{N,TA},OB<:CoordinateSpace{N,TB},IB<:AbstractDomain} = 
				CostComposition(A, B) 

compose(::AbstractCost, ::AbstractMap) = throw(SimpleAlgebraFailure("The output of the Map does not match the input of the cost"))
 =#
compose(A::AbstractCost, B::AbstractMap) =CostComposition(A, B) 

apply_(A::CostComposition, v) = apply(A.left,apply(A.right,v))


include("./CostL2.jl")
include("./CostHyperbolic.jl")