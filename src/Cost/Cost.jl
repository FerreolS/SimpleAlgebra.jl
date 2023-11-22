
abstract type AbstractCost{I}  <: AbstractMap{I,Scalar{Real}}  end

outputspace(::AbstractCost) = Scalar{Real}()


																
struct CostComposition{I,Left<:AbstractCost,Right<:AbstractMap} <:  AbstractCost{I}
	left::Left
	right::Right
	function CostComposition(A::Dleft, B::Dright) where {Ileft, I2, O2,Dleft<:AbstractCost{Ileft},  Dright<:AbstractMap{I2,O2}} 
		    return new{I2, Dleft,Dright}(A,B)
	end
end
@functor CostComposition


#= compose(A::AbstractCost{Ileft}, B::AbstractMap{Iright,Oright}) where {N,Tleft,
																Tright<:Tleft,
																Oright<:CoordinateSpace{Tright,N},
																Ileft<:CoordinateSpace{Tleft,N},
																Iright} = CostComposition(A, B) 
 =#																
compose(A::AbstractCost, B::AbstractMap) = CostComposition(A, B) 

compose(::AbstractCost{O}, ::AbstractMap{I,O}) where {I,O} 	= throw(SimpleAlgebraFailure("The output of the Map does not match the input of the cost"))

inputspace(A::CostComposition)  = inputspace(A.right)

apply_(A::CostComposition, v) = apply(A.left,apply(A.right,v))


include("./CostL2.jl")
