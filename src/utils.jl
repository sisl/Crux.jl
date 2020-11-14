sdim(mdp) = length(convert_s(AbstractVector, rand(initialstate(mdp)), mdp))
adim(mdp) = length(actions(mdp))

## GPU Stuff
function CUDA.copyto!(Cto::Chain, Cfrom::Chain)
    for i = 1:length(Flux.params(Cto).order.data)
        copyto!(Flux.params(Cto).order.data[i], Flux.params(Cfrom).order.data[i])
    end
end


function globalnorm(ps::Flux.Params, gs::Flux.Zygote.Grads)
   gnorm = 0f0
   for p in ps 
       gs[p] === nothing && continue 
       curr_norm = maximum(abs.(gs[p]))
       gnorm =  curr_norm > gnorm  ? curr_norm : gnorm
   end 
   return gnorm
end

