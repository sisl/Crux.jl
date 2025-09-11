module WBLoggerExt

using Crux, TensorBoardLogger, WeightsAndBiasLogger

function TensorBoardLogger.log_value(logger::WBLogger, name::AbstractString, value::Real; step=0)
    info_dict = Dict(string(name) => value)
    wandb.log(info_dict, step=step)
end

function Crux.log_fig(::WBLogger, filename)
    vid = wandb.Video(filename, format="gif")
    wandb.log(Dict("gif"=>vid))
    rm(filename)
end

function Crux.build_logger(; use_wandb, config, project, entity, notes, dir)
    if use_wandb
        return WBLogger(config=config, project=project, entity=entity, notes=notes)
    else
        return TBLogger(dir, tb_increment)
    end
end

end
