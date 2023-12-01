using PyPlot
using PyCall
### set some reasonable plotting defaults
rc("font"; size=12)
PyCall.PyDict(matplotlib."rcParams")["axes.spines.top"] = false
PyCall.PyDict(matplotlib."rcParams")["axes.spines.right"] = false

function plot_progress(losses, t_rewds, f_pred_next_locs, f_plannings; fname="figs/progress.png")
    ts = 1:length(losses)
    data = [losses, t_rewds, f_pred_next_locs, f_plannings]
    labs = ["losses","reward", "prediction sʼ", "fraction of planning"]
    fig, axs = plt.subplots(2,2)
    for i in 1:4
        axs[i].plot(ts, data[i])
        axs[i].set_xlabel("epochs")
        axs[i].set_ylabel(labs[i])
        axs[i].set_title(labs[i])
    end
    tight_layout()
    plt.savefig(fname; bbox_inches="tight")
    close()
end

