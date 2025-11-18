import os
import argparse
from train import TrainMeanField


parser = argparse.ArgumentParser(description='Energy-Guided Diffusion for Chip Placement')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--multi_gpu', action='store_true', help='Enable multi-GPU training')
parser.add_argument('--mode', default='Diffusion', choices=["Diffusion"], help='Training mode')
parser.add_argument('--EnergyFunction', default='ChipPlacement', choices=["ChipPlacement"], help='Energy function')
parser.add_argument('--IsingMode', default='Chip_20_components',
                    choices=["Chip_dummy", "Chip_small", "Chip_medium", "Chip_large", "Chip_huge",
                             "Chip_20_components", "Chip_50_components", "Chip_100_components",
                             "Chip_v1_curriculum_stage1_n100", "Chip_v1_curriculum_stage2_n150",
                             "Chip_v1_curriculum_stage3_n200", "Chip_v1_curriculum_stage4_n250",
                             "Chip_v1_curriculum_stage5_n350"],
                    help='Training dataset configuration')
parser.add_argument('--graph_mode', default='normal', choices=["normal"], help='GNN architecture mode')
parser.add_argument('--train_mode', default='PPO', choices=["PPO"], help='Training algorithm')
parser.add_argument('--AnnealSchedule', default='linear', choices=["linear", "cosine", "exp"], help='Annealing schedule')
parser.add_argument('--temps', default=[0.], type=float, help='Temperature values', nargs="+")
parser.add_argument('--T_target', default=0., type=float, help='Target temperature')
parser.add_argument('--N_warmup', default=0, type=int, help='Number of warmup epochs')
parser.add_argument('--N_anneal', default=[2000], type=int, help='Number of training epochs', nargs="+")
parser.add_argument('--N_equil', default=0, type=int, help='Number of equilibration epochs')
parser.add_argument('--lrs', default=[3e-4], type=float, help='Learning rate values', nargs="+")
parser.add_argument('--lr_schedule', default="cosine", choices=["cosine", "None"], help='Learning rate schedule')
parser.add_argument('--seed', default=[123], type=int, help='Random seeds', nargs="+")
parser.add_argument('--GPUs', default=["0"], type=str, help='GPU device IDs', nargs="+")
parser.add_argument('--n_hidden_neurons', default=[64], type=int, help='Hidden layer dimensions', nargs="+")
parser.add_argument('--n_rand_nodes', default=2, type=int, help='Number of random node features')
parser.add_argument('--stop_epochs', default=10000, type=int, help='Maximum training epochs')
parser.add_argument('--n_diffusion_steps', default=[50], type=int, help='Number of diffusion timesteps', nargs="+")
parser.add_argument('--time_encoding', default="sinusoidal", type=str, help='Timestep encoding method')
parser.add_argument('--noise_potential', default=["gaussian"], type=str, choices=["gaussian"], help='Noise distribution', nargs="+")
parser.add_argument('--n_basis_states', default=[10], type=int, help='Number of parallel trajectories per graph', nargs="+")
parser.add_argument('--n_test_basis_states', default=20, type=int, help='Number of trajectories during evaluation')
parser.add_argument('--batch_size', default=[16], type=int, help='Batch size (number of graphs)', nargs="+")
parser.add_argument('--minib_diff_steps', default=10, type=int, help='Minibatch size for diffusion steps')
parser.add_argument('--minib_basis_states', default=5, type=int, help='Minibatch size for trajectories')
parser.add_argument('--inner_loop_steps', default=2, type=int, help='PPO inner loop iterations')
parser.add_argument('--n_GNN_layers', default=[8], type=int, help='Number of GNN message passing layers', nargs="+")
parser.add_argument('--project_name', default="", type=str, help='Wandb project name')
parser.add_argument('--beta_factor', default=[1.], type=float, help='Noise schedule factor', nargs="+")
parser.add_argument('--loss_alpha', default=0.0, type=float, help='KL divergence weight')
parser.add_argument('--mov_average', default=0.0009, type=float, help='Moving average coefficient')
parser.add_argument('--TD_k', default=3, type=float, help='GAE lambda parameter')
parser.add_argument('--clip_value', default=0.2, type=float, help='PPO clip epsilon')
parser.add_argument('--value_weighting', default=0.65, type=float, help='Value loss coefficient')
parser.add_argument('--mem_frac', default=".90", type=str, help='GPU memory fraction')
parser.add_argument('--diff_schedule', default="linear", type=str, help='Diffusion noise schedule')
parser.add_argument('--proj_method', default="None", choices=["None"], type=str, help='Projection method (unused)')
parser.add_argument('--overlap_weight', default=50.0, type=float, help='Overlap penalty weight')
parser.add_argument('--boundary_weight', default=50.0, type=float, help='Boundary penalty weight')
parser.add_argument('--overlap_threshold', default=0.1, type=float, help='Overlap normalization threshold')
parser.add_argument('--boundary_threshold', default=0.1, type=float, help='Boundary normalization threshold')
parser.add_argument('--use_normalization', action='store_true', help='Enable reward normalization')
parser.add_argument('--no-use_normalization', dest='use_normalization', action='store_false', help='Disable reward normalization')
parser.add_argument('--reward_scale', default=0.01, type=float, help='Reward scaling factor')
parser.set_defaults(use_normalization=True)
parser.add_argument('--linear_message_passing', action='store_true')
parser.add_argument('--no-linear_message_passing', dest='linear_message_passing', action='store_false')
parser.add_argument('--relaxed', action='store_true')
parser.add_argument('--no-relaxed', dest='relaxed', action='store_false')
parser.add_argument('--time_conditioning', action='store_true')
parser.add_argument('--no-time_conditioning', dest='time_conditioning', action='store_false')
parser.add_argument('--deallocate', action='store_true')
parser.add_argument('--no-deallocate', dest='time_conditioning', action='store_false')
parser.add_argument('--jit', action='store_true')
parser.add_argument('--no-jit', dest='jit', action='store_false')
parser.add_argument('--mean_aggr', action='store_true')
parser.add_argument('--no-mean_aggr', dest='mean_aggr', action='store_false')
parser.add_argument('--grad_clip', action='store_true')
parser.add_argument('--no-grad_clip', dest='grad_clip', action='store_false')
parser.add_argument('--graph_norm', action='store_true')
parser.add_argument('--no-graph_norm', dest='graph_norm', action='store_false')
parser.add_argument('--sampling-temp', default=0., type = float, help='define sampling temperature for asymptoticly unbiased estimations')
parser.add_argument('--n_sampling_rounds', default=5, type = int, help='how often the the basis states are sampled in a loop in unbiased estimations')
parser.add_argument('--bfloat16', action='store_true')
parser.add_argument('--no-bfloat16', dest='bfloat16', action='store_false')
parser.set_defaults(bfloat16=False)
parser.add_argument('--load_wandb_id', default=None, type=str, help='wandb run id to load checkpoint from (for curriculum training)')
parser.add_argument('--load_best', action='store_true', help='load best checkpoint instead of last')
parser.set_defaults(load_best=False)

parser.set_defaults(CE=False)
parser.set_defaults(graph_norm=True)
parser.set_defaults(grad_clip=True)
parser.set_defaults(mean_aggr=True)
parser.set_defaults(relaxed=True)
parser.set_defaults(time_conditioning=True)
parser.set_defaults(deallocate=False)
parser.set_defaults(jit=True)
parser.set_defaults(linear_message_passing=True)
parser.set_defaults(multi_gpu=True)
args = parser.parse_args()


def meanfield_run():
    resources_per_trial = 1.
    devices = args.GPUs
    n_workers = int(len(devices)/resources_per_trial)

    device_str = ""
    for idx, device in enumerate(devices):
        if (idx != len(devices) - 1):
            device_str += str(devices[idx]) + ","
        else:
            device_str += str(devices[idx])

    print(device_str)

    if(len(args.GPUs) > 1):
        device_str = ""
        for idx, device in enumerate(devices):
            if (idx != len(devices) - 1):
                device_str += str(devices[idx]) + ","
            else:
                device_str += str(devices[idx])

        print(device_str, type(device_str))
    else:
        device_str = str(args.GPUs[0])

    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = device_str

    nh = args.n_hidden_neurons[0]

    local_mode = args.debug

    if local_mode:
        print("Init ray in local_mode!")
    elif(args.multi_gpu):
        pass

    if(local_mode):
        import jax
        run(flexible_config = {"jit": False}, overwrite = True)
    elif(args.multi_gpu):
        detect_and_run_for_loops()

def detect_and_run_for_loops():
    nh = args.n_hidden_neurons[0]

    seeds = args.seed
    lrs = args.lrs
    N_anneals = args.N_anneal
    GNN_layers = args.n_GNN_layers
    temps = args.temps
    n_diffusion_steps = args.n_diffusion_steps
    n_basis_states = args.n_basis_states
    beta_factors = args.beta_factor
    batch_sizes = args.batch_size
    noise_potentials = args.noise_potential

    for seed in seeds:
        for lr in lrs:
            for N_anneal in N_anneals:
                for GNN_layer in GNN_layers:
                    for temp in temps:
                        for diff_steps in n_diffusion_steps:
                            for n_basis_state in n_basis_states:
                                for beta_factor in beta_factors:
                                    for batch_size in batch_sizes:
                                        for noise_potential in noise_potentials:

                                            ###checks
                                            if(args.train_mode != "REINFORCE"):
                                                if(diff_steps%args.minib_diff_steps!= 0):
                                                    raise ValueError("args.n_diffusion_steps%args.miniminib_diff_steps is not zero!")
                                                if(n_basis_state%args.minib_basis_states!= 0):
                                                    raise ValueError("args.n_basis_sates%args.minib_basis_states is not zero!")

                                                if (batch_size % len(args.GPUs) != 0):
                                                    raise ValueError("args.batch_size%len(args.GPUs) should be zero!")

                                            flexible_config = {
                                                "mode": args.mode,
                                                "dataset_name": args.IsingMode,
                                                "problem_name": args.EnergyFunction,
                                                "jit": args.jit,
                                                "wandb": True,

                                                "seed": seed,
                                                "lr": lr,

                                                "random_node_features": True,
                                                "n_random_node_features": args.n_rand_nodes,
                                                "relaxed": args.relaxed,
                                                "T_max": temp,
                                                "N_warmup": args.N_warmup,
                                                "N_anneal": N_anneal,
                                                "N_equil": args.N_equil,
                                                "n_hidden_neurons": nh,
                                                "n_features_list_prob": [2],
                                                "n_features_list_nodes": [nh, nh],
                                                "n_features_list_edges": [nh, nh],
                                                "n_features_list_messages": [nh, nh],
                                                "n_features_list_encode": [nh, nh],
                                                "n_features_list_decode": [nh, nh],
                                                "n_message_passes": GNN_layer,
                                                "message_passing_weight_tied": False,
                                                "n_diffusion_steps": diff_steps,
                                                "N_basis_states": n_basis_state,
                                                "batch_size": batch_size,
                                                "beta_factor": beta_factor,
                                                "stop_epochs": args.stop_epochs,
                                                "noise_potential": noise_potential,
                                                "time_conditioning": args.time_conditioning,
                                                "project_name": args.project_name,
                                                "linear_message_passing": args.linear_message_passing,

                                                "n_random_node_features": args.n_rand_nodes,
                                                "mean_aggr": args.mean_aggr,
                                                "grad_clip": args.grad_clip,
                                                "graph_mode": args.graph_mode,
                                                "loss_alpha": args.loss_alpha,
                                                "train_mode": args.train_mode,

                                                "inner_loop_steps": args.inner_loop_steps,
                                                "minib_diff_steps": args.minib_diff_steps,
                                                "minib_basis_states": args.minib_basis_states,
                                                "graph_norm": args.graph_norm,
                                                "proj_method": args.proj_method,
                                                "diff_schedule": args.diff_schedule,
                                                "mov_average": args.mov_average,
                                                "sampling_temp": args.sampling_temp,
                                                "n_sampling_rounds": args.n_sampling_rounds,
                                                "n_test_basis_states": args.n_test_basis_states,
                                                "bfloat16": args.bfloat16,
                                                "T_target": args.T_target,
                                                "AnnealSchedule": args.AnnealSchedule,
                                                "time_encoding": args.time_encoding,
                                                "lr_schedule": args.lr_schedule,
                                                "TD_k": args.TD_k,
                                                "clip_value": args.clip_value,
                                                "value_weighting": args.value_weighting,
                                                "overlap_weight": args.overlap_weight,
                                                "boundary_weight": args.boundary_weight,
                                                "overlap_threshold": args.overlap_threshold,
                                                "boundary_threshold": args.boundary_threshold,
                                                "use_normalization": args.use_normalization,
                                                "reward_scale": args.reward_scale,
                                            }

                                            flexible_config["continuous_dim"] = 2

                                            run(flexible_config=flexible_config, overwrite=True)




def run( flexible_config, overwrite = True):

    config = {
        "mode": "Diffusion",  # either Diffusion or MeanField
        "dataset_name": "RB_iid_small",
        "problem_name": "MIS",
        "jit": True,
        "wandb": True,

        "seed": 123,
        "lr": 1e-4,
        "batch_size": 30, # H
        "N_basis_states": 30, # n_s

        "random_node_features": True,
        "n_random_node_features": 5,
        "relaxed": True,

        "T_max": 0.05,
        "N_warmup": 0,
        "N_anneal": 2000,
        "N_equil": 0,
        "stop_epochs": 800,

        "n_hidden_neurons": 64,
        "n_features_list_prob": [64, 2],
        "n_features_list_nodes": [64, 64],
        "n_features_list_edges": [10],
        "n_features_list_messages": [64, 64],
        "n_features_list_encode": [30],
        "n_features_list_decode": [64],
        "n_message_passes": 2,
        "message_passing_weight_tied": False,
        "linear_message_passing": True,
        "edge_updates": False,
        "n_diffusion_steps": 1,
        "beta_factor": 0.1,
        "noise_potential": "annealed_obj",

        "time_conditioning": True,

        "project_name": args.project_name,
        "mean_aggr": False,
        "grad_clip": True,
        "messeage_concat": False,
        "graph_mode": "normal",
        "loss_alpha": 0.0,
        "train_mode": "REINFORCE",
        "inner_loop_steps": 2,
        "minib_diff_steps": 3,
        "minib_basis_states": 10,
        "graph_norm": False,
        "proj_method": "None",
        "diff_schedule": "DiffUCO",
        "mov_average": 0.05,
        "sampling_temp": 1.4,
        "n_sampling_rounds": 5,
        "n_test_basis_states": 20,
        "bfloat16": False,
        "T_target": 0.,
        "AnnealSchedule": "linear",
        "time_encoding": "one_hot",
        "lr_schedule": "cosine",
        "TD_k": 3,
        "clip_value": 0.2,
        "value_weighting": 0.65,
        "continuous_dim": 0,  # Default 0 for discrete; set to 2 for ChipPlacement
        "overlap_weight": 2000.0,  # ChipPlacement: overlap penalty weight (use 50-100 with normalization!)
        "boundary_weight": 2000.0,  # ChipPlacement: boundary penalty weight (use 50-100 with normalization!)
        "use_constraints_in_training": False,  # ChipPlacement: two-stage approach (train HPWL+spread, eval with constraints)
        "spread_weight": 0.5,  # ChipPlacement: spread regularization weight (prevents stacking in HPWL-only training)
        "use_min_distance": False,  # ChipPlacement: use minimum distance constraint (alternative to spread)
        "min_distance_weight": 0.1,  # ChipPlacement: minimum distance penalty weight
        "min_distance_threshold": 0.2,  # ChipPlacement: minimum distance threshold
        "use_normalization": True,  # ChipPlacement: use normalization (TSP-style, default True)
        "reward_scale": 0.01,  # ChipPlacement: reward scaling when normalization disabled
        "grid_width": 10,  # GridChipPlacement: grid width (10x10 = 100 cells)
        "grid_height": 10,  # GridChipPlacement: grid height
        "collision_weight": 1.5,  # GridChipPlacement: collision penalty weight (TSP-style)
    }

    if(overwrite):
        for key in flexible_config:
            if(key in config.keys()):
                config[key] = flexible_config[key]
            else:
                raise ValueError("key does not exist")

    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = str(args.mem_frac)
    if(args.deallocate):
        os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"

    train = TrainMeanField(config,
                          load_wandb_id=args.load_wandb_id,
                          load_best_parameters=args.load_best)

    train.train()





if __name__ == "__main__":
    meanfield_run()
