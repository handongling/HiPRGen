from HiPRGen.network_loader import *
from HiPRGen.species_questions import *
from HiPRGen.mc_analysis import *

network_loader = NetworkLoader(
    './scratch/li_test/rn.sqlite',
    './scratch/li_test/mol_entries.pickle',
    './scratch/li_test/initial_state.sqlite',
)

ec = network_loader.mol_entries[160]

network_loader.load_trajectories()
network_loader.load_initial_state()
seeds = network_loader.trajectories.keys()
sr = SimulationReplayer(network_loader)