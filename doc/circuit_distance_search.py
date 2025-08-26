# This is the script that includes the parameters to explore effective distance. The results are given in our paper arXiv:2504.02673
# We do not encourage running this script on laptops, unless parameter dont_explore_edges _increasing_symptom_degree is set to True 
# For BPC code [[144,8,12]], we requested 100G memory on Duke computer cluster.
from quits.circuit import *
from quits.simulation import get_stim_mem_result
from quits.decoder import sliding_window_circuit_mem, detector_error_model_to_matrix
from quits.qldpc_code import *
import stim

#[[72,8,8]]
factor=3
lift_size = 12
p1=[0,1,5]
p2=[0,1,8]

code = BpcCode(p1, p2, lift_size, factor)  # Define the BpcCode object
code.build_graph(seed=1)                   # Build the Tanner graph and assign directions to its edges.

p = 2e-3           # physical error rate, does not matter in circuit distance search
num_rounds = 2    # number of rounds (T-1)
basis = 'Z'        

circuit = stim.Circuit(get_qldpc_mem_circuit(code, p, p, p, p, num_rounds, basis=basis))
model = circuit.detector_error_model(decompose_errors=False)
detector_error_matrix, observables_matrix, priors= detector_error_model_to_matrix(model)

err_list = circuit.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=6,
                                                     dont_explore_edges_with_degree_above=6,
                                                     dont_explore_edges_increasing_symptom_degree=False) #Return a combination of error mechanisms that create a non detectable logical error
#see also stim documentation: https://github.com/quantumlib/Stim/blob/main/doc/python_api_reference_vDev.md#stim.Circuit.search_for_undetectable_logical_errors

print(len(err_list))


#[[90,8,10]]
factor=3
lift_size = 15
p1=[0,1,5]
p2=[0,2,7]

code = BpcCode(p1, p2, lift_size, factor)  # Define the BpcCode object
code.build_graph(seed=1)                   # Build the Tanner graph and assign directions to its edges.

p = 2e-3           # physical error rate, does not matter in circuit distance search
num_rounds = 2    # number of rounds (T-1)
basis = 'Z'        

circuit = stim.Circuit(get_qldpc_mem_circuit(code, p, p, p, p, num_rounds, basis=basis))
model = circuit.detector_error_model(decompose_errors=False)
detector_error_matrix, observables_matrix, priors= detector_error_model_to_matrix(model)

err_list = circuit.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=6,
                                                     dont_explore_edges_with_degree_above=6,
                                                     dont_explore_edges_increasing_symptom_degree=False) #Return a combination of error mechanisms that create a non detectable logical error

print(len(err_list))


#[[144,8,12]]
factor=3
lift_size = 24
p1=[0,1,5]
p2=[0,1,11]

code = BpcCode(p1, p2, lift_size, factor)  # Define the BpcCode object
code.build_graph(seed=1)                   # Build the Tanner graph and assign directions to its edges.

p = 2e-3           # physical error rate, does not matter in circuit distance search
num_rounds = 1    # number of rounds (T-1)
basis = 'Z'        

circuit = stim.Circuit(get_qldpc_mem_circuit(code, p, p, p, p, num_rounds, basis=basis))
model = circuit.detector_error_model(decompose_errors=False)
detector_error_matrix, observables_matrix, priors= detector_error_model_to_matrix(model)

err_list = circuit.search_for_undetectable_logical_errors(dont_explore_detection_event_sets_with_size_above=6,
                                                     dont_explore_edges_with_degree_above=6,
                                                     dont_explore_edges_increasing_symptom_degree=False) #Return a combination of error mechanisms that create a non detectable logical error
print(len(err_list))
