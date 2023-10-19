# diffusion_tp_generation

This repo contains the full pipeline for generating Mueller potential transition paths used in the paper "Diffusion Methods for Generating Transition Paths". This paper has been submitted to IOP - Machine Learning: Science and Technology. 

Transition paths can be generated using Euler-Maruyama sampling in transition_path_sampling. The code for training a score network and generating samples using the reverse SDE can be found in transition_path_generation.

This code was developed by Luke Triplett: luke.triplett@duke.edu, under the supervision of Jianfeng Lu: jianfeng@math.duke.edu.