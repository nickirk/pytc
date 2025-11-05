# Add at the top of your script, BEFORE importing JAX
import os
from line_profiler import LineProfiler
import unittest
import io
from pytc.autodiff.test.test_mcmc import TestHartreeFockEnergy
from pytc.autodiff.mcmc import metropolis_hastings, sample, metropolis_hastings_importance_sampling
from pytc.autodiff.mcmc import burn_in_with_importance, metropolis_hastings, _one_electron_move
from pytc.autodiff.ansatz.det import SlaterDet, _update_grad_lap_rows
from pytc.autodiff.ansatz.sj import SlaterJastrow
from pyscf.dft import numint  # Import numint to profile AO evaluations

# Create test instance
test = TestHartreeFockEnergy()

# Create line profiler
profile = LineProfiler()

# Add functions to profile
profile.add_function(test.run_hf_energy_test)
profile.add_function(metropolis_hastings)
profile.add_function(sample)
profile.add_function(metropolis_hastings)
profile.add_function(_one_electron_move)
profile.add_function(_update_grad_lap_rows)


# Add SlaterDet functions that are likely bottlenecks
profile.add_function(SlaterDet.__call__)
profile.add_function(SlaterDet.value_and_grad)
profile.add_function(SlaterDet.matrix)
profile.add_function(SlaterDet.grad)
profile.add_function(SlaterDet.laplacian)
profile.add_function(SlaterDet.ao2mo)
profile.add_function(SlaterJastrow.local_energy)

# Add the PySCF numint.eval_ao function which is the deepest bottleneck
profile.add_function(numint.eval_ao)

# Run the profiled test
profile.runcall(test.test_benzene)

# Print results to console
profile.print_stats()

# To save results to a file, use dump_stats instead
profile.dump_stats('profile_results.lprof')
print("Profile data saved to 'profile_results.lprof'")
print("View it with: python -m line_profiler profile_results.lprof")

# Alternative: capture output and write to file
with open('profile_results.txt', 'w') as f:
    # Redirect output to a string buffer
    s = io.StringIO()
    profile.print_stats(stream=s)
    f.write(s.getvalue())
    print("Text report saved to 'profile_results.txt'")
