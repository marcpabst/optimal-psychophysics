# Optimal Adaptive Psychometrics

This package provides a Rust implementation of different ways to obtain  estimates for real-time adaptive psychometric testing. This includes parameter estimation and Expected Information Gain (EIG) calculations to select the next stimulus or item to present to the test taker.

# Goals
- Speed: High perfomance for small sample sizes and real-time applications ideally using SIMD instructions where possible
- Correctness: Accurate and reliable estimates
- Simplicity: Easy to use and understand
- Portability: SHould work on most

# Non-goals
- Scalability: Not designed for big datasets (i.e., no support for GPUs or distributed computing)
- Flexibility: Not designed for complex models or (exploratory) data analysis - you should probbaly use PyMC3 or Stan instead
