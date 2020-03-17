## Active Learning from Divergent Multiple Teachers

The idea behind "Active Learning from Divergent Multiple Teachers" is to construct multiple teacher neural networks with different architectures, each potentially producing varying softmax results for an input $X$. Then, KL divergence is applied to all possible combinations of these softmax outputs. After that, mean of the all the KL-divergences is calculated. The samples with highest mean KL-divergence is chosen as a prime candidate for being sent to oracle for being labelled. Those samples with low KL-divergence are considered as confident samples and multiple softmax probability distributions are treated as multiple soft-labels for the student network. The student network learns from this multi-label distribution. During inference time, the student network looks outputs with highest agreement.

This paradigm of distilling multiple parent architectures is known as multi-distillation.
