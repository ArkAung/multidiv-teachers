## Active Learning from Divergent Multiple Teachers

The idea behind "Active Learning from Divergent Multiple Teachers" is to construct multiple teacher neural networks with different architectures, each potentially producing varying softmax results for an input $X$. Then, KL divergence is applied to all possible combinations of these softmax outputs. After that, mean of the all the KL-divergences is calculated. The samples with highest mean KL-divergence is chosen as a prime candidate for being sent to oracle for being labelled. Those samples with low KL-divergence are considered as confident samples and multiple softmax probability distributions are treated as multiple soft-labels for the student network. The student network learns from this multi-label distribution. During inference time, the student network looks outputs with highest agreement.

### Pipeline

* Train $N$ neural networks on training data
** Add temperature layer right before softmax layer
* Use those networks to make predictions on unlabelled data
* Get the softmax values of unlabelled from those networks
* Use multi-teacher knowledge distillation to train a student network for faster inference time, using the softmax outputs of training data from teacher network works
* Calculate the mean divergences among all possible combinations of softmax pairs from $N$ networks. There will be $N_c_2$ combinations
* Get the top $100$ data points with highest divergence to be sent to oracle.
** This can be elaborated into better selection process of cutting the number of data points which are above certain divergence threshold.
* 
