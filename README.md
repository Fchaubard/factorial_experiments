# factorial_experiments
This is a research repo for getting an LLM to learn factorial. 

LLMs are hypothesized to have the capacity to map between ANY arbitrary sequences (seq2seq), provided that a meaningful underlying structure exists between the input sequences and output sequences. More concretely, this capacity is contingent on the presence of sufficient mutual information between the sequences, as demonstrated in tasks such as GSM8k, machine translation, and MMLU.

But when can it not? When does it fail to find it when there is complete information in the input to produce the output. When f(x) -> y is simple, knowable and deterministic (such as factorial).

We find that if the problem 1) requires recursive computation, 2) is not “reducible” to a simpler subproblem (such as the sum of #s from 1 to n which is reducible to n*(n+1)/2), and 3) is not broken down for the model via COT (chain of thought) prompting during training, LLMs can not find the mapping between input and output sequences and, more strongly, are mathematically incapable of estimating the solution function f() (i.e. Factorial, Fibonnaci, etc.). Note, Factorial could be approximated with sterling’s eq, but this is not the exact solution and the error is 1/(12*n), so at lower n the error term is noticeable, and similarly Fibonnaci can also be approximated with Binet’s eq, but this is not the exact solution and the error is ϕ^n/(sqrt(5)) where ϕ is the golden ratio (1.62), so its already noticeable at even small n and grows.

This is because LLM architectures are not recurrent. They can not learn a program f() once and then call it as many times as is needed. Instead they are feed forward, and thus, even in the best case, the model would have to learn the recursive function f() in each transformer layer, and during the forward inference, call the function f() in each layer to produce the right output. If there are m layers, and we ask it to solve f(n) with m<n, then it is impossible for the model to solve f(n) unless it can smear the computation over n tokens (i.e. via COT style answering or otherwise). 

In practice, this program is not learned m times, but instead, the model will only memorize the answers for the few examples provided to the model, which of course does not generalize to a test set. 



Discussion here:
https://docs.google.com/document/d/1JNeVKeuaTjJ1w_VprRs-hZihKBpLqRKQ_xPzWzKvNmM/edit?tab=t.0
