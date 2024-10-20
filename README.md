# factorial_experiments
This is a research repo for getting an LLM to learn factorial. 

LLMs are hypothesized to have the capacity to map between ANY arbitrary sequences (seq2seq), provided that a meaningful underlying structure exists between the input sequences and output sequences. More concretely, this capacity is contingent on the presence of sufficient mutual information between the sequences, as demonstrated in tasks such as GSM8k, machine translation, and MMLU.

But when can it not? When does it fail to find it when there is complete information in the input to produce the output. When f(x) -> y is simple, knowable and deterministic (such as factorial).

We find that if the problem 1) requires recursive computation, 2) is not “reducible” to a simpler subproblem (such as the sum of #s from 1 to n which is reducible to n*(n+1)/2), and 3) is not broken down for the model via COT (chain of thought) prompting during training, LLMs can not find the mapping between input and output sequences and, more strongly, are mathematically incapable of estimating the solution function f() (i.e. Factorial, Fibonnaci, etc.). Note, Factorial could be approximated with sterling’s eq, but this is not the exact solution and the error is 1/(12*n), so at lower n the error term is noticeable, and similarly Fibonnaci can also be approximated with Binet’s eq, but this is not the exact solution and the error is ϕ^n/(sqrt(5)) where ϕ is the golden ratio (1.62), so its already noticeable at even small n and grows.

This is because LLM architectures are not recurrent. They can not learn a program f() once and then call it as many times as is needed. Instead they are feed forward, and thus, even in the best case, the model would have to learn the recursive function f() in each transformer layer, and during the forward inference, call the function f() in each layer to produce the right output. If there are m layers, and we ask it to solve f(n) with m<n, then it is impossible for the model to solve f(n) unless it can smear the computation over n tokens (i.e. via COT style answering or otherwise). 

In practice, this program is not learned m times, but instead, the model will only memorize the answers for the few examples provided to the model, which of course does not generalize to a test set. 

This is because of SGD in its current form. Incrementally better thetas are ones that memorize NOT generalize. Finding the general solution requires a huge JUMP where incrementally different thetas from that "global minimizer" are all sharply worse. i.e. theta_star => n! = n*(n-1)!. If for example, we found n! = n*(1.1*n-1)! that would explode and be very wrong. 

So the correct solution is at the bottom of a sharp, deep crevasse. 

The memorizing solution is at the bottom of a deep wide bowl, that gets incrementally better as you memorize a bit more of the input-output tokens mapping. So this is of course the much more likely solution that is found by SGD.

So how do we STOP MEMORIZATION and inspire GENERALIZATION! The issue is SGD in its current form. 

If we think about humans, we don't memorize. In fact, its really hard for humans to memorize things. It takes great effort to memorize a poem. But LLMs can do it with ease. Alteratively, we can't help but generalize from an example. Almost to a fault! We are always looking for the general rule vs. the specific example. If we get T boned in an intersection on a saturday at 1pm, we get a fear of all cars all the time (too broad a generalization), vs. the correct update would be to be careful when crossing an intersection perhaps or if it was a drunk driver, don't update at all (the right amount of generalization), vs. don't cross intersections at 1pm on saturdays (the too specific, dumb generalization, that SGD does today by default). 

--

The way I look at it is if I give you a single input-output pair, what do you learn from it? What is the “lesson”? How much information can we take away from it? It takes a lot of thought just to decide how to update policy from that single sample. 

Let’s do a real life example. Let’s say you go to IHOP for lunch at 1pm on a Saturday with Francois, and lets just say I DEEPLY insult you. What is the lesson? What is the best update to your policy? 0) no update 1) avoid that specific IHOP at 1pm on Saturdays with Francois (this is what backprop does today essentially if we have no data augs) 2) avoid all IHOPs at 1pm on Saturdays with Francois (this is what backprop does today if we include data augs to span all IHOPS) 3) even more so, avoid all IHOPs at all times with all people (this is what backprop does today if we include data augs to span all time and ppl) 4) avoid all pancake restaurants at all times with all people (this is what backprop does today if we include data augs to span all pancake restaurants with all people) 5) avoid all restaurants at all times with all people (this is what backprop does today if we include data augs to span all restaurants). 6) avoid Francois at 1pm on Saturdays (if we add data augs to span all places) 7) avoid Francois always (if we add data augs to span all places and time) 8) avoid all white males always (if we add data augs to span all places and time and names) 9) avoid all ppl always (if we add data augs to span all places, ppl and time).  

TLDR: Humans think critically about the takeaway given a single example, try to “make sense” of it, then and only then we apply that gradient, the 'common sense' gradient, to our weights to update our policy. This way we update our policy in an intelligent way, vs just blindly updating weights. 

We call this common sense but we can’t give these models common sense. What is and what isn’t common sense? How can we iterate over MUCH LESS data (like 3 examples or even 1 example) and decide how to update our policies in a smart way that won't be memorizing. 

-- 

In conclusion, we need an optimization procedure that reduces train loss at the same rate as val loss at every step. This means that every update to the weights is improving on the global objective function vs. memorizing. If the train loss declines MORE than the val loss, then the gradient direction was following the path toward memorization vs. generalization. 


So how can we change SGD to find the path of generalization and not follow the path of memorization??

That is the golden question we explore in this repo. 


Further experiment results / discussion here:
https://docs.google.com/document/d/1JNeVKeuaTjJ1w_VprRs-hZihKBpLqRKQ_xPzWzKvNmM/edit?tab=t.0
