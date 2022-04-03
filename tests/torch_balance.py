import torch
import sys
import numpy as np
def balanced_assignment(scores, max_iterations=100):
    scores=torch.transpose(scores, 0, 1)
    num_workers, num_jobs = scores.size()
    jobs_per_worker = num_jobs//num_workers
    value = scores.clone()

    iterations=0
    eps = (scores.max()-scores.min())/50
    if eps<1e-04:
        eps=1e-04

    print("eps:"+str(eps))
    cost = scores.new_zeros(1, num_jobs)
    jobs_with_bids = torch.zeros(num_workers).bool()
    while not jobs_with_bids.all():
#        print("iter:"+str(iterations))
        top_values, top_index = torch.topk(value, k=jobs_per_worker+1, dim=1)
#        print("value:")
#        print(value)
#        print("top_index:")
#        print(top_index)
#        print("top_value:")
#        print(top_values)
        bid_increments = top_values[:,:-1]-top_values[:, -1:]+eps
        bids=torch.scatter(torch.zeros(num_workers, num_jobs), dim=1, index=top_index[:,:-1], src=bid_increments)
        
#        print("bids:")
#        print(bids)

        if 0<iterations<max_iterations:
            bids[top_bidders, jobs_with_bids]=eps

#        print("bids:")
#        print(bids)

        top_bids, top_bidders = bids.max(dim=0)
        jobs_with_bids = top_bids>0

#        print("top_bidders:")
#        print(top_bidders)
#        print("top_bids:")
#       print(top_bids)
        top_bidders=top_bidders[jobs_with_bids]
        cost=cost+top_bids
        value=scores-cost
        
#        print("cost:")
#        print(cost)
#        print("top_index:")
#       print(top_index)

        if iterations<max_iterations:
#    print("top_bidders:")
#           print(top_bidders)
#           print("jobs_with_bids:")
#           print(jobs_with_bids)
            value[top_bidders, jobs_with_bids]=1e12
            print("!!!!!!!!!!!!!")
            print("value:")
            print(value)
            print("top_bidders:")
            print(top_bidders)
            print("jobs_with_bids:")
            print(jobs_with_bids)
        else:
            value[top_bidders, jobs_with_bids]=scores[top_bidders, jobs_with_bids]
        iterations+=1

    return top_index[:,:-1].reshape(-1)

if __name__ == '__main__':
    a=np.arange(80).reshape((20,4))
    a=torch.tensor(a)
    print(balanced_assignment(a))









