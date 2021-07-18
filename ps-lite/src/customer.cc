/**
 *  Copyright (c) 2015 by Contributors
 */
#include "ps/internal/customer.h"
#include "ps/internal/postoffice.h"
namespace ps {

const int Node::kEmpty = std::numeric_limits<int>::max();
const int Meta::kEmpty = std::numeric_limits<int>::max();

Customer::Customer(int app_id, int customer_id,
                   const Customer::RecvHandle &recv_handle) :
    app_id_(app_id),
    customer_id_(customer_id), recv_handle_(recv_handle) {
    cur_timestamp = 0;
    Postoffice::Get()->AddCustomer(this);
    // for debug set num_threads = 5;
    int num_threads = 5;
    for (int i = 0; i < num_threads; i++) {
        recv_threads_.emplace_back(new std::thread(&Customer::Receiving, this));
    }
}

Customer::~Customer() {
    Postoffice::Get()->RemoveCustomer(this);
    Message msg;
    msg.meta.control.cmd = Control::TERMINATE;
    msg.meta.control.barrier_group = 0;
    msg.meta.control.msg_sig = 0;
    recv_queue_.Push(msg);
    for (auto &thread : recv_threads_)
        thread->join();
}

int Customer::NewRequest(int recver) {
    std::lock_guard<std::mutex> lk(tracker_mu_);
    assert(recver == kServerGroup);
    tracker_[cur_timestamp] = false;
    return cur_timestamp++;
}

void Customer::WaitRequest(int timestamp) {
    std::unique_lock<std::mutex> lk(tracker_mu_);
    tracker_cond_.wait(lk, [this, timestamp] { return tracker_[timestamp]; });
    tracker_.erase(timestamp);
}

void Customer::Receiving() {
    while (true) {
        Message recv;
        // thread safe
        recv_queue_.WaitAndPop(&recv);
        if (!recv.meta.control.empty()
            && recv.meta.control.cmd == Control::TERMINATE) {
            recv_queue_.Push(recv);
            break;
        }
        recv_handle_(recv);
        if (!recv.meta.request) {
            std::lock_guard<std::mutex> lk(tracker_mu_);
            tracker_[recv.meta.timestamp] = true;
            tracker_cond_.notify_all();
        }
    }
}

} // namespace ps
