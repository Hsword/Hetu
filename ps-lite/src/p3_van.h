/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_P3_VAN_H_
#define PS_P3_VAN_H_
#include <memory>
namespace ps {

/**
 * \brief P3 based Van implementation
 */
class P3Van : public ZMQVan {
public:
    P3Van() {
    }
    virtual ~P3Van() {
    }

protected:
    void Start(int customer_id) override {
        start_mu_.lock();
        if (init_stage == 0) {
            // start sender
            // set num_threads= 5 for debug
            int num_threads = 5;
            for (int i = 0; i < num_threads; i++)
                sender_threads_.emplace_back(
                    new std::thread(&P3Van::Sending, this));
            // sender_thread_ = std::unique_ptr<std::thread>(
            //      new std::thread(&P3Van::Sending, this));
            init_stage++;
        }
        start_mu_.unlock();
        ZMQVan::Start(customer_id);
    }

    void Stop() override {
        ZMQVan::Stop();
        for (auto &thread : sender_threads_)
            thread->join();
    }

    int SendMsg(const Message &msg) override {
        send_queue_.Push(msg);
        return 0;
    }

    void Sending() {
        while (true) {
            Message msg;
            send_queue_.WaitAndPop(&msg);
            ZMQVan::SendMsg(msg);
            if (!msg.meta.control.empty()
                && msg.meta.control.cmd == Control::TERMINATE) {
                // debug for stop
                send_queue_.Push(msg);
                break;
            }
        }
    }

private:
    /** the thread for sending messages */
    // using multithread
    std::vector<std::shared_ptr<std::thread>> sender_threads_;
    ThreadsafePQueue send_queue_;
    int init_stage = 0;
};
} // namespace ps

#endif // PS_P3_VAN_H_
