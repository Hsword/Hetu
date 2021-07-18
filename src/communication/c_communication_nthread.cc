#include "../common/dlarray.h"
#include <stdio.h>
#include <omp.h>
#include <chrono>
#include <stdlib.h>
#include <zmq.h>
#include <string.h>
#include <thread>
#include <iostream>
using namespace std;
using namespace std::chrono;

extern "C" {
int worker_number;
int worker_id;
const char *this_worker;
int *segment_begin;
int *segment_end;
float *output;
char buf[10];
float *receive_buf;

void *context_backward = zmq_ctx_new();
void *socket_backward = zmq_socket(context_backward, ZMQ_REQ);

void *context_forward = zmq_ctx_new();
void *socket_forward = zmq_socket(context_forward, ZMQ_REP);

const char *pre = "tcp://";

void show(float *, int);
void DL_Connect_Init(int _worker_number, int _worker_id, char *source_ip,
                     char *dest_ip) {
    worker_number = _worker_number;
    worker_id = _worker_id;
    char *_source_ip = new char[50];
    char *_dest_ip = new char[50];

    int pre_len = strlen(pre);

    for (int i = 0; i < pre_len; i++) {
        _source_ip[i] = pre[i];
        _dest_ip[i] = pre[i];
    }
    int len1 = strlen(source_ip);
    for (int i = 0; i < len1; i++) {
        _source_ip[i + pre_len] = source_ip[i];
    }
    _source_ip[len1 + pre_len] = 0;

    int len2 = strlen(dest_ip);
    for (int i = 0; i < len2; i++) {
        _dest_ip[i + pre_len] = dest_ip[i];
    }
    _dest_ip[len2 + pre_len] = 0;

    // connect the server
    zmq_connect(socket_backward, _dest_ip);
    zmq_bind(socket_forward, _source_ip);
    delete[] _source_ip;
    delete[] _dest_ip;
}

void send_msg(void *_socket, void *data, int len) {
    char buff[10];
    int bytes;
    bytes = zmq_send(_socket, data, len, 0);
    bytes = zmq_recv(socket_backward, buf, 10, 0);
    buf[bytes] = '\0';
    if (strcmp(buf, "ok") != 0)
        printf("error\n");
}

void receive_msg(void *_socket, void *data, int len) {
    int bytes;
    bytes = zmq_recv(socket_forward, data, len, 0);
    const char *rep = "ok";
    bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
}

int DLArray_len;
void DL_Communicate_Init(DLArray *arr) {
    delete[] segment_begin;
    delete[] segment_end;
    delete[] receive_buf;
    DLArray_len = 1;
    segment_begin = new int[worker_number + 2];
    segment_end = new int[worker_number + 2];

    for (int i = 0; i < arr->ndim; i++) {
        DLArray_len *= arr->shape[i];
    }
    // divide
    const int segment_size = DLArray_len / worker_number;
    const int residual = DLArray_len % worker_number;

    // the buffer for receive
    receive_buf = new float[segment_size + 1];

    for (int i = 0; i < worker_number; i++) {
        segment_begin[i] = segment_size;
        if (i < residual)
            segment_begin[i]++;
    }
    segment_end[0] = segment_begin[0];
    for (int i = 1; i < worker_number; i++) {
        segment_end[i] = segment_end[i - 1] + segment_begin[i];
    }
    for (int i = 0; i < worker_number; i++) {
        segment_begin[i] = segment_end[i] - segment_begin[i];
    }
}

void add(float *now, float *delta, int begin, int len) {
#pragma omp parallel for
    for (int i = 0; i < len; i++) {
        now[begin + i] += delta[i];
    }
}

void average(float *now, int len) {
#pragma omp parallel for
    for (int i = 0; i < len; i++) {
        now[i] /= worker_number;
    }
}

void check() {
    if (errno == EAGAIN)
        printf("EAGAIN\n");
    else if (errno == ENOTSUP)
        printf("ENOTSUP\n");
    else if (errno == EFSM)
        printf("EFSM\n");
    else if (errno == ETERM)
        printf("ETERM\n");
    else if (errno == ENOTSOCK)
        printf("ENOTSOCK\n");
    else if (errno == EINTR)
        printf("EINTR\n");
    else if (errno == EHOSTUNREACH)
        printf("EHOSTUNREACH\n");
}

void RingAllreduce_Server(float *data, int len) {
    output = new float[len];

    memcpy(output, data, len * sizeof(float));
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;

        // send
        bytes = zmq_send(
            socket_backward, output + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);

        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");

        // receive

        bytes = zmq_recv(
            socket_forward, receive_buf,
            sizeof(float)
                * (segment_end[(segment - 1 + worker_number) % worker_number]
                   - segment_begin[(segment - 1 + worker_number)
                                   % worker_number]),
            0);

        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
        add(output, receive_buf,
            segment_begin[(segment - 1 + worker_number) % worker_number],
            (segment_end[(segment - 1 + worker_number) % worker_number]
             - segment_begin[(segment - 1 + worker_number) % worker_number]));
    }
    memcpy(data, output, len * sizeof(float));
    delete[] output;
}

void _RingAllreduce_Server(float *data, int len) {
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;

        // send
        bytes = zmq_send(
            socket_backward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);

        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");

        // receive
        bytes = zmq_recv(
            socket_forward, receive_buf,
            sizeof(float)
                * (segment_end[(segment - 1 + worker_number) % worker_number]
                   - segment_begin[(segment - 1 + worker_number)
                                   % worker_number]),
            0);

        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
        add(data, receive_buf,
            segment_begin[(segment - 1 + worker_number) % worker_number],
            (segment_end[(segment - 1 + worker_number) % worker_number]
             - segment_begin[(segment - 1 + worker_number) % worker_number]));
    }
}

void _RingAllreduce_Server_nthread(float *data, int len) {
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;

        // send thread
        std::thread send_thread(
            send_msg, socket_backward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]));

        // receive thread
        std::thread receive_thread(
            receive_msg, socket_forward, receive_buf,
            sizeof(float)
                * (segment_end[(segment - 1 + worker_number) % worker_number]
                   - segment_begin[(segment - 1 + worker_number)
                                   % worker_number]));
        send_thread.join();
        receive_thread.join();
        add(data, receive_buf,
            segment_begin[(segment - 1 + worker_number) % worker_number],
            (segment_end[(segment - 1 + worker_number) % worker_number]
             - segment_begin[(segment - 1 + worker_number) % worker_number]));
    }
}

void RingAllgather_Server(float *data, int len) {
    float *output;
    output = new float[len];
    memcpy(output, data, len * sizeof(float));
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // send
        bytes = zmq_send(socket_backward,
                         output + segment_begin[(segment + 1) % worker_number],
                         sizeof(float)
                             * (segment_end[(segment + 1) % worker_number]
                                - segment_begin[(segment + 1) % worker_number]),
                         0);
        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");

        // receive
        bytes = zmq_recv(
            socket_forward, output + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);
        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
    }
    average(output, len);
    memcpy(data, output, len * sizeof(float));
    delete[] output;
}

void _RingAllgather_Server(float *data, int len) {
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // send
        bytes = zmq_send(socket_backward,
                         data + segment_begin[(segment + 1) % worker_number],
                         sizeof(float)
                             * (segment_end[(segment + 1) % worker_number]
                                - segment_begin[(segment + 1) % worker_number]),
                         0);
        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");

        // receive
        bytes = zmq_recv(
            socket_forward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);
        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
    }
    average(data, len);
}

void _RingAllgather_Server_nthread(float *data, int len) {
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // send thread
        std::thread send_thread(
            send_msg, socket_backward,
            data + segment_begin[(segment + 1) % worker_number],
            sizeof(float)
                * (segment_end[(segment + 1) % worker_number]
                   - segment_begin[(segment + 1) % worker_number]));

        // receive thread
        std::thread receive_thread(
            receive_msg, socket_forward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]));
        send_thread.join();
        receive_thread.join();
    }
    average(data, len);
}

void RingAllreduce_Client(float *data, int len) {
    float *output;
    output = new float[len];
    memcpy(output, data, len * sizeof(float));
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // receive
        bytes = zmq_recv(
            socket_forward, receive_buf,
            sizeof(float)
                * (segment_end[(segment - 1 + worker_number) % worker_number]
                   - segment_begin[(segment - 1 + worker_number)
                                   % worker_number]),
            0);

        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
        add(output, receive_buf,
            segment_begin[(segment - 1 + worker_number) % worker_number],
            (segment_end[(segment - 1 + worker_number) % worker_number]
             - segment_begin[(segment - 1 + worker_number) % worker_number]));
        // send
        bytes = zmq_send(
            socket_backward, output + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);

        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");
    }
    memcpy(data, output, len * sizeof(float));
    delete[] output;
}

void _RingAllreduce_Client(float *data, int len) {
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // receive
        bytes = zmq_recv(
            socket_forward, receive_buf,
            sizeof(float)
                * (segment_end[(segment - 1 + worker_number) % worker_number]
                   - segment_begin[(segment - 1 + worker_number)
                                   % worker_number]),
            0);

        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
        // send
        bytes = zmq_send(
            socket_backward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);

        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");
        add(data, receive_buf,
            segment_begin[(segment - 1 + worker_number) % worker_number],
            (segment_end[(segment - 1 + worker_number) % worker_number]
             - segment_begin[(segment - 1 + worker_number) % worker_number]));
    }
}

void _RingAllreduce_Client_nthread(float *data, int len) {
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // receive thread
        std::thread receive_thread(
            receive_msg, socket_forward, receive_buf,
            sizeof(float) * (segment_end[segment] - segment_begin[segment]));

        // send thread
        std::thread send_thread(
            send_msg, socket_backward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]));

        send_thread.join();
        receive_thread.join();
        add(data, receive_buf,
            segment_begin[(segment - 1 + worker_number) % worker_number],
            (segment_end[(segment - 1 + worker_number) % worker_number]
             - segment_begin[(segment - 1 + worker_number) % worker_number]));
    }
}

void RingAllgather_Client(float *data, int len) {
    float *output;
    output = new float[len * sizeof(float)];
    memcpy(output, data, len * sizeof(float));
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;

        // receive
        bytes = zmq_recv(
            socket_forward, output + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);
        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
        // send
        bytes = zmq_send(socket_backward,
                         output + segment_begin[(segment + 1) % worker_number],
                         sizeof(float)
                             * (segment_end[(segment + 1) % worker_number]
                                - segment_begin[(segment + 1) % worker_number]),
                         0);
        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");
    }
    average(output, len);
    memcpy(data, output, len * sizeof(float));
    delete[] output;
}

void _RingAllgather_Client(float *data, int len) {
    int bytes;
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // receive
        bytes = zmq_recv(
            socket_forward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]), 0);
        const char *rep = "ok";
        bytes = zmq_send(socket_forward, rep, strlen(rep), 0);
        // send
        bytes = zmq_send(socket_backward,
                         data + segment_begin[(segment + 1) % worker_number],
                         sizeof(float)
                             * (segment_end[(segment + 1) % worker_number]
                                - segment_begin[(segment + 1) % worker_number]),
                         0);
        bytes = zmq_recv(socket_backward, buf, 10, 0);
        buf[bytes] = '\0';
        if (strcmp(buf, "ok") != 0)
            printf("error\n");
    }
    average(data, len);
}

void _RingAllgather_Client_nthread(float *data, int len) {
    for (int i = 0; i < worker_number - 1; i++) {
        int segment = (worker_id - i + worker_number) % worker_number;
        // receive thread
        std::thread receive_thread(
            receive_msg, socket_forward, data + segment_begin[segment],
            sizeof(float) * (segment_end[segment] - segment_begin[segment]));

        // send thread
        std::thread send_thread(
            send_msg, socket_backward,
            data + segment_begin[(segment + 1) % worker_number],
            sizeof(float)
                * (segment_end[(segment + 1) % worker_number]
                   - segment_begin[(segment + 1) % worker_number]));

        send_thread.join();
        receive_thread.join();
    }
    average(data, len);
}

void DL_Communicate_Close() {
    zmq_close(socket_forward);
    zmq_ctx_destroy(context_forward);

    zmq_close(socket_backward);
    zmq_ctx_destroy(context_backward);
}

void DL_Communicate_Server(DLArray *arr) {
    _RingAllreduce_Server_nthread((float *)(arr->data), DLArray_len);
    _RingAllgather_Server_nthread((float *)(arr->data), DLArray_len);
}

void DL_Communicate_Client(DLArray *arr) {
    _RingAllreduce_Client_nthread((float *)(arr->data), DLArray_len);
    _RingAllgather_Client_nthread((float *)(arr->data), DLArray_len);
}
}
