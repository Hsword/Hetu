/**
 *  Copyright (c) 2015 by Contributors
 */
#ifndef PS_INTERNAL_MESSAGE_H_
#define PS_INTERNAL_MESSAGE_H_
#include "common/sarray.h"
#include "ps/psf/PSFunc.h"
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ps {

/**
 * \brief information about a node
 */
struct Node {
    /** \brief the empty value */
    static const int kEmpty;
    /** \brief default constructor */
    Node() : id(kEmpty), port(kEmpty), is_recovery(false) {
    }
    /** \brief node roles */
    enum Role { SERVER, WORKER, SCHEDULER };
    /** \brief get debug string */
    std::string DebugString() const {
        std::stringstream ss;
        ss << "role="
           << (role == SERVER ? "server" :
                                (role == WORKER ? "worker" : "scheduler"))
           << (id != kEmpty ? ", id=" + std::to_string(id) : "")
           << ", ip=" << hostname << ", port=" << port
           << ", is_recovery=" << is_recovery;

        return ss.str();
    }
    /** \brief get short debug string */
    std::string ShortDebugString() const {
        std::string str = role == SERVER ? "S" : (role == WORKER ? "W" : "H");
        if (id != kEmpty)
            str += "[" + std::to_string(id) + "]";
        return str;
    }
    /** \brief the role of this node */
    Role role;
    /** \brief node id */
    int id;
    /** \brief customer id */
    int customer_id;
    /** \brief hostname or ip */
    std::string hostname;
    /** \brief the port this node is binding */
    int port;
    /** \brief whether this node is created by failover */
    bool is_recovery;
};
/**
 * \brief meta info of a system control message
 */
struct Control {
    /** \brief empty constructor */
    Control() : cmd(EMPTY) {
    }
    /** \brief return true is empty */
    inline bool empty() const {
        return cmd == EMPTY;
    }
    /** \brief get debug string */
    std::string DebugString() const {
        if (empty())
            return "";
        std::vector<std::string> cmds = {"EMPTY",   "TERMINATE", "ADD_NODE",
                                         "BARRIER", "ACK",       "HEARTBEAT"};
        std::stringstream ss;
        ss << "cmd=" << cmds[cmd];
        if (node.size()) {
            ss << ", node={";
            for (const Node &n : node)
                ss << " " << n.DebugString();
            ss << " }";
        }
        if (cmd == BARRIER)
            ss << ", barrier_group=" << barrier_group;
        if (cmd == ACK)
            ss << ", msg_sig=" << msg_sig;
        return ss.str();
    }
    /** \brief all commands */
    enum Command { EMPTY, TERMINATE, ADD_NODE, BARRIER, ACK, HEARTBEAT };
    /** \brief the command */
    Command cmd;
    /** \brief node infos */
    std::vector<Node> node;
    /** \brief the node group for a barrier, such as kWorkerGroup */
    int barrier_group;
    /** message signature */
    uint64_t msg_sig;
};
/**
 * \brief meta info of a message
 */
struct Meta {
    /** \brief the empty value */
    static const int kEmpty;
    /** \brief default constructor */
    Meta() :
        app_id(kEmpty), customer_id(kEmpty), timestamp(kEmpty), sender(kEmpty),
        recver(kEmpty), request(false), priority(kEmpty),
        psftype(PsfType::DensePull) {
    }
    std::string DebugString() const {
        std::stringstream ss;
        if (sender == Node::kEmpty) {
            ss << "?";
        } else {
            ss << sender;
        }
        ss << " => " << recver;
        if (timestamp != kEmpty)
            ss << ", timestamp=" << timestamp;
        if (!control.empty()) {
            ss << ", control={ " << control.DebugString() << " }";
        } else {
            ss << ", app_id=" << app_id << ", customer_id=" << customer_id
               << ", priority=" << priority << ", psfType=" << psftype;
        }
        return ss.str();
    }
    /** \brief the unique id of the application of messsage is for*/
    int app_id;
    /** \brief customer id*/
    int customer_id;
    /** \brief the timestamp of this message */
    int timestamp;
    /** \brief the node id of the sender of this message */
    int sender;
    /** \brief the node id of the receiver of this message */
    int recver;
    /** \brief whether or not this is a request message*/
    bool request;
    /** \brief system control message */
    Control control;
    /** \brief message priority */
    int priority;
    /** \brief server-side computation op for keys */
    PsfType psftype;
};
/**
 * \brief messages that communicated amaong nodes.
 */
struct Message {
    /** \brief the meta info of this message */
    Meta meta;
    /** \brief the large chunk of data of this message */
    std::vector<SArray<char>> data;

    std::string DebugString() const {
        std::stringstream ss;
        ss << meta.DebugString();
        if (data.size()) {
            ss << " Body:";
            for (const auto &d : data)
                ss << " data_size=" << d.size();
        }
        return ss.str();
    }
};
} // namespace ps
#endif // PS_INTERNAL_MESSAGE_H_
