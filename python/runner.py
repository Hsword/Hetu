import argparse
import yaml
import os
import signal
import multiprocessing
import subprocess
import paramiko
import socket
import psutil
import contextlib
import hetu as ht

_procs = []


def signal_handler(signal, frame):
    print("SIGINT signal caught, stop Training")
    for proc in _procs:
        proc.kill()
    global executor_shell
    executor_shell.kill()
    exit(0)


def start_sched():
    os.environ["DMLC_ROLE"] = "scheduler"
    ht.scheduler_init()
    ht.scheduler_finish()


def start_server():
    os.environ["DMLC_ROLE"] = "server"
    ht.server_init()
    ht.server_finish()


@ contextlib.contextmanager
def ssh_connect(host, identify_file):
    try:
        ssh_directory = os.path.expanduser('~/.ssh') if identify_file == '' else os.path.dirname(
            os.path.abspath(os.path.expanduser(identify_file)))
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        private = paramiko.RSAKey.from_private_key_file(
            os.path.join(ssh_directory, 'id_rsa'))
        config = paramiko.config.SSHConfig.from_path(
            os.path.join(ssh_directory, 'config'))
        conf = config.lookup(host)
        ssh.connect(hostname=conf['hostname'], port=conf['port'],
                    username=conf['user'], pkey=private)
        yield ssh
    finally:
        ssh.close()


def start_remote_server(host, local_server_num, identify_file):
    with ssh_connect(host, identify_file) as ssh:
        sftp = ssh.open_sftp()
        sftp.put('/tmp/hetu_ps_config.yml',
                 '/tmp/hetu_ps_config.yml', confirm=True)
        sftp.close()
        stdin, stdout, stderr = ssh.exec_command(
            'python -m hetu.launcher /tmp/hetu_ps_config.yml -n %d' % local_server_num)
        stdout = stdout.read().decode()
        stderr = stderr.read().decode()
        if stdout:
            print('From remote %s stdout:\n %s' % (host, stdout.strip()))
        if stderr:
            print('From remote %s stderr:\n %s' % (host, stderr.strip()))


def send_config(host, identify_file):
    with ssh_connect(host, identify_file) as ssh:
        sftp = ssh.open_sftp()
        sftp.put('/tmp/hetu_config.yml', '/tmp/hetu_config.yml', confirm=True)
        sftp.close()


def get_nic_names(local_address, remote_hostnames, identify_file):
    # get local interface
    nics = dict()
    for iface, addrs in psutil.net_if_addrs().items():
        for addr in addrs:
            if addr.family == socket.AF_INET:
                nics[addr.address] = iface
    local_nic = nics[local_address]

    # get remote interfaces
    command_prefix = "\"from socket import AF_INET;\nfrom psutil import net_if_addrs;\n" +\
        "nics = dict();\nfor iface, addrs in net_if_addrs().items():\n    for addr in addrs:" +\
        "\n        if addr.family == AF_INET:\n            nics[addr.address] = iface;\n"
    ssh_directory = os.path.expanduser('~/.ssh') if identify_file == '' else os.path.dirname(
        os.path.abspath(os.path.expanduser(identify_file)))
    remote_nics = set()
    for hostname in remote_hostnames:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        private = paramiko.RSAKey.from_private_key_file(
            os.path.join(ssh_directory, 'id_rsa'))
        config = paramiko.config.SSHConfig.from_path(
            os.path.join(ssh_directory, 'config'))
        conf = config.lookup(hostname)
        command = command_prefix + "print(nics[\'%s\'])\"" % (conf['hostname'])
        ssh.connect(hostname=conf['hostname'], port=conf['port'],
                    username=conf['user'], pkey=private)
        stdin, stdout, stderr = ssh.exec_command('python -c %s' % command)
        stdout = stdout.read().decode()
        stderr = stderr.read().decode()
        remote_nics.add(stdout.strip())
        if stderr:
            print('From remote %s stderr:\n %s' % (hostname, stderr.strip()))
        ssh.close()

    remote_nics.add(local_nic)
    return list(remote_nics)


def get_subnet(local_address, remote_hostnames, identify_file=''):
    ssh_directory = os.path.expanduser('~/.ssh') if identify_file == '' else os.path.dirname(
        os.path.abspath(os.path.expanduser(identify_file)))
    config = paramiko.config.SSHConfig.from_path(
        os.path.join(ssh_directory, 'config'))
    remote_address = [config.lookup(hostname)['hostname']
                      for hostname in remote_hostnames]
    remote_address.append(local_address)
    address_pool = set()
    for addr in remote_address:
        binary_repr = int(''.join([format(int(part), '08b')
                                   for part in addr.split('.')]), 2)
        address_pool.add(format(binary_repr+1, '032b'))
        address_pool.add(format(binary_repr-1, '032b'))
    address_pool = list(address_pool)
    longestCommonPrefix = 0
    for item in zip(*address_pool):
        if len(set(item)) > 1:
            break
        longestCommonPrefix += 1
    if longestCommonPrefix > 30:
        longestCommonPrefix = 30
    assert longestCommonPrefix >= 16, 'Hosts not in the same subnet!'
    commonAddress = address_pool[0][:longestCommonPrefix] + \
        '0' * (32 - longestCommonPrefix)
    parts = [commonAddress[:8], commonAddress[8:16],
             commonAddress[16:24], commonAddress[24:]]
    subnet = '.'.join([str(int(part, 2))
                       for part in parts]) + '/%d' % longestCommonPrefix
    return subnet


def main():
    signal.signal(signal.SIGINT, signal_handler)
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', default=None,
                        help='Configuration file.')
    parser.add_argument('-w', '--workers', type=int, default=0,
                        help='Shorthand for the number of local worker.')
    parser.add_argument('-s', '--servers', type=int, default=0,
                        help='Shorthand for the number of local server.')
    parser.add_argument('-i', '--identify', default='',
                        help='SSH identify file.')
    parser.add_argument('command', nargs=argparse.REMAINDER,
                        help='Command to be executed.')
    args = parser.parse_args()
    settings = ht.DistConfig(args.config, args.servers, args.workers)
    print(settings)
    if settings.enable_PS:
        ps_config = settings.make_ps_config()
        for k, v in ps_config.items():
            os.environ[k] = str(v)
    settings.save('/tmp/hetu_config.yml')
    for host in settings.hosts:
        if host != settings.chief:
            send_config(host, args.identify)

    global executor_shell
    if len(settings.hosts) == 1:
        # single machine
        # TODO: add hostdress validation check
        if settings.enable_PS:
            proc = multiprocessing.Process(target=start_sched)
            _procs.append(proc)
            for i in range(settings.num_servers):
                proc = multiprocessing.Process(target=start_server)
                _procs.append(proc)
        for proc in _procs:
            proc.start()
        mpi_command = 'mpirun --allow-run-as-root --tag-output -np %d %s' % (
            settings.num_workers, ' '.join(args.command))
        env = dict(os.environ)
        if settings.enable_PS:
            env["DMLC_ROLE"] = "worker"
        executor_shell = subprocess.Popen(
            mpi_command, shell=True, env=env, stdout=None, stderr=None)
        for proc in _procs:
            proc.join()
        executor_shell.wait()
    else:
        # multi machines

        #! nic names not used currently, use subnets instead; nccl_socket_name please specified in /etc/bash.bashrc
        #! nic methods cannot support different nic name on different machines
        # nics = get_nic_names(chief_address, set(hosts) - {chief}, args.identify)
        # joined_nics = ','.join(nics)
        subnet = get_subnet(settings.chief_address, set(
            settings.hosts) - {settings.chief}, args.identify)
        if settings.enable_PS:
            with open('/tmp/hetu_ps_config.yml', 'w') as fw:
                yaml.dump({'shared': ps_config}, fw)
            proc = multiprocessing.Process(target=start_sched)
            _procs.append(proc)
        for node in settings.hosts:
            if node == settings.chief:
                for i in range(settings.servers.get(node, 0)):
                    proc = multiprocessing.Process(target=start_server)
                    _procs.append(proc)
            else:
                if settings.servers.get(node, 0):
                    proc = multiprocessing.Process(target=start_remote_server, args=[
                                                   node, settings.servers[node], args.identify])
                    _procs.append(proc)
        for proc in _procs:
            proc.start()
        basic_args = '--allow-run-as-root --tag-output'
        hosts_in_command = ','.join(
            ['%s:%d' % (node, nworkers) for node, nworkers in settings.workers.items()])
        mpi_ssh_args = '' if args.identify == '' else '-bootstrap=ssh -bootstrap-exec-args -i %s' % args.identify
        tcp_intf_arg = '-mca btl_tcp_if_include %s' % subnet
        # tcp_intf_arg = '-mca btl_tcp_if_include %s' % joined_nics
        # nccl_socket_intf_arg = '-x NCCL_SOCKET_IFNAME=%s' % joined_nics
        env_list = ' '.join(['-x %s=%s' % (k, str(v)) for k, v in ps_config.items()] + [
                            '-x DMLC_ROLE=worker']) if settings.enable_PS else ''
        mpi_command = (
            'mpirun {basic_args} '
            '--host {hosts} '
            '{mpi_ssh_args} '
            '{tcp_intf_arg} '
            # '{nccl_socket_intf_arg} '
            '{env} '
            '{command}'
            .format(basic_args=basic_args,
                    hosts=hosts_in_command,
                    mpi_ssh_args=mpi_ssh_args,
                    tcp_intf_arg=tcp_intf_arg,
                    # nccl_socket_intf_arg=nccl_socket_intf_arg,
                    env=env_list,
                    command=' '.join(args.command))
        )
        executor_shell = subprocess.Popen(
            mpi_command, shell=True, stdout=None, stderr=None)
        for proc in _procs:
            proc.join()
        executor_shell.wait()


if __name__ == '__main__':
    #! need to modify /etc/bash.bashrc on other machines for:
    #       * specify NCCL_SOCKET_IFNAME
    #       * specify PATH for mpirun support
    #       * activate conda environment
    #       * specify PYTHONPATH for hetu support
    #! ssh process to other machines for server CANNOT receive SIGINT from Ctrl+C on this machine, please kill on other machines
    main()
