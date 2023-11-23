import paramiko
from scp import SCPClient, SCPException
import os

class SSHManager:
    """
    usage:
        >>> import SSHManager
        >>> ssh_manager = SSHManager()
        >>> ssh_manager.create_ssh_client(hostname, username)
        >>> ssh_manager.send_command("ls -al")
        >>> ssh_manager.send_file("/path/to/local_path", "/path/to/remote_path")
        >>> ssh_manager.get_file("/path/to/remote_path", "/path/to/local_path")
        ...
        >>> ssh_manager.close_ssh_client()
    """
    def __init__(self):
        self.ssh_client = None

    def create_ssh_client(self, hostname, username):
        """Create SSH client session to remote server"""
        if self.ssh_client is None:
            key = paramiko.RSAKey.from_private_key_file("key")
            self.ssh_client = paramiko.SSHClient()
            self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh_client.connect(hostname, username=username, pkey=key)
        else:
            print("SSH client session exist.")

    def close_ssh_client(self):
        """Close SSH client session"""
        self.ssh_client.close()

    def send_file(self, local_path, remote_path):
        """Send a single file to remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, remote_path, preserve_times=True)
        except SCPException as e:
            raise Exception(str(e))
    
    def send_directory(self, local_path, remote_path):
        """Send an entire directory to remote path"""
        self.send_command("mkdir -p " + remote_path)
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.put(local_path, recursive=True, remote_path=remote_path, preserve_times=True)
        except SCPException as e:
            raise Exception(str(e))

    def get_file(self, remote_path, local_path):
        """Get a single file from remote path"""
        try:
            with SCPClient(self.ssh_client.get_transport()) as scp:
                scp.get(remote_path, local_path)
        except SCPException as e:
            raise Exception(str(e))

    def send_command(self, command):
        """Send a single command"""
        stdin, stdout, stderr = self.ssh_client.exec_command(command)
        return stdout.readlines()