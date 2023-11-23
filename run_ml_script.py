import gen_summary
from sshmanager import SSHManager


gen_summary.main()

ssh_manager = SSHManager()
ssh_manager.create_ssh_client('54.180.107.205', 'ubuntu')
ssh_manager.send_file('/srv/ml-server/result/summary.json', '/srv/Emostock/summary/summary.json')
ssh_manager.close_ssh_client()