import math
import os
import shutil
import scapy.all as scapy
from tqdm import tqdm

clean_protocols = '"not arp and not dns and not stun and not dhcpv6 and not icmpv6 and not icmp and not dhcp and not llmnr and not nbns and not ntp and not igmp"'
breakpoint()
exact_5 = False  # True represents each flow contains exact 5 packets

for p, d, f in os.walk('.'):
    for dir in d:
        print("----------")
        print(f'[{dir}]')
        print("----------")
        for pp, dd, ff in os.walk(os.path.join(p, dir)):
            breakpoint()
            for file in ff:
                print(file + '...')
                # rename pcapng as pcap
                if file.endswith('.pcapng'):
                    new_file = file.replace('.pcapng', '.pcap')
                    shutil.move(os.path.join(pp, file), os.path.join(pp, new_file))
                    file = new_file
                org_file = os.path.join(pp, file)
                
                if exact_5:
                    # truncate pcap at 100000 packets
                    new_org_file = os.path.join(pp, file.replace('.pcap', '_100000.pcap'))
                    os.system(f"tcpdump -r {org_file} -w {new_org_file} -c 100000")
                    os.remove(org_file)
                    org_file = new_org_file
                
                # remove unwanted protocols
                clean_file = org_file.replace('.pcap', '.clean.pcap')
                os.system(f"tshark -F pcap -r {org_file} -Y {clean_protocols} -w {clean_file}")
                os.remove(org_file)
                org_file = clean_file
                
                # split by session using cmd variable
                output_path = org_file.replace('.pcap', '')
                os.makedirs(output_path, exist_ok=True)
                # cmd = "mono /3241903007/workstation/AnomalyTrafficDetection/ConfusionModel/tools/SplitCap.exe -r %s -s session -o " + output_path
                cmd = f"mono /3241903007/workstation/AnomalyTrafficDetection/ET-BERT/tools/SplitCap.exe -r {org_file} -s session -o {output_path} > /dev/null"
                os.system(cmd)
                os.remove(org_file)  # 删除原始 session 文件
                
                # process each session file
                for ppp, ddd, fff in os.walk(output_path):
                    for file_ in fff:
                        pkts = scapy.rdpcap(os.path.join(ppp, file_))
                        if not exact_5:
                            # 保留所有 session，包数少于5也保留
                            print(f"[提示] {file_} 包数: {len(pkts)}（已保留）")
                        else:
                            if len(pkts) < 5:
                                continue
                            flow_num = len(pkts) // 5
                            for i in range(flow_num):
                                flow_file = os.path.join(ppp, file_.replace('.pcap', f'_flow{i}.pcap'))
                                scapy.wrpcap(flow_file, pkts[i * 5: i * 5 + 5])
                            os.remove(os.path.join(ppp, file_))  # 删除原 session 文件
                    break
            break
    break
