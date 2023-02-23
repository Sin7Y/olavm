import subprocess
import os

res = subprocess.run("free -h", shell=True, capture_output=True)
print(res)

mem_m, mem_used, mem_free = map(int, os.popen('free -t -m').readlines()[1].split()[1:4])
print(mem_m)
print(mem_used)
print(mem_free)
tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:4])
print(tot_m)
print(used_m)
print(free_m)