import os

lines = os.popen('ps aux | grep train_cylinder_asym_ood_final.py').readlines()
print(lines)
for line in lines:
    line = line.split()
    p = line[1]
    print(f'killing {p}')
    os.system(f'kill -9 {p}')
