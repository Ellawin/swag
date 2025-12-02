import numpy as np
import matplotlib.pyplot as plt

# PVP 2
file_pvp2 = open("pvp2/Data_sensitivity_pvp_2.txt", 'r')
list_gold2 = []
lam_s_ru2 = []
lam_s_rd2 = []
lam_s_tu2 = []
lam_s_td2 = []
S_max_ru2 = []
S_max_rd2 = []
S_max_tu2 = []
S_max_td2 = []


for line in file_pvp2:
    all_lines = file_pvp2.readlines()

interesting_lines = all_lines[10:29]
#print(interesting_lines)

for line in interesting_lines:
    line = line.rstrip().split(',')
    print("The line is", line)
    print("The line[0] is : ", line[0])
    print("The float(line[0]) is : ", float(line[0]))
    list_gold2.append(float(line[0]))   
    lam_s_ru2.append(float(line[1]))
    lam_s_rd2.append(float(line[2]))
    lam_s_tu2.append(float(line[3]))
    lam_s_td2.append(float(line[4]))
    S_max_ru2.append(float(line[5]))
    S_max_rd2.append(float(line[6]))
    S_max_tu2.append(float(line[7]))
    S_max_td2.append(float(line[8])) 

print("list list_gold : ", list_gold2)
#print('There are', count, 'lines in the file')

list_gold2 = np.array(list_gold2)
print("arrat list_gold : ", list_gold2)
lam_s_ru2 = np.array(lam_s_ru2)
lam_s_rd2 = np.array(lam_s_rd2)
lam_s_tu2 = np.array(lam_s_tu2)
lam_s_td2 = np.array(lam_s_td2)
S_max_ru2 = np.array(S_max_ru2)
S_max_rd2 = np.array(S_max_rd2)
S_max_tu2 = np.array(S_max_tu2)
S_max_td2 = np.array(S_max_td2)

# PVP 0

file_pvp0 = open("pvp0/air/thick_gold_5-41-2/Data_sensitivity_pvp.txt", 'r')
list_gold0 = []
lam_s_ru0 = []
lam_s_rd0 = []
lam_s_tu0 = []
lam_s_td0 = []
S_max_ru0 = []
S_max_rd0 = []
S_max_tu0 = []
S_max_td0 = []


for line in file_pvp0:
    all_lines = file_pvp0.readlines()

interesting_lines = all_lines[10:23]
#print(interesting_lines)

for line in interesting_lines:
    line = line.rstrip().split(',')
    # print("The line is", line)
    # print("The line[0] is : ", line[0])
    # print("The float(line[0]) is : ", float(line[0]))
    list_gold0.append(float(line[0]))   
    lam_s_ru0.append(float(line[1]))
    lam_s_rd0.append(float(line[2]))
    lam_s_tu0.append(float(line[3]))
    lam_s_td0.append(float(line[4]))
    S_max_ru0.append(float(line[5]))
    S_max_rd0.append(float(line[6]))
    S_max_tu0.append(float(line[7]))
    S_max_td0.append(float(line[8])) 

#print("list list_gold : ", list_gold0)
#print('There are', count, 'lines in the file')

list_gold = np.array(list_gold0)
#print("arrat list_gold : ", list_gold0)
lam_s_ru0 = np.array(lam_s_ru0)
lam_s_rd0 = np.array(lam_s_rd0)
lam_s_tu0 = np.array(lam_s_tu0)
lam_s_td0 = np.array(lam_s_td0)
S_max_ru0 = np.array(S_max_ru0)
S_max_rd0 = np.array(S_max_rd0)
S_max_tu0 = np.array(S_max_tu0)
S_max_td0 = np.array(S_max_td0)

plt.figure(1)
plt.title("Sensitivity")
#plt.tight_layout
plt.subplot(2,2,1)
plt.plot(list_gold0,S_max_ru0,label="PVP = 0 nm")
plt.plot(list_gold2,S_max_ru2,label="PVP = 2 nm")
#plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Ru")
plt.subplot(2,2,2)
plt.plot(list_gold0,S_max_rd0,label="PVP = 0 nm")
plt.plot(list_gold2,S_max_rd2,label="PVP = 2 nm")
#plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Rd")
plt.subplot(2,2,3)
plt.plot(list_gold0,S_max_tu0,label="PVP = 0 nm")
plt.plot(list_gold2,S_max_tu2,label="PVP = 2 nm")
plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Tu")
plt.subplot(2,2,4)
plt.plot(list_gold0,S_max_td0,label="PVP = 0 nm")
plt.plot(list_gold2,S_max_td2,label="PVP = 2 nm")
plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Td")
plt.tight_layout
plt.savefig("sensitivity_RTall_allgold_pvp0-2_v2.pdf")
plt.savefig("sensitivity_RTall_allgold_pvp0-2_v2.jpg")
plt.show(block=False)

# plt.figure(2)
# plt.plot(list_gold0,S_max_ru0,label="PVP = 0 nm")
# plt.xlabel("Gold thickness (nm)")
# plt.ylabel("Ru pvp0")
# plt.show(block=False)

# plt.figure(3)
# plt.plot(list_gold2,S_max_ru2,label="PVP = 2 nm")
# plt.xlabel("Gold thickness (nm)")
# plt.ylabel("Ru pvp2")

# plt.show(block=False)

# plt.figure(4)
# plt.plot(list_gold0,S_max_rd0,label="PVP = 0 nm")
# plt.ylabel("Rd pvp0")
# plt.xlabel("Gold thickness (nm)")

# plt.show(block=False)


# plt.figure(5)
# plt.plot(list_gold2,S_max_rd2,label="PVP = 0 nm")
# plt.xlabel("Gold thickness (nm)")
# plt.ylabel("Rd pvp2")

# plt.show(block=False)

# plt.figure(6)
# plt.plot(list_gold0,S_max_tu0,label="PVP = 0 nm")
# plt.xlabel("Gold thickness (nm)")
# plt.ylabel("Tu pvp0")

# plt.show(block=False)

# plt.figure(7)
# plt.plot(list_gold2,S_max_tu2,label="PVP = 2 nm")
# plt.xlabel("Gold thickness (nm)")
# plt.ylabel("Tu pvp2")

# plt.show(block=False)

# plt.figure(8)
# plt.plot(list_gold0,S_max_td0,label="PVP = 0 nm")
# plt.xlabel("Gold thickness (nm)")
# plt.ylabel("Td pvp0")

# plt.show(block=False)

# plt.figure(9)
# plt.plot(list_gold2,S_max_td2,label="PVP = 2 nm")
# plt.xlabel("Gold thickness (nm)")
# plt.ylabel("Td pv20")

# plt.show(block=False)

plt.figure(10)
plt.title("Shift Reso sensitivity")
#plt.tight_layout
plt.subplot(2,2,1)
plt.plot(list_gold0,lam_s_ru0,label="PVP = 0 nm")
plt.plot(list_gold2,lam_s_ru2,label="PVP = 2 nm")
#plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Ru")
plt.subplot(2,2,2)
plt.plot(list_gold0,lam_s_rd0,label="PVP = 0 nm")
plt.plot(list_gold2,lam_s_rd2,label="PVP = 2 nm")
#plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Rd")
plt.subplot(2,2,3)
plt.plot(list_gold0,lam_s_tu0,label="PVP = 0 nm")
plt.plot(list_gold2,lam_s_tu2,label="PVP = 2 nm")
plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Tu")
plt.subplot(2,2,4)
plt.plot(list_gold0,lam_s_td0,label="PVP = 0 nm")
plt.plot(list_gold2,lam_s_td2,label="PVP = 2 nm")
plt.xlabel("Gold thickness (nm)")
plt.legend()
plt.ylabel("Td")
plt.tight_layout
plt.savefig("Wavelength_sensitivity_RTall_allgold_pvp0-2_v2.pdf")
plt.savefig("Wavelength_sensitivity_RTall_allgold_pvp0-2_v2.jpg")
plt.show(block=False)

plt.show()