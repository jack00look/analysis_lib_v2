import zerorpc
import numpy as np
import matplotlib.pyplot as plt

DMD_SERVER_IP = '192.168.1.154'
DMD_SERVER_PORT = '6001'

client = zerorpc.Client()
client.connect(f"tcp://{DMD_SERVER_IP}:{DMD_SERVER_PORT}")

print('Server response:', client.hello())

profile_x = np.arange(2048)*1.019
profile_y = np.ones_like(profile_x)
x_center = 1040#1067
w = 150
x_min = x_center - w
x_max = x_center + w
profile_y[(profile_x >= x_min) & (profile_x <= x_max)] = 0.2

last_2d_image = client.get_last_2d_image()
fig,ax = plt.subplots()
ax.imshow(last_2d_image, cmap='viridis')
# print('Last 2D image shape:', last_2d_image)
# plt.imshow(last_2d_image, cmap='viridis')
# plt.colorbar(label='Intensity')
# plt.title('Last 2D Image from DMD Server')
# plt.xlabel('Pixel X')
# plt.ylabel('Pixel Y')
# plt.show()

last_atoms_profile = client.get_last_atoms_profile()
print(last_atoms_profile.keys())
try:
    x_last = last_atoms_profile['x']
    y_last = last_atoms_profile['y']
    plt.plot(x_last, y_last, label='Last Atoms Profile')
except Exception as e:
    print('Error retrieving last atoms profile:', e)
fig1, ax1 = plt.subplots()
ax1.plot(profile_x, profile_y, label='Generated Profile')
ax1.set_xlabel('Position (µm)')
ax1.set_ylabel('Intensity (a.u.)')
ax1.set_title('Generated Load Profile for DMD')
ax1.legend()
ax1.grid()
plt.show()

print('sending profile')
profile_y_dmd = np.zeros((1080,1920))
profile_y_dmd[450:550,910:1010] = 1.
#profile_y_dmd[500:550,:] = 1.
# y_c=500
# x_c = 960
# dx = 40
# dy = 40
# profile_y_dmd[y_c-dy:y_c+dy,x_c-dx:x_c+dx] = 1.
# dx_hole = 20
# dy_hole = 20
# profile_y_dmd[y_c-dy_hole:y_c+dy_hole,x_c-dx_hole:x_c+dx_hole] = 0.

profile_x_atoms = np.arange(2048)*1.019
profile_y_atoms = np.ones_like(profile_x_atoms)
x1 = 898
x2 = 1137
x3 = 970
dx = 5
profile_y_atoms[(profile_x_atoms >= x1-dx) & (profile_x_atoms <= x1+dx)] = 0.
profile_y_atoms[(profile_x_atoms >= x2-dx) & (profile_x_atoms <= x2+dx)] = 0.
profile_y_atoms[(profile_x_atoms >= x3-dx) & (profile_x_atoms <= x3+dx)] = 0.
fig2, ax2 = plt.subplots()
ax2.plot(profile_x_atoms, profile_y_atoms, label='Atoms Profile to Load')
ax2.set_xlabel('Position (µm)')
ax2.set_ylabel('Intensity (a.u.)')
ax2.set_title('Atoms Profile to Load')
ax2.legend()
ax2.grid()
plt.show()

status = client.load_1d_profile(profile_x_atoms.tolist(), profile_y_atoms.tolist())
#status = client.load_raw_image_2d(profile_y_dmd.tolist())
print('status: ', status)

client.close()