import zerorpc
import numpy as np
import matplotlib.pyplot as plt

DMD_SERVER_IP = '192.168.1.154'
DMD_SERVER_PORT = '6001'

client = zerorpc.Client()
client.connect(f"tcp://{DMD_SERVER_IP}:{DMD_SERVER_PORT}")

print('Server response:', client.hello())

profile_dmd_x = np.arange(1920)
profile_dmd_y = np.ones_like(profile_dmd_x)
x_center = 960
DX = 300
dx = 10
profile_dmd_y[x_center-DX-dx:x_center-DX+dx] = 0.
profile_dmd_y[x_center+DX-dx:x_center+DX+dx] = 0.
profile_dmd_y[x_center-2*DX-dx:x_center-2*DX+dx] = 0.
profile_dmd_y[x_center+2*DX-dx:x_center+2*DX+dx] = 0.
profile_dmd_y[x_center-3*DX-dx:x_center-3*DX+dx] = 0.
profile_dmd_y[x_center+3*DX-dx:x_center+3*DX+dx] = 0.
profile_dmd_y[x_center-dx:x_center+dx] = 0.

client.load_1d_profile_dmd(profile_dmd_y.tolist())

client.close()