import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

lim = 5

def visualize(data):
    R, P = data.shape[:2]
    X = [[float(data[0][i][0])] for i in range(P)]
    Y = [[float(data[0][i][1])] for i in range(P)]
    Z = [[float(data[0][i][2])] for i in range(P)]
    fig = plt.figure()
    ax = plt.subplot(111, aspect= 'equal', projection= '3d')
    lines = []
    for i in range(P):
        lines.extend(ax.plot(
            [float(data[0][i][0])], [float(data[0][i][1])], [float(data[0][i][2])], 'r-'
        ))
    
    plt.xlim(-lim, lim)
    plt.ylim(-lim, lim)
    ax.set_zlim(0, lim)
    count = 0

    def init():
        return lines

    def animate(frame):
        nonlocal X, Y, Z, count
        count += 1
        print(count)
        if count < R:
            for i in range(P):
                X[i].append(float(data[count][i][0]))
                Y[i].append(float(data[count][i][1]))
                Z[i].append(float(data[count][i][2]))
                # Z[i].append(0.)
                lines[i].set_data(X[i], Y[i])
                lines[i].set_3d_properties(Z[i])
        return lines

    anim = animation.FuncAnimation(fig, animate, init_func= init, blit= True, interval= 100)
    plt.show()

TEXT_FORMAT = True

if TEXT_FORMAT:
    with open('trajectories.txt', 'r') as f:
        P, R = map(int, f.readline().split())
        data = []
        for i in range(R):
            data.extend(list(map(float, f.readline().split())))
        data = np.array(data, dtype= np.float32).reshape(R, P, 3)
    visualize(data)
else:
    with open('trajectories.bin', 'rb') as f:
        P = np.frombuffer(f.read(4), dtype= np.int32)[0]
        R = np.frombuffer(f.read(4), dtype= np.int32)[0]
        data = np.fromfile(f, dtype= np.float32, count= R * P * 3).reshape(R, P, 3)
    visualize(data)