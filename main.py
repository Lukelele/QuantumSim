import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numba
from numba import jitclass
        
class System:
    def __init__(self):
        self.simulation_time = 50e-1
        self.dt = 1e-6
        self.simulation_space = 3
        self.dx = 0.02
        self.x = np.linspace(-self.simulation_space, self.simulation_space, int(self.simulation_space/self.dx))
        self.initial_wavefunction = np.exp(-self.x**2/(2*1**2) + 1j*5*self.x) / np.sqrt(1*np.sqrt(np.pi))
        self.potential = self.x ** 2
        self.wavefunction = np.zeros([int(self.simulation_time / self.dt), len(self.initial_wavefunction)]).astype(complex)
        self.wavefunction[0] = self.initial_wavefunction

    def set_initial_wavefunction(self, initial_wavefunction):
        self.wavefunction[0] = initial_wavefunction

    def set_potential(self, potential):
        self.potential = potential


    def simulate(self):
        for t in range(int(self.simulation_time / self.dt) - 1):
            self.wavefunction[t][0] = 0
            self.wavefunction[t][-1] = 0
            for x in range(1, len(self.wavefunction[t]) - 1):
                self.wavefunction[t+1][x] = self.wavefunction[t][x] + 1j/2 * (self.dt/(self.dx**2)) * (self.wavefunction[t][x+1] - 2 * self.wavefunction[t][x] + self.wavefunction[t][x-1]) - 1j * self.dt * self.potential[x] * self.wavefunction[t][x]

            # norm = np.sum(np.abs(self.wavefunction[t+1])**2)
            # self.wavefunction[t+1] = self.wavefunction[t+1] / norm
        return self.wavefunction


system = System()
psi = system.simulate()


fig = plt.figure()
ax = fig.add_subplot()
#ax.set_xlim(-system.simulation_space, system.simulation_space)
ax.set_ylim(0, 3)
line, = ax.plot(np.abs(psi[0]) ** 2)

def update(i):
    line.set_ydata(np.abs(psi[i]) ** 2)
    ax.set_title(i)
    return line,

ani = animation.FuncAnimation(fig, update, frames=len(psi), interval=1, blit=True)
plt.show()



plt.show()
