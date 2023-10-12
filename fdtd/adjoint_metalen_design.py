'''
Author: Radillus
Date: 2023-05-22 00:44:38
LastEditors: Radillus
LastEditTime: 2023-10-10 18:15:05
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import meep as mp
import meep.adjoint as mpa
from numpy import ndarray
import numpy as np
from autograd import numpy as npa
import torch
from lion_pytorch import Lion
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


# mp.verbosity(3)

writer = SummaryWriter()

DESIGN_SIZE = 100

dpml = 1.0       # PML层厚度 μm
dsub = 0.5       # 基底厚度 μm
dmon = 0.3     # 模型顶端与监视器距离 μm
dpad = 0.5       # 监视器与PML层距离 μm
gh = 0.5         # 设计区域高度 μm
lcen = 0.98      # 中心波长 μm
gp = 5.0         # 周期 μm
resolution = 20

fcen = 1/lcen
df = 0.2*fcen
sx = dpml + dsub + gh + dmon + dpad + dpml
sy = gp
sz = gp

k_point = mp.Vector3(0, 0, 0)
air = mp.Medium(index=1.0)
Si = mp.Medium(index=3.4)
SiO2 = mp.Medium(index=1.5)

cell_size = mp.Vector3(sx, sy, sz)

pml_layers = [mp.PML(thickness=dpml, direction=mp.X)]

src_pt = mp.Vector3(-0.5*sx+dpml+0.5*dsub, 0, 0)
src_size = mp.Vector3(0, sy, sz)

len_pt = mp.Vector3(-0.5*sx+dpml+dsub+0.5*gh, 0, 0)
len_size = mp.Vector3(gh, sy, sz)
base_pt = mp.Vector3(-0.5*sx+0.5*(dpml+dsub), 0, 0)
base_size = mp.Vector3(dpml+dsub, sy, sz)

mon_pt = mp.Vector3(-0.5*sx+dpml+dsub+gh+dmon, 0, 0)
mon_size = mp.Vector3(0, sy*1.1, sz*1.1)

sources = [mp.Source(
    mp.GaussianSource(fcen, fwidth=df, is_integrated=True),
    component=mp.Ez,
    center=src_pt,
    size=src_size,
)]

# base
geometry = [mp.Block(
    material=SiO2,
    size=base_size,
    center=base_pt,
)]

# lens
lens_area = mp.MaterialGrid(
    mp.Vector3(1, DESIGN_SIZE, DESIGN_SIZE),
    medium1=air,
    medium2=Si,
    do_averaging=False,
)

design_region = mpa.DesignRegion(
    lens_area,
    size=len_size,
    center=len_pt,
)

geometry.append(mp.Block(
    material=lens_area,
    size=len_size,
    center=len_pt,
))

sim = mp.Simulation(
    cell_size=cell_size,
    resolution=20,
    geometry=geometry,
    sources=sources,
    boundary_layers=pml_layers,
    k_point=k_point,
    ensure_periodicity=False,
)

dft_obj = mpa.FourierFields(sim, mp.Volume(center=mon_pt, size=mon_size), mp.Ez)

def _square_wavefront(x1:float, x2:float, y1:float, y2:float, x_num:int, y_num:int) -> ndarray:
    x = np.linspace(x1, x2, x_num)
    y = np.linspace(y1, y2, y_num)
    xs, ys = np.meshgrid(x, y)
    Lambda = 0.98
    f = 1000
    target_wf = 2*np.pi/Lambda * (f-np.sqrt(xs**2+ys**2+f**2))
    target_wf = np.mod(target_wf, 2*np.pi) - np.pi
    return target_wf

def get_target(x:float = None, y:float = None) -> ndarray:
    if x is None:
        x = np.random.uniform(-100, 100)
    if y is None:
        y = np.random.uniform(-100, 100)
    x_start = (x - 0.5) * 5
    x_end = (x + 0.5) * 5
    Y_start = (y - 0.5) * 5
    y_end = (y + 0.5) * 5
    return _square_wavefront(x_start, x_end,  Y_start, y_end, 100, 100)

TARGET_ANGLE = get_target()
writer.add_image('target', TARGET_ANGLE, dataformats='HW')

j_time = 0
def J(field):
    global j_time
    angle = npa.angle(field[0, 6:-6, 6:-6])
    loss = npa.mean((npa.sin(npa.abs(angle - TARGET_ANGLE) - npa.pi/2)+1)/2)
    if j_time % 2 == 0:
        writer.add_image('field/real', np.real(field[0, 6:-6, 6:-6]), j_time//2, dataformats='HW')
        writer.add_image('field/imag', np.imag(field[0, 6:-6, 6:-6]), j_time//2, dataformats='HW')
        writer.add_image('field/phase', np.angle(field[0, 6:-6, 6:-6]), j_time//2, dataformats='HW')
    j_time += 1
    return loss

opt = mpa.OptimizationProblem(
    simulation=sim,
    objective_functions=[J],
    objective_arguments=[dft_obj],
    design_regions=[design_region],
    fcen=fcen,
    df=0,
    nf=1,
    decay_by=1e-6,
    minimum_run_time=100,
    maximum_run_time=500,
)

design = torch.rand(DESIGN_SIZE*DESIGN_SIZE, requires_grad=True)
optimizer = Lion([design], lr=0.1)
schedular = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, verbose=True, factor=0.318, threshold=1e-5)
for step in range(1000):
    writer.add_image('design', design.detach().numpy().reshape(DESIGN_SIZE, DESIGN_SIZE), step, dataformats='HW')
    f0, dJ_du = opt([design.detach().numpy()])
    writer.add_scalar('f0', f0, step)
    writer.add_image('dJ_du', np.squeeze(dJ_du).reshape(DESIGN_SIZE, DESIGN_SIZE), step, dataformats='HW')
    design.grad = torch.from_numpy(np.squeeze(dJ_du)).float()
    optimizer.step()
    schedular.step(f0)
    with torch.no_grad():
        design = torch.clamp_(design, 0, 1)
