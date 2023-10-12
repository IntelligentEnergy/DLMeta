'''
Author: Radillus
Date: 2023-05-22 00:44:38
LastEditors: Radillus
LastEditTime: 2023-09-29 13:27:08
Description: 

Copyright (c) 2023 by Radillus, All Rights Reserved. 
'''
import meep as mp
import numpy as np


mp.verbosity.set(0)


def simulate(len_mat:np.ndarray) -> np.ndarray:
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

    up_to_scale = (np.ceil(resolution*gp*2/len_mat.shape[0]).astype(int), np.ceil(resolution*gp*2/len_mat.shape[1]).astype(int))
    len_mat = np.repeat(len_mat, up_to_scale[0], axis=0)
    len_mat = np.repeat(len_mat, up_to_scale[1], axis=1)

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
        mp.Vector3(1, *len_mat.shape),
        medium1=air,
        medium2=Si,
        weights=len_mat,
        do_averaging=False,
    )
    geometry.append(mp.Block(
        material=lens_area,
        size=len_size,
        center=len_pt,
    ))

    sim = mp.Simulation(
        cell_size=cell_size,
        resolution=resolution,
        geometry=geometry,
        sources=sources,
        boundary_layers=pml_layers,
        k_point=k_point,
        ensure_periodicity=False,
    )

    dft_obj = sim.add_dft_fields([mp.Ez], fcen, 0, 1, center=mon_pt, size=mon_size)

    sim.run(until_after_sources=mp.stop_when_dft_decayed(tol=1e-7, minimum_run_time=100, maximum_run_time=1000))

    Ez = sim.get_dft_array(dft_obj, mp.Ez, 0)
    epsilon = sim.get_array(center=len_pt, size=mon_size, component=mp.Dielectric, snap=True)

    return Ez[6:-6, 6:-6], epsilon[6:-6, 6:-6]
