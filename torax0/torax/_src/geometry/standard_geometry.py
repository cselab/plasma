from collections.abc import Mapping
import dataclasses
import logging
import jax
import numpy as np
import scipy
from torax._src import array_typing
from torax._src import constants
from torax._src.geometry import geometry
from torax._src.geometry import geometry_loader
from torax._src.torax_pydantic import interpolated_param_2d
import typing_extensions

_RHO_SMOOTHING_LIMIT = 0.1


@jax.tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class StandardGeometry(geometry.Geometry):
    Ip_from_parameters: bool = dataclasses.field(metadata=dict(static=True))
    Ip_profile_face: array_typing.FloatVectorFace
    psi: array_typing.FloatVectorCell
    psi_from_Ip: array_typing.FloatVectorCell
    psi_from_Ip_face: array_typing.FloatVectorFace
    j_total: array_typing.Array
    j_total_face: array_typing.FloatVectorFace
    delta_upper_face: array_typing.FloatVectorFace
    delta_lower_face: array_typing.FloatVectorFace


@dataclasses.dataclass(frozen=True)
class StandardGeometryIntermediates:
    geometry_type: geometry.GeometryType
    Ip_from_parameters: bool
    R_major: array_typing.FloatScalar
    a_minor: array_typing.FloatScalar
    B_0: array_typing.FloatScalar
    psi: array_typing.Array
    Ip_profile: array_typing.Array
    Phi: array_typing.Array
    R_in: array_typing.Array
    R_out: array_typing.Array
    F: array_typing.Array
    int_dl_over_Bp: array_typing.Array
    flux_surf_avg_1_over_R: array_typing.Array
    flux_surf_avg_1_over_R2: array_typing.Array
    flux_surf_avg_Bp2: array_typing.Array
    flux_surf_avg_RBp: array_typing.Array
    flux_surf_avg_R2Bp2: array_typing.Array
    flux_surf_avg_B2: array_typing.Array
    flux_surf_avg_1_over_B2: array_typing.Array
    delta_upper_face: array_typing.Array
    delta_lower_face: array_typing.Array
    elongation: array_typing.Array
    vpr: array_typing.Array
    n_rho: int
    hires_factor: int
    z_magnetic_axis: array_typing.FloatScalar | None

    def __post_init__(self):
        if self.flux_surf_avg_Bp2[-1] < 1e-10:
            rhon = np.sqrt(self.Phi / self.Phi[-1])
            spline = lambda rho, data, x, bc_type: scipy.interpolate.CubicSpline(
                rho[:-1],
                data[:-1],
                bc_type=bc_type,
            )(x)
            flux_surf_avg_Bp2_edge = spline(
                rhon,
                self.flux_surf_avg_Bp2,
                1.0,
                bc_type='not-a-knot',
            )
            int_dl_over_Bp_edge = spline(
                rhon,
                self.int_dl_over_Bp,
                1.0,
                bc_type='not-a-knot',
            )
            g2_edge_ratio = (flux_surf_avg_Bp2_edge * int_dl_over_Bp_edge**
                             2) / (self.flux_surf_avg_Bp2[-2] *
                                   self.int_dl_over_Bp[-2]**2)
            if g2_edge_ratio > 1.0:
                bc_type = 'not-a-knot'
            else:
                bc_type = 'natural'
            set_edge = lambda array: spline(rhon, array, 1.0, bc_type)
            self.int_dl_over_Bp[-1] = set_edge(self.int_dl_over_Bp)
            self.flux_surf_avg_Bp2[-1] = set_edge(self.flux_surf_avg_Bp2)
            self.flux_surf_avg_1_over_R2[-1] = set_edge(
                self.flux_surf_avg_1_over_R2)
            self.flux_surf_avg_RBp[-1] = set_edge(self.flux_surf_avg_RBp)
            self.flux_surf_avg_R2Bp2[-1] = set_edge(self.flux_surf_avg_R2Bp2)
            self.vpr[-1] = set_edge(self.vpr)
        rhon = np.sqrt(self.Phi / self.Phi[-1])
        idx_limit = np.argmin(np.abs(rhon - _RHO_SMOOTHING_LIMIT))
        self.flux_surf_avg_Bp2[:] = _smooth_savgol(self.flux_surf_avg_Bp2,
                                                   idx_limit, 2)
        self.flux_surf_avg_R2Bp2[:] = _smooth_savgol(self.flux_surf_avg_R2Bp2,
                                                     idx_limit, 2)
        self.flux_surf_avg_RBp[:] = _smooth_savgol(self.flux_surf_avg_RBp,
                                                   idx_limit, 1)
        self.vpr[:] = _smooth_savgol(self.vpr, idx_limit, 1)

    @classmethod
    def from_chease(
        cls,
        geometry_directory: str | None,
        geometry_file: str,
        Ip_from_parameters: bool,
        n_rho: int,
        R_major: float,
        a_minor: float,
        B_0: float,
        hires_factor: int,
    ) -> typing_extensions.Self:
        chease_data = geometry_loader.load_geo_data(
            geometry_directory, geometry_file,
            geometry_loader.GeometrySource.CHEASE)
        psiunnormfactor = R_major**2 * B_0
        psi = chease_data['PSIchease=psi/2pi'] * psiunnormfactor * 2 * np.pi
        Ip_chease = (chease_data['Ipprofile'] / constants.CONSTANTS.mu_0 *
                     R_major * B_0)
        Phi = (chease_data['RHO_TOR=sqrt(Phi/pi/B0)'] *
               R_major)**2 * B_0 * np.pi
        R_in_chease = chease_data['R_INBOARD'] * R_major
        R_out_chease = chease_data['R_OUTBOARD'] * R_major
        F = chease_data['T=RBphi'] * R_major * B_0
        int_dl_over_Bp = (chease_data['Int(Rdlp/|grad(psi)|)=Int(Jdchi)'] *
                          R_major / B_0)
        flux_surf_avg_1_over_R = chease_data['<1/R>profile'] / R_major
        flux_surf_avg_1_over_R2 = chease_data['<1/R**2>'] / R_major**2
        flux_surf_avg_Bp2 = chease_data['<Bp**2>'] * B_0**2
        flux_surf_avg_RBp = chease_data[
            '<|grad(psi)|>'] * psiunnormfactor / R_major
        flux_surf_avg_R2Bp2 = (chease_data['<|grad(psi)|**2>'] *
                               psiunnormfactor**2 / R_major**2)
        flux_surf_avg_B2 = chease_data['<B**2>'] * B_0**2
        flux_surf_avg_1_over_B2 = chease_data['<1/B**2>'] / B_0**2
        rhon = np.sqrt(Phi / Phi[-1])
        vpr = 4 * np.pi * Phi[-1] * rhon / (F * flux_surf_avg_1_over_R2)
        return cls(
            geometry_type=geometry.GeometryType.CHEASE,
            Ip_from_parameters=Ip_from_parameters,
            R_major=np.array(R_major),
            a_minor=np.array(a_minor),
            B_0=np.array(B_0),
            psi=psi,
            Ip_profile=Ip_chease,
            Phi=Phi,
            R_in=R_in_chease,
            R_out=R_out_chease,
            F=F,
            int_dl_over_Bp=int_dl_over_Bp,
            flux_surf_avg_1_over_R=flux_surf_avg_1_over_R,
            flux_surf_avg_1_over_R2=flux_surf_avg_1_over_R2,
            flux_surf_avg_Bp2=flux_surf_avg_Bp2,
            flux_surf_avg_RBp=flux_surf_avg_RBp,
            flux_surf_avg_R2Bp2=flux_surf_avg_R2Bp2,
            flux_surf_avg_B2=flux_surf_avg_B2,
            flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
            delta_upper_face=chease_data['delta_upper'],
            delta_lower_face=chease_data['delta_bottom'],
            elongation=chease_data['elongation'],
            vpr=vpr,
            n_rho=n_rho,
            hires_factor=hires_factor,
            z_magnetic_axis=None,
        )

    @classmethod
    def _from_fbt(
        cls,
        LY: Mapping[str, np.ndarray],
        L: Mapping[str, np.ndarray],
        Ip_from_parameters: bool = True,
        n_rho: int = 25,
        hires_factor: int = 4,
    ) -> typing_extensions.Self:
        R_major = LY['rgeom'][-1]
        B_0 = LY['rBt'] / R_major
        a_minor = LY['aminor'][-1]
        if 'FtPVQ' in LY:
            Phi = LY['FtPVQ']
        else:
            logging.warning(
                'FtPVQ not found in LY, using FtPQ instead. Please upgrade to'
                ' a newer version of MEQ as the source of the LY data. This will'
                ' throw an error in a future version.')
            Phi = LY['FtPQ']
        rhon = np.sqrt(Phi / Phi[-1])
        psi = L['pQ']**2 * (LY['FB'] - LY['FA']) + LY['FA']
        LY_Q1Q = np.where(LY['Q1Q'] != 0, LY['Q1Q'], constants.CONSTANTS.eps)
        flux_surf_avg_B2 = B_0**2 / np.sqrt(1.0 - LY['epsilon']**2)
        flux_surf_avg_1_over_B2 = B_0**-2 * (1.0 + 1.5 * LY['epsilon']**2)
        return cls(
            geometry_type=geometry.GeometryType.FBT,
            Ip_from_parameters=Ip_from_parameters,
            R_major=R_major,
            a_minor=a_minor,
            B_0=B_0,
            psi=psi,
            Phi=Phi,
            Ip_profile=np.abs(LY['ItQ']),
            R_in=LY['rgeom'] - LY['aminor'],
            R_out=LY['rgeom'] + LY['aminor'],
            F=np.abs(LY['TQ']),
            int_dl_over_Bp=1 / LY_Q1Q,
            flux_surf_avg_1_over_R=LY['Q0Q'],
            flux_surf_avg_1_over_R2=LY['Q2Q'],
            flux_surf_avg_Bp2=np.abs(LY['Q3Q']) / (4 * np.pi**2),
            flux_surf_avg_RBp=np.abs(LY['Q5Q']) / (2 * np.pi),
            flux_surf_avg_R2Bp2=np.abs(LY['Q4Q']) / (2 * np.pi)**2,
            flux_surf_avg_B2=flux_surf_avg_B2,
            flux_surf_avg_1_over_B2=flux_surf_avg_1_over_B2,
            delta_upper_face=LY['deltau'],
            delta_lower_face=LY['deltal'],
            elongation=LY['kappa'],
            vpr=4 * np.pi * Phi[-1] * rhon / (np.abs(LY['TQ']) * LY['Q2Q']),
            n_rho=n_rho,
            hires_factor=hires_factor,
            z_magnetic_axis=LY['zA'],
        )


def build_standard_geometry(
    intermediate: StandardGeometryIntermediates, ) -> StandardGeometry:
    rho_intermediate = np.sqrt(intermediate.Phi / (np.pi * intermediate.B_0))
    rho_norm_intermediate = rho_intermediate / rho_intermediate[-1]
    C1 = intermediate.int_dl_over_Bp
    C0 = intermediate.flux_surf_avg_RBp * C1
    C2 = intermediate.flux_surf_avg_1_over_R2 * C1
    C3 = intermediate.flux_surf_avg_Bp2 * C1
    C4 = intermediate.flux_surf_avg_R2Bp2 * C1
    g0 = C0 * 2 * np.pi
    g1 = C1 * C4 * 4 * np.pi**2
    g2 = C1 * C3 * 4 * np.pi**2
    g3 = C2[1:] / C1[1:]
    g3 = np.concatenate((np.array([1 / intermediate.R_in[0]**2]), g3))
    g2g3_over_rhon = g2[1:] * g3[1:] / rho_norm_intermediate[1:]
    g2g3_over_rhon = np.concatenate((np.zeros(1), g2g3_over_rhon))
    dpsidrhon = (
        intermediate.Ip_profile[1:] *
        (16 * constants.CONSTANTS.mu_0 * np.pi**3 * intermediate.Phi[-1]) /
        (g2g3_over_rhon[1:] * intermediate.F[1:]))
    dpsidrhon = np.concatenate((np.zeros(1), dpsidrhon))
    psi_from_Ip = scipy.integrate.cumulative_trapezoid(
        y=dpsidrhon,
        x=rho_norm_intermediate,
        initial=0.0,
    )
    psi_from_Ip += intermediate.psi[0]
    psi_from_Ip[-1] = psi_from_Ip[-2] + (
        16 * constants.CONSTANTS.mu_0 * np.pi**3 *
        intermediate.Phi[-1]) * intermediate.Ip_profile[-1] / (
            g2g3_over_rhon[-1] * intermediate.F[-1]) * (
                rho_norm_intermediate[-1] - rho_norm_intermediate[-2])
    vpr = intermediate.vpr
    spr = vpr * intermediate.flux_surf_avg_1_over_R / (2 * np.pi)
    volume_intermediate = scipy.integrate.cumulative_trapezoid(
        y=vpr, x=rho_norm_intermediate, initial=0.0)
    area_intermediate = scipy.integrate.cumulative_trapezoid(
        y=spr, x=rho_norm_intermediate, initial=0.0)
    dI_tot_drhon = np.gradient(intermediate.Ip_profile, rho_norm_intermediate)
    j_total_face_bulk = dI_tot_drhon[1:] / spr[1:]
    j_total_face_axis = j_total_face_bulk[0]
    j_total = np.concatenate(
        [np.array([j_total_face_axis]), j_total_face_bulk])
    mesh = interpolated_param_2d.Grid1D(nx=intermediate.n_rho)
    rho_b = rho_intermediate[-1]
    rho_face_norm = mesh.face_centers
    rho_norm = mesh.cell_centers
    rho_hires_norm = np.linspace(
        0, 1, intermediate.n_rho * intermediate.hires_factor)
    rho_hires = rho_hires_norm * rho_b
    rhon_interpolation_func = lambda x, y: np.interp(x, rho_norm_intermediate,
                                                     y)
    vpr_face = rhon_interpolation_func(rho_face_norm, vpr)
    vpr = rhon_interpolation_func(rho_norm, vpr)
    spr_face = rhon_interpolation_func(rho_face_norm, spr)
    spr_cell = rhon_interpolation_func(rho_norm, spr)
    spr_hires = rhon_interpolation_func(rho_hires_norm, spr)
    delta_upper_face = rhon_interpolation_func(rho_face_norm,
                                               intermediate.delta_upper_face)
    delta_lower_face = rhon_interpolation_func(rho_face_norm,
                                               intermediate.delta_lower_face)
    delta_face = 0.5 * (delta_upper_face + delta_lower_face)
    elongation = rhon_interpolation_func(rho_norm, intermediate.elongation)
    elongation_face = rhon_interpolation_func(rho_face_norm,
                                              intermediate.elongation)
    Phi_face = rhon_interpolation_func(rho_face_norm, intermediate.Phi)
    Phi = rhon_interpolation_func(rho_norm, intermediate.Phi)
    F_face = rhon_interpolation_func(rho_face_norm, intermediate.F)
    F = rhon_interpolation_func(rho_norm, intermediate.F)
    F_hires = rhon_interpolation_func(rho_hires_norm, intermediate.F)
    psi = rhon_interpolation_func(rho_norm, intermediate.psi)
    psi_from_Ip_face = rhon_interpolation_func(rho_face_norm, psi_from_Ip)
    psi_from_Ip = rhon_interpolation_func(rho_norm, psi_from_Ip)
    j_total_face = rhon_interpolation_func(rho_face_norm, j_total)
    j_total = rhon_interpolation_func(rho_norm, j_total)
    Ip_profile_face = rhon_interpolation_func(rho_face_norm,
                                              intermediate.Ip_profile)
    Rin_face = rhon_interpolation_func(rho_face_norm, intermediate.R_in)
    Rin = rhon_interpolation_func(rho_norm, intermediate.R_in)
    Rout_face = rhon_interpolation_func(rho_face_norm, intermediate.R_out)
    Rout = rhon_interpolation_func(rho_norm, intermediate.R_out)
    g0_face = rhon_interpolation_func(rho_face_norm, g0)
    g0 = rhon_interpolation_func(rho_norm, g0)
    g1_face = rhon_interpolation_func(rho_face_norm, g1)
    g1 = rhon_interpolation_func(rho_norm, g1)
    g2_face = rhon_interpolation_func(rho_face_norm, g2)
    g2 = rhon_interpolation_func(rho_norm, g2)
    g3_face = rhon_interpolation_func(rho_face_norm, g3)
    g3 = rhon_interpolation_func(rho_norm, g3)
    g2g3_over_rhon_face = rhon_interpolation_func(rho_face_norm,
                                                  g2g3_over_rhon)
    g2g3_over_rhon_hires = rhon_interpolation_func(rho_hires_norm,
                                                   g2g3_over_rhon)
    g2g3_over_rhon = rhon_interpolation_func(rho_norm, g2g3_over_rhon)
    gm4 = rhon_interpolation_func(rho_norm,
                                  intermediate.flux_surf_avg_1_over_B2)
    gm4_face = rhon_interpolation_func(rho_face_norm,
                                       intermediate.flux_surf_avg_1_over_B2)
    gm5 = rhon_interpolation_func(rho_norm, intermediate.flux_surf_avg_B2)
    gm5_face = rhon_interpolation_func(rho_face_norm,
                                       intermediate.flux_surf_avg_B2)
    volume_face = rhon_interpolation_func(rho_face_norm, volume_intermediate)
    volume = rhon_interpolation_func(rho_norm, volume_intermediate)
    area_face = rhon_interpolation_func(rho_face_norm, area_intermediate)
    area = rhon_interpolation_func(rho_norm, area_intermediate)
    return StandardGeometry(
        geometry_type=intermediate.geometry_type,
        torax_mesh=mesh,
        Phi=Phi,
        Phi_face=Phi_face,
        R_major=intermediate.R_major,
        a_minor=intermediate.a_minor,
        B_0=intermediate.B_0,
        volume=volume,
        volume_face=volume_face,
        area=area,
        area_face=area_face,
        vpr=vpr,
        vpr_face=vpr_face,
        spr=spr_cell,
        spr_face=spr_face,
        delta_face=delta_face,
        g0=g0,
        g0_face=g0_face,
        g1=g1,
        g1_face=g1_face,
        g2=g2,
        g2_face=g2_face,
        g3=g3,
        g3_face=g3_face,
        g2g3_over_rhon=g2g3_over_rhon,
        g2g3_over_rhon_face=g2g3_over_rhon_face,
        g2g3_over_rhon_hires=g2g3_over_rhon_hires,
        gm4=gm4,
        gm4_face=gm4_face,
        gm5=gm5,
        gm5_face=gm5_face,
        F=F,
        F_face=F_face,
        F_hires=F_hires,
        R_in=Rin,
        R_in_face=Rin_face,
        R_out=Rout,
        R_out_face=Rout_face,
        Ip_from_parameters=intermediate.Ip_from_parameters,
        Ip_profile_face=Ip_profile_face,
        psi=psi,
        psi_from_Ip=psi_from_Ip,
        psi_from_Ip_face=psi_from_Ip_face,
        j_total=j_total,
        j_total_face=j_total_face,
        delta_upper_face=delta_upper_face,
        delta_lower_face=delta_lower_face,
        elongation=elongation,
        elongation_face=elongation_face,
        spr_hires=spr_hires,
        rho_hires_norm=rho_hires_norm,
        rho_hires=rho_hires,
        Phi_b_dot=np.asarray(0.0),
        _z_magnetic_axis=intermediate.z_magnetic_axis,
    )


def _smooth_savgol(
    data: np.ndarray,
    idx_limit: int,
    polyorder: int,
    window_length: int = 5,
    preserve_first: bool = True,
) -> np.ndarray:
    if idx_limit == 0:
        return data
    smoothed_data = scipy.signal.savgol_filter(data,
                                               window_length,
                                               polyorder,
                                               mode='nearest')
    first_point = data[0] if preserve_first else smoothed_data[0]
    return np.concatenate([
        np.array([first_point]), smoothed_data[1:idx_limit], data[idx_limit:]
    ])
