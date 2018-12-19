/*******************************************************************************
 * TRANS_NEUT case
 *
 * 
 * Solves equations for 
 *  plasmas density                  Ni
 *  parallel ion velocity            Vi
 *  electron and ion temperatures    Te, Ti
 * and also equations of neutrals:
 *  atom density                     Nn
 *  perpendicular velocity           Vn 
 *  molecule density                 Nm
 *  molecule radial velocity         Vmx
 * 
 * Intended to be run for NZ=1 (i.e. X and Y only) at first,
 * 3D SMBI has also been tested no problem for a short time injection
 *
 *******************************************************************************/

#include <bout.hxx>
#include <boutmain.hxx>
#include <derivs.hxx>

#include <initialprofiles.hxx>
#include <invert_laplace.hxx>
#include <invert_parderiv.hxx>
#include <interpolation.hxx>

#include "hermes-1.hxx"
#include <field_factory.hxx>
#include "div_ops.hxx"
#include "loadmetric.hxx"
#include <bout/constants.hxx>
#include <bout/assert.hxx>


#include <cmath>
#include <math.h>

// Vectors
Vector3D b0xcv; // Curvature term
Vector3D B0vec; // Vector of B0
Vector3D Vm,V;
Vector3D grad_perp_Diffi,grad_perp_Diffn,grad_perp_chii,grad_perp_chie;

//2D Evolving fields from Grid file
Field2D Te_grid,Ti_grid,Ni_grid;
Field2D Te_exp,Ti_exp,Ni_exp;

//2D Psi from Grid file
Field2D psixy,B0;
Field2D phi2D;
// 3D evolving fields
Field3D  Te, Ni, Vi, Ti, Nn,Tn, Vn, Nm ,Vmx,Vort;             // Vi and Vn are paralell velocity
Field3D temp_Ni,temp_Nn,temp_Nm,temp_Ti,temp_Te;         // temporary variables for minivalue protection

// 3D initial profiles Hmode
Field3D Ni0_Hmode,Ti0_Hmode,Te0_Hmode;

// Non-linear coefficientsx
Field3D kappa_Te, kappa_Ti;
Field3D kappa_Te_fl,kappa_Ti_fl;                         // heat flux limit coefficients 

// Thermal Ion and Electron Speed
Field3D V_th_i,V_th_e,V_th_n;

//ionization/CX/rates for Plasmas and Neutrals
Field3D nu_ionz,nu_CX,nu_ionz_n,nu_CX_n,nu_diss,nu_rec;

// Collisional time between electron and ions
Field3D tau_ei;

//ion viscosity
Field3D eta0_i,eta0_n;

// Density/heat diffusion coefficients of plasmas
Field3D Diff_ni_perp,chi_i_perp,chi_e_perp;
Field3D Diffc_ni_step,chic_i_step,chic_e_step;
Field3D Diffc_ni_Hmode,chic_i_Hmode,chic_e_Hmode;

// average over Y direction for Diffscoefs_Hmode 
BoutReal aveYg11J_core;
Field2D aveY_g11,aveY_J, aveY_g11J;

// fluxes averaged over Y
Field3D Flux_Ni_aveY,Flux_Ti_aveY,Flux_Te_aveY;

// Classical Diffusion coefficients of neutrals 
Field3D Diffc_nn_perp,Diffc_nn_par,Diffc_nm_perp;
Field3D Diffc_nn_perp_fl;  //flux limited
// Source Terms
Field3D Si_p,Si_CX,S_diss,S_rec;
Field3D S_pi_ext,S_Ee_ext,S_Ei_ext;      // external sources

///ZYLHJ///
Field3D Dperp2psi,GradPhi2, Dperp2Phi, bracketPhiP, Dperp2Pi,Btemp,bracketphiNi,bracketphiVi,bracketPiVi,Ni_curvature1,Ni_curvature2,bracketphiTe,bracketphiTi,bracketphipei,Te_curvature1,Te_curvature2,Te_curvature3,Te_current1,Ti_curvature1,Ti_curvature2,Ti_curvature3,Ti_current1;
BoutReal mi_me, Upara0, Upara1, Upara2,Jpara0,Jpara1, Jpara2, Jpara3,Nipara0,Vipara0,Tepara0,Tepara1,Tipara0,Tipara1;


// 3D total values
Field3D Nit, Tit, Tet, Vit;

// pressures
Field3D  Pe,Pi,pei,pn,pm;

// parallel pressure gradient
Field3D Grad_par_pei,Grad_par_pn;

// gradient length related quatities
Field3D Grad_par_logNn,Grad_par_Tn;

//pressure gradient length
Field3D dpei_dx,Heavistep;

//radial derivatives for Hmode diffusion coef calculation
Field3D DDX_Ni,DDX_Te,DDX_Ti;
Field3D D2DX2_Ni, D2DX2_Ti, D2DX2_Te;

// Variations for Sheath Boundary condition 
Field3D c_se,Vn_pol,Vn_perp,q_si_kappa,q_se_kappa;
Field3D Gamma_ni,Gamma_nn,Gamma_ni_Diffc,Gamma_nn_Diffc;
Field3D Gamma_ni_wall,Gamma_nn_wall,Gamma_nnw_Diffc;

// spitzer resistivty
Field3D eta_spitzer;

// Bootstrap current 
Field3D Jpar_BS0;
Field3D L31, L32, L34;
Field3D f31, f32ee, f32ei, f34, ft;
Field3D nu_estar, nu_istar, BSal0, BSal;
BoutReal Aratio;

///////////ZYLHJ //

Field3D Telim, Nelim;

Field3D Jpar,phi,U,psi;

//  brackets: b0 x Grad(f) dot Grad(g) / B = [f, g]
//  Method to use: BRACKET_ARAKAWA, BRACKET_STD or BRACKET_SIMPLE

/*
  BRACKET_METHOD bm_exb, bm_mag; // Bracket method for advection terms
  int bracket_method_exb, bracket_method_mag;
*/

// Metric coefficients
Field2D Rxy, Bpxy, Btxy, hthe;
Field2D I;

// Max grid number in X Y Z
int NX,NY,NZ;

// width of private flux (PF) region in Y grid
int jyseps11;

// filter 
BoutReal filter_para;

//Prameter for heat flux limit
Field2D q95;
BoutReal q95_input; 

// Initial profile parameters
BoutReal Te_core,Te_edge,Ti_core,Ti_edge,Ni_core,Ni_edge;
BoutReal Initfile_x0,Initfile_w_ped;
BoutReal x0_ped,width_ped,coef_coregrad_ped,coef_corenlarge_ped;
BoutReal density_unit;                   // Number density [m^-3]

// parameters
BoutReal Te_x, Ti_x,Tm_x, Ni_x, Vi_x, bmag, rho_s, fmei, AA, ZZ;
BoutReal Zi;                  // charge number of ion same as ZZ
BoutReal minimum_val;
BoutReal Mm,Mi,Mn;
BoutReal W_ionz,W_diss,W_bind,W_rec;
BoutReal Lbar,tbar,Lp,Lp_crit,Lnn_min;
BoutReal lambda_ei, lambda_ii;
BoutReal N_cubic_Lbar;           //particle numbers in cubic of Lbar^3
BoutReal nu_hat, mui_hat, wci, nueix, nuiix;

BoutReal Nm0,Vm0;

BoutReal Diffc_ni_perp,Difft_ni_perp,chic_i_perp,chit_i_perp;
BoutReal chic_e_perp,chit_e_perp,chic_n_perp;
int t_output;

//Boundaries of 3D-const fueling flux
BoutReal CF_BC_x0,CF_BC_y0,CF_BC_y1,CF_BC_z0,CF_BC_z1;  // SMBI_LFS
BoutReal Sheath_BC_x0;
BoutReal Rate_recycle,alpha_vn,angle_B_plate,Tn_plate,Vn_th_plate;
BoutReal Lni_wall;      //Gradient length Ni at wall
int Sheath_width;

//radial global x 
BoutReal x_rela;

// parameters of external sources 
BoutReal x0_extS,width_extS,coef_coregrad_extS;
BoutReal amp_spi_ext,amp_see_ext,amp_sei_ext;

//constant gradient at xin with auto unit
BoutReal dNidx_xin_au,dTidx_xin_au,dTedx_xin_au;
BoutReal psi_xout_y0,psi_axis,psi_bndry;

//Evolving Equation control

bool  NOT_SOLVE_FOR_Vi;
//step function of diffusion coefficients
BoutReal diffusion_coef_step0, diffusion_coef_step1;
bool  diffusion_coef_step_function;

// diffusion coefficients of Hmode profiles
BoutReal diffusion_coef_Hmode0, diffusion_coef_Hmode1;
bool  diffusion_coef_Hmode, diffusion_PF_constant;

//logical paramter
bool profiles_lasttimestep,load_grid_profiles, initial_profile_exp, initial_profile_linear;
bool load_experiment_profiles;
bool initial_profile_Hmode,initial_PF_edgeval;
bool external_sources,extsrcs_balance_diffusion;
bool noshear, include_curvature,Turb_Diff_on;
bool SMBI_LFS, Sheath_BC,SBC_particle_recycle,Wall_particle_recycle;
bool term_GparkappaTe_GparTe;
bool terms_recombination,terms_Gradpar_pn,terms_Gradpar_eta0n;
bool terms_NnGradpar_Vn,terms_Diffcnn_par;
bool terms_Gradperp_diffcoefs;
bool nlfilter_noisy_data,nlfilter_Gradpar_logNn,nlfilter_Vnpol;
bool BScurrent,spitzer_resist;
bool term_driftExB,term_driftcurvature,term_drift_diamag,term_parallel_current;
bool reversed_mag_field;   // reversed magnetic field///
bool phi3d, split_n0, newXZsolver;bool boussinesq; // Use a fixed density (Nnorm) in the vorticity equation

  LaplaceXY *laplacexy; // Laplacian solver in X-Y (n=0)
  Laplacian *phiSolver; // Old Laplacian in X-Z
  LaplaceXZ *newSolver; // New Laplacian in X-Z


const Field3D ret_const_flux_BC(const Field3D &var, const BoutReal value);
void Diag_neg_value (const Field3D &f1,const Field3D &f2,const Field3D &f3, const Field3D &f4);
const Field3D field_larger(const Field3D &f, const BoutReal limit);
const Field3D field_smaller(const Field3D &f, const BoutReal limit);
void SBC_Dirichlet_SWidth1(Field3D &var, const Field3D &value);
void SBC_Dirichlet(Field3D &var, const Field3D &value);
void SBC_Gradpar(Field3D &var, const Field3D &value);
void SBC_yup_eq(Field3D &var, const Field3D &value);
void SBC_ydown_eq(Field3D &var, const Field3D &value);
void SBC_yup_Grad_par(Field3D &var, const Field3D &value);
void SBC_ydown_Grad_par(Field3D &var, const Field3D &value);
void WallBC_Xout_GradX(Field3D &var, const Field3D &value);
void WallBC_Xout_GradX_len(Field3D &var, BoutReal value);

const Field3D BS_ft(const int index);
const Field3D F31(const Field3D input);
const Field3D F32ee(const Field3D input);
const Field3D F32ei(const Field3D input);

Field3D eta;
BoutReal vacuum_pressure;
BoutReal vacuum_trans; // Transition width
Field3D vac_mask;
BoutReal vac_lund, core_lund;       // Lundquist number S = (Tau_R / Tau_A). -ve -> infty
BoutReal vac_resist,  core_resist;

//const BoutReal PI = 3.14159265;
const BoutReal MU0 = 4.0e-7*PI;
//const BoutReal Mi = 2.0*1.6726e-27; // Ion mass
const BoutReal mass_unit = 2.0*1.6726e-27; // Proton mass ZYLHJ
const BoutReal KB = 1.38065e-23;     // Boltamann constant
const BoutReal ee = 1.602e-19;       // ln(Lambda)
const BoutReal eV_K = 11605.0;         // 1eV = 11605K

const BoutReal C_fe_sheat=7.;     // coefficient of parallel heat transmission in sheat BC 
const BoutReal C_fi_sheat=2.5;


int physics_init(bool restarting)
{
   
  t_output=1;
  
  output.write("Solving transport equations for Plasmas: Ni, Vi, Ti, Te and Neutrals: Nn Vn \n");

  /////////////// LOAD DATA FROM GRID FILE //////////////

// Load 2D profiles
  mesh->get(Te_grid, "Te0");    // eV
  mesh->get(Ti_grid, "Ti0");    // eV
  mesh->get(Ni_grid, "Ni0");    // hl2a 10^21/m^3

  mesh->get(Te_exp, "Te_exp");    // eV
  mesh->get(Ti_exp, "Ti_exp");    // eV
  mesh->get(Ni_exp, "Ni_exp");    // hl2a 10^19/m^3

  // Load Psi
  mesh->get(psixy,"psixy");         // unit m^2 T
  mesh->get(psi_axis,"psi_axis");
  mesh->get(psi_bndry,"psi_bndry");
  // Load curvature term
  b0xcv.covariant = false; // Read contravariant components
  Vm.covariant = false;
  mesh->get(b0xcv, "bxcv"); // mixed units x: T y: m^-2 z: m^-2
  mesh->get(B0, "Bxy");      //ZYLHJ
  // Load metrics
  GRID_LOAD(Rxy);         // Major radius [m]
  GRID_LOAD2(Bpxy, Btxy); // Poloidal, Toroidal B field [T]
  GRID_LOAD(hthe);        // Poloidal arc length [m / radian]
  //mesh->get(mesh->dx,   "dpsi");
  //mesh->get(mesh->dx, "dx");  //1D test only
  mesh->get(I,    "sinty");// m^-2 T^-1
  mesh->get(NX,"nx");   
  mesh->get(NY,"ny");   

  mesh->get(jyseps11,"jyseps1_1");   


  if(mesh->get(bmag, "bmag")) 
     bmag=1.0;
  if(mesh->get(Lbar, "rmag"))
     Lbar=1.0;  

  // Load normalisation values
  //GRID_LOAD(Te_x);
  //GRID_LOAD(Ti_x);
  //GRID_LOAD(Ni_x);
  //GRID_LOAD(bmag);


  /////////////// READ OPTIONS //////////////////////////

  // Read some parameters
  Options *globalOptions = Options::getRoot();
  Options *options = globalOptions->getSection("trans_neu");
 
  OPTION(options, NOT_SOLVE_FOR_Vi, false);  // 
  
  OPTION(options, minimum_val,  1.e-10);    // minimum value limit for densities Ns  

  OPTION(options, NZ,  1);        // maximum grid number in Z

  OPTION(options, Te_x,  10.0);    // Read in eV 
  OPTION(options, Ti_x,  10.0);    // eV
  OPTION(options, Tm_x,  0.0258);    // eV
  OPTION(options, Ni_x,  1.0);     // in 1.e^19 m^-3
  OPTION(options, density_unit,   1.0e19); // Number density [m^-3]

  OPTION(options, Lp_crit,  5.e-5);    // in m

  OPTION(options, include_curvature, false);
  OPTION(options, noshear,           true);

  OPTION(options, profiles_lasttimestep, false);  // priority I
  OPTION(options, load_grid_profiles, false);     // priority II
  OPTION(options, load_experiment_profiles, false);     // priority III
  OPTION(options, initial_profile_exp, true);     // priority III
  OPTION(options, initial_profile_linear, false); // priority III
  OPTION(options, initial_profile_Hmode, false); // priority III
  OPTION(options, initial_PF_edgeval, false);    // set edge value in Private Flux region in X-gfile


  OPTION(options, Turb_Diff_on, false);     // include Turbulent diffusions
  OPTION(options, SMBI_LFS, true); 
  OPTION(options, Sheath_BC, false);
  OPTION(options, Sheath_width, 0);
  OPTION(options, SBC_particle_recycle, false);
  OPTION(options, Wall_particle_recycle, false);
  OPTION(options, Rate_recycle, 0.50);
  OPTION(options, Lni_wall, 0.05);     // in m

  OPTION(options, spitzer_resist,         false);

  OPTION(options, BScurrent,         false);
  OPTION(options, Aratio,            0.35);

  OPTION(options, nlfilter_noisy_data, false);
  OPTION(options, nlfilter_Gradpar_logNn, false);
  OPTION(options, nlfilter_Vnpol, false);
  OPTION(options, filter_para, 0.20);

  OPTION(options, term_GparkappaTe_GparTe, true); 
  OPTION(options, terms_recombination, false);  
  OPTION(options, terms_Gradpar_pn, true);  
  OPTION(options, terms_Gradpar_eta0n, true);  
  OPTION(options, terms_Diffcnn_par, false);  
  OPTION(options, terms_NnGradpar_Vn, true);  
  OPTION(options, terms_Gradperp_diffcoefs, true);  
  OPTION(options, term_driftExB, true);
  OPTION(options, term_driftcurvature, true);
  OPTION(options, term_drift_diamag, true);
  OPTION(options, term_parallel_current, false);
//ZYLHJ
    OPTION(options, reversed_mag_field, false);
    OPTION(options, phi3d, false);
    OPTION(options, split_n0, true);
    OPTION(options, boussinesq, true); 


 
  OPTION(options, alpha_vn, 0.);            // a.u. control para. for Vn at SBC
  OPTION(options, angle_B_plate, 3.14/6.);  // Read in rad, angle between B and plate

  OPTION(options, AA, 2.0);
  OPTION(options, ZZ, 1.0);
  Zi=ZZ;                                   // Zi is applied in BS_current from Tianyang

  OPTION(options, Mi, 1.0);        // Read in Mi
  OPTION(options, Mn, 1.0);        // Read in Mi
  OPTION(options, Mm, 2.0);        // Read in Mi
  OPTION(options, W_ionz, 20.0);    // Read in eV
  OPTION(options, W_diss, 4.5);     // in eV
  OPTION(options, W_rec, 4.5);      // in eV

  OPTION(options, W_bind, 0.5);     // in eV

  OPTION(options, dNidx_xin_au, -65.);    // a.u.
  OPTION(options, dTidx_xin_au, -3500.);  // a.u.
  OPTION(options, dTedx_xin_au, -3500.);  // a.u.
  OPTION(options, psi_xout_y0, 0.229543);  //m^2 T
  OPTION(options, Tn_plate, 1. );  //eV
  OPTION(options, Te_core, 1000. );  //eV
  OPTION(options, Te_edge, 10. );    //eV
  OPTION(options, Ti_core, 1000. );  //eV
  OPTION(options, Ti_edge, 10. );    //eV
  OPTION(options, Ni_core, 2. );     //in 1.e^19 m^-3
  OPTION(options, Ni_edge, 0.1 );    //in 1.e^19 m^-3
  OPTION(options, Initfile_x0, 0.4 );    //in a.u.
  OPTION(options, Initfile_w_ped, 0.2 );    //in a.u.
  OPTION(options, x0_ped, 0.90);            //in a.u. relative to normalized psi
  OPTION(options, width_ped, 0.062);        //in a.u.
  OPTION(options, coef_coregrad_ped, 0.01);//in a.u.
  OPTION(options, coef_corenlarge_ped, 18.);//in a.u.


  OPTION(options, q95_input,  5.);      // in a.u. a factor for heat flux limit
  OPTION(options, Lnn_min,  1.e-3);      // in m a factor for neutral density flux limit

  //OPTION(options, chi_perp,  0.6); // Read in m^2 / s 
  //OPTION(options, D_perp,    0.6);
  OPTION(options, Vm0,   -500.0);    // Read in m/s
  //Vm0 = 0.0;
  OPTION(options, Nm0,  1.e7);      // Read in in 1.e^19 m^-3

  OPTION(options, diffusion_coef_step_function,  false); 
  OPTION(options, diffusion_coef_step0,  0.1); // Read in m^2 / s 
  OPTION(options, diffusion_coef_step1,  1.0); // Read in m^2 / s 
  OPTION(options, aveYg11J_core,  0.005);  // Read in a.u.
  OPTION(options, diffusion_coef_Hmode,  false); 
  OPTION(options, diffusion_coef_Hmode0,  1.); // Read in m^2 / s 
  OPTION(options, diffusion_coef_Hmode1,  10.0); // Read in m^2 / s 
  OPTION(options, diffusion_PF_constant,  false); // Read in m^2 / s 


  OPTION(options, Diffc_ni_perp,  0.1); // Read in m^2 / s 
  OPTION(options, Difft_ni_perp,  1.0);
  OPTION(options, chic_i_perp,    0.4);
  OPTION(options, chit_i_perp,    4.0);
  OPTION(options, chic_e_perp,    0.6);
  OPTION(options, chit_e_perp,    6.0);
  OPTION(options, chic_n_perp, 0.4);
  

  //**** Boundaries of Const Flux of fueling *//
  OPTION(options, CF_BC_x0,    1.01);
  OPTION(options, CF_BC_y0,    0.47);
  OPTION(options, CF_BC_y1,    0.53);
  OPTION(options, CF_BC_z0,    0.0);
  OPTION(options, CF_BC_z1,    2.0);     //default smbi in whole Z
  OPTION(options, Sheath_BC_x0, 0.84);
  Diffc_nm_perp=0.01;
   
   // external sources
  OPTION(options, external_sources,  false); 
  OPTION(options, extsrcs_balance_diffusion,  false); 
  OPTION(options, x0_extS, 0.85);            //in a.u. relative to normalized psi
  OPTION(options, width_extS, 0.062);        //in a.u.
  OPTION(options, coef_coregrad_extS, 0.021);//in a.u.
  OPTION(options, amp_spi_ext, 1.e-5);        //in N0/s,  N0=1e^19/m^3
  OPTION(options, amp_see_ext, 1.e-4);        //in eV*N0/s
  OPTION(options, amp_sei_ext, 1.e-4);        //in eV*N0/s

  // Vacuum region control
  OPTION(options, vacuum_pressure,   0.02);   // Fraction of peak pressure
  OPTION(options, vacuum_trans,      0.005);  // Transition width in pressure
  // Resistivity and hyper-resistivity options
  OPTION(options, vac_lund,          0.0);    // Lundquist number in vacuum region
  OPTION(options, core_lund,         0.0);    // Lundquist number in core region


//////ZYLHJ
/*
 OPTION(options, bracket_method_exb, 0);
  switch(bracket_method_exb) {
  case 0: {
    bm_exb = BRACKET_STD;
    output << "\tBrackets for ExB: default differencing\n";
    break;
  }
  case 1: {
    bm_exb = BRACKET_SIMPLE;
    output << "\tBrackets for ExB: simplified operator\n";
    break;
  }
  case 2: {
    bm_exb = BRACKET_ARAKAWA;
    output << "\tBrackets for ExB: Arakawa scheme\n";
    break;
  }
  case 3: {
    bm_exb = BRACKET_CTU;
    output << "\tBrackets for ExB: Corner Transport Upwind method\n";
    break;
  }
  default:
    output << "ERROR: Invalid choice of bracket method. Must be 0 - 3\n";
    return 1;
  }

OPTION(options, bracket_method_mag, 2);
  switch(bracket_method_mag) {
  case 0: {
    bm_mag = BRACKET_STD;
    output << "\tBrackets: default differencing\n";
    break;
  }
  case 1: {
    bm_mag = BRACKET_SIMPLE;
    output << "\tBrackets: simplified operator\n";
    break;
  }
  case 2: {
    bm_mag = BRACKET_ARAKAWA;
    output << "\tBrackets: Arakawa scheme\n";
    break;
  }
  case 3: {
    bm_mag = BRACKET_CTU;
    output << "\tBrackets: Corner Transport Upwind method\n";
    break;
  }
  default:
    output << "ERROR: Invalid choice of bracket method. Must be 0 - 3\n";
    return 1;
  }
 */ 
  ////////////// CALCULATE PARAMETERS ///////////////////

 //in order to using formulas of wci and rhos in Gaussian cgs units, 
 // bmag in SI unit Tesla multiply by 1.e4 changes to Gaussian unit gauss 
  // Te_x in unit eV, no need to transfer

  // !!!Be very careful when using quantities calculated below, units transfer needed
  
  Ni_x *= 1.0e13;    // in unit cm^-3 now
  bmag *= 1.0e4;     // in unit gauss now

  output.write("Calculating parameters in Gaussian units and then transfer to SI units \n"); 
  output.write("\tIn Gaussian cgs units:  Ni_x %e cm^-3, Te_x %e eV, bmag %e gauss  \n",Ni_x,Te_x,bmag);
  
  output.write("\tparameters:  AA=mi/mp %e,  ZZ %e \n",AA,ZZ);

  rho_s = 1.02e2*sqrt(AA*Te_x)/ZZ/bmag;   // unit cm
  wci   = 9.58e3*ZZ*bmag/AA;
  Vi_x = wci * rho_s;                     // in unit cm/s                                     
///Solve the problem of 'Monitor signalled to quit. Returning'/// ZYLHJ

Vn_perp = 0.0;

  fmei  = 1./1836.2/AA; 
  mi_me = 1./fmei;                                         // no unit
  lambda_ei = 24.-log(sqrt(Ni_x)/Te_x);                          // in unit cm
  lambda_ii = 23.-log(ZZ*ZZ*ZZ*sqrt(2.*Ni_x)/pow(Ti_x, 1.5));    // unit cm
  nueix     = 2.91e-6*Ni_x*lambda_ei/pow(Te_x, 1.5);             // unit 1/s
  //nuiix     = 4.78e-8*pow(ZZ,4.)*Ni_x*lambda_ii/pow(Ti_x, 1.5)/sqrt(AA);   // unit 1/s
  nuiix     = 4.78e-8*Ni_x*lambda_ii/pow(Ti_x, 1.5)/sqrt(AA);         // unit 1/s

 //after using formulas of wci and rhos in Gaussian cgs units, 
 // bmag in Gaussian unit gauss divided by 1.e4 changes back to Gaussian SI unit Tesla

  Ni_x /= 1.0e13;                         // back to unit 1.e^19 m^-3 mow
  bmag /=1.0e4;                           // back to unit tesla now
  Vi_x /= 1.e2;                           // back to unit m/s

  tbar = Lbar/Vi_x;                       // in unit s

  N_cubic_Lbar = Ni_x*1.e19*Lbar*Lbar*Lbar;  // unit 1


  output.write("\tIn SI units:  Ni_x %e e^19 m^-3,  Te_x %e eV, bmag %e tesla \n",
	      Ni_x,Te_x,bmag);


Upara0= bmag*bmag*tbar*tbar/(MU0*Ni_x*density_unit*Mi*mass_unit*Lbar*Lbar);//this normalization factor is equal to 1, since Lbar/tbar=Bbar/sqr(μ0*Mi*Ni)

Upara1= KB*Te_x*eV_K*tbar/(ee*bmag*Lbar*Lbar);
Upara2= 1.0;


output.write("vorticity cinstant: Upara0 = %e     Upara1 = %e  Upara2 = %e \n", Upara0, Upara1, Upara2);

  ///////////// PRINT Z INFORMATION /////////////////////
  
  BoutReal hthe0;
  if(GRID_LOAD(hthe0) == 0) {
    output.write("    ****NOTE: input from BOUT, Z length needs to be divided by %e\n", hthe0/Lbar);
  }

 if(!include_curvature)
    b0xcv = 0.0;

  if(noshear) {
    if(include_curvature)
      b0xcv.z += I*b0xcv.x;
    mesh->ShiftXderivs = false;
    I = 0.0;
  }
  
  //////////////////////////////////////////////////////////////
  // SHIFTED RADIAL COORDINATES

  if(mesh->ShiftXderivs) {
    if(mesh->IncIntShear) {
      // BOUT-06 style, using d/dx = d/dpsi + I * d/dz
      mesh->IntShiftTorsion = I;
      
    }else {
      // Dimits style, using local coordinate system
      if(include_curvature)
	b0xcv.z += I*b0xcv.x;
      I = 0.0;  // I disappears from metric
    }
  }

  ///////////// NORMALISE QUANTITIES ////////////////////

  output.write("\tNormalising to Lbar = %e m, tbar %e s, V_Ti %e m/s \n", Lbar, tbar, Vi_x);

   // Normalise geometry 
  Rxy /= Lbar;
  hthe /= Lbar;
  mesh->dx /= Lbar*Lbar*bmag;
  //mesh->dx /= Lbar;
  I   *= Lbar*Lbar*bmag;
  psixy /= Lbar*Lbar*bmag;
  psi_axis /= Lbar*Lbar*bmag;
  psi_bndry /= Lbar*Lbar*bmag;
  psi_xout_y0 /= Lbar*Lbar*bmag;
  // Normalise magnetic field
  b0xcv.x /= bmag;
  b0xcv.y *= Lbar*Lbar;
  b0xcv.z *= Lbar*Lbar;
  
  Bpxy /= bmag;
  Btxy /= bmag;
  mesh->Bxy  /= bmag;
  B0 /= bmag;

  b0xcv *= 2. / mesh->Bxy;

///////Set B field vector

  B0vec.covariant = false;
  B0vec.x = 0.;
  B0vec.y = Bpxy / hthe;
  B0vec.z = 0.;

  // Normalise coefficients
  Lni_wall /= Lbar;
  Lp_crit /= Lbar;
  W_ionz /= Te_x;
  W_diss /= Te_x;
  W_bind /= Te_x;

  Tn_plate /= Te_x;
  Tm_x /= Te_x;
  Te_core /= Te_x;
  Te_edge /= Te_x;
  Ti_core /= Te_x;
  Ti_edge /= Te_x;
  Ni_core /= Ni_x;
  Ni_edge /= Ni_x;
    
  Te_grid /= Te_x;
  Ti_grid /= Te_x;
  Ni_grid =Ni_grid*100./Ni_x;  //hl2a grid in unit 10^19/m^3 

  Te_exp /= Te_x;
  Ti_exp /= Te_x;
  Ni_exp /= Ni_x;  
      

  Vm0 /= Lbar/tbar;

  Diffc_nm_perp /= Lbar*Lbar/tbar;
  Diffc_ni_perp /= Lbar*Lbar/tbar;
  Difft_ni_perp /= Lbar*Lbar/tbar;

  chic_i_perp /= Lbar*Lbar/tbar;
  chit_i_perp /= Lbar*Lbar/tbar;
  chic_e_perp /= Lbar*Lbar/tbar;
  chit_e_perp /= Lbar*Lbar/tbar;

  diffusion_coef_step0 /= Lbar*Lbar/tbar;
  diffusion_coef_step1 /= Lbar*Lbar/tbar;
  diffusion_coef_Hmode0 /= Lbar*Lbar/tbar;
  diffusion_coef_Hmode1 /= Lbar*Lbar/tbar;

  amp_spi_ext /= Ni_x/tbar;
  amp_see_ext /= Ni_x*Te_x/tbar;
  amp_sei_ext /= Ni_x*Te_x/tbar;

  output.write("\tDiffusion coefficients in unit Lbar^2/tbar : \n");
  output.write("\tDiffc_perp %e, Difft_perp %e \n", Diffc_ni_perp, Difft_ni_perp);
  output.write("\tchic_i_perp %e, chit_i_perp %e \n", chic_i_perp,chit_i_perp);
  output.write("\tchic_e_perp %e, chit_e_perp %e \n", chic_e_perp,chit_e_perp);
  output.write("\tdiff_coef_step0 %e, diff_coef_step1 %e \n", diffusion_coef_step0,diffusion_coef_step1);
  output.write("\tdiff_coef_Hmode0 %e, diff_coef_Hmode1 %e \n", diffusion_coef_Hmode0,diffusion_coef_Hmode1);


  if (phi3d) {
#ifdef PHISOLVER
    phiSolver3D = Laplace3D::create();
#endif
  } else {
    if (split_n0) {
      // Create an XY solver for n=0 component
      laplacexy = new LaplaceXY(mesh);
      // Set coefficients for Boussinesq solve
      laplacexy->setCoefs(1. / SQ(mesh->Bxy), 0.0);
      phi2D = 0.0; // Starting guess
    }

    // Create an XZ solver
    OPTION(options, newXZsolver, false);
    if (newXZsolver) {
      // Test new LaplaceXZ solver
      newSolver = LaplaceXZ::create(mesh);
      // Set coefficients for Boussinesq solve
     newSolver->setCoefs(1. / SQ(mesh->Bxy), 0.0);
    } else {
      // Use older Laplacian solver
       phiSolver = Laplacian::create(globalOptions->getSection("phiSolver"));
      // Set coefficients for Boussinesq solve
 
      phiSolver->setCoefC(1. / SQ(mesh->Bxy));
    }
    phi = 0.0;
  }




  /////////////// CALCULATE METRICS /////////////////

  mesh->g11 = (Rxy*Bpxy)^2;
  mesh->g22 = 1.0 / (hthe^2);
  mesh->g33 = (I^2)*mesh->g11 + (mesh->Bxy^2)/mesh->g11;
  mesh->g12 = 0.0;  
  mesh->g13 = -I*mesh->g11;
  mesh->g23 = -Btxy/(hthe*Bpxy*Rxy);
  
  mesh->J = hthe / Bpxy;
  
  mesh->g_11 = 1.0/mesh->g11 + ((I*Rxy)^2);
  mesh->g_22 = (mesh->Bxy*hthe/Bpxy)^2;
  mesh->g_33 = Rxy*Rxy;
  mesh->g_12 = Btxy*hthe*I*Rxy/Bpxy;
  mesh->g_13 = I*Rxy*Rxy;
  mesh->g_23 = Btxy*hthe*Rxy/Bpxy;
  
  mesh->geometry(); // Calculate other metrics
  
  // SET VARIABLE LOCATIONS *******************//

  Ni.setLocation(CELL_CENTRE);
  Ti.setLocation(CELL_CENTRE);
  Te.setLocation(CELL_CENTRE);
  Vi.setLocation(CELL_CENTRE);
  Nn.setLocation(CELL_CENTRE);
  Tn.setLocation(CELL_CENTRE);
  Vn.setLocation(CELL_CENTRE);
  Nm.setLocation(CELL_CENTRE);
  Vmx.setLocation(CELL_CENTRE); 

////ZYLHJ
    Vort.setLocation(CELL_CENTRE);
    phi.setLocation(CELL_CENTRE);
    Jpar.setLocation(CELL_CENTRE);
    V.setLocation(CELL_CENTRE);
    psi.setLocation(CELL_CENTRE);
  //////////////// BOUNDARIES ///////////////////////
  // Set BOUNDARiES first here, and then apply them every time in physics run/////
  //
 
   Ni.setBoundary("Ni");
   Ti.setBoundary("Ti");
   Te.setBoundary("Te");
   Vi.setBoundary("Vi");
   Nn.setBoundary("Nn");
   Tn.setBoundary("Tn");
   Vn.setBoundary("Vn");
   Nm.setBoundary("Nm");
   //Vmx.setBoundary("Vmx");
   Vm.setBoundary("Vm");
   Vort.setBoundary("U");
   phi.setBoundary("phi");  ///ZYLHJ
   Jpar.setBoundary("J");
   V.setBoundary("phi");
   psi.setBoundary("phi");
   //Set Boundary for other output variables
   tau_ei.setBoundary("Tn");
   nu_ionz.setBoundary("Tn");
   nu_CX.setBoundary("Tn");
   nu_ionz_n.setBoundary("Tn");
   nu_CX_n.setBoundary("Tn");
   Si_p.setBoundary("Tn");
   Si_CX.setBoundary("Tn");
   S_diss.setBoundary("Tn");
   Vn_perp.setBoundary("Tn");

   grad_perp_Diffi.setBoundary("Tn");
   grad_perp_Diffn.setBoundary("Tn");
   grad_perp_chii.setBoundary("Tn");
   grad_perp_chie.setBoundary("Tn");

   bracketphiTe.setLocation(CELL_CENTRE);
   bracketphiTe.setBoundary("Te");
   bracketphiTi.setLocation(CELL_CENTRE);
   bracketphiTi.setBoundary("Te");
   bracketphipei.setLocation(CELL_CENTRE);
   bracketphipei.setBoundary("Te");
   bracketphiNi.setLocation(CELL_CENTRE);
   bracketphiNi.setBoundary("phi");  
   bracketphiVi.setLocation(CELL_CENTRE);
   bracketphiVi.setBoundary("phi");
   bracketPiVi.setLocation(CELL_CENTRE);
   bracketPiVi.setBoundary("Vi"); 
   Ni_curvature1.setLocation(CELL_CENTRE);
   Ni_curvature1.setBoundary("Ni");
   Ni_curvature2.setLocation(CELL_CENTRE);
   Ni_curvature2.setBoundary("Ni");
   Te_curvature1.setLocation(CELL_CENTRE);
   Te_curvature1.setBoundary("Te");
   Te_curvature2.setLocation(CELL_CENTRE);
   Te_curvature2.setBoundary("Te");
   Te_curvature3.setLocation(CELL_CENTRE);
   Te_curvature3.setBoundary("Te");
   Te_current1.setLocation(CELL_CENTRE);
   Te_current1.setBoundary("Te");
   Ti_current1.setLocation(CELL_CENTRE);
   Ti_current1.setBoundary("Te");
   Ti_curvature1.setLocation(CELL_CENTRE);
   Ti_curvature1.setBoundary("Te");
   Ti_curvature2.setLocation(CELL_CENTRE);
   Ti_curvature2.setBoundary("Te");
   Ti_curvature3.setLocation(CELL_CENTRE);
   Ti_curvature3.setBoundary("Te");



   Grad_par_pei.setBoundary("Tn");
   Grad_par_pn.setBoundary("Tn");
   Grad_par_logNn.setBoundary("Tn");
   Grad_par_Tn.setBoundary("Tn");
   DDX_Ni.setBoundary("Tn");
   DDX_Ti.setBoundary("Tn");
   DDX_Te.setBoundary("Tn");
   D2DX2_Ni.setBoundary("Tn");
   D2DX2_Ti.setBoundary("Tn");
   D2DX2_Te.setBoundary("Tn");

   Flux_Ni_aveY.setBoundary("Tn");
   Flux_Ti_aveY.setBoundary("Tn");
   Flux_Te_aveY.setBoundary("Tn");

////ZYLHJ
GradPhi2.setLocation(CELL_CENTRE);
GradPhi2.setBoundary("phi");
bracketPhiP.setLocation(CELL_CENTRE);
bracketPhiP.setBoundary("pei");
Dperp2Phi.setLocation(CELL_CENTRE);
Dperp2Phi.setBoundary("phi");

//GradPhi1.setLocation(CELL_CENTRE);
//GradPhi1.setBoundary("phi");

Dperp2Pi.setLocation(CELL_CENTRE);
Dperp2Pi.setBoundary("pei");

Dperp2psi.setLocation(CELL_CENTRE);
Dperp2psi.setBoundary("phi");

  ///////////// SET EVOLVING VARIABLES //////////////
  //
  // Tell BOUT++ which variables to evolve
  // add evolving variables to the communication object



  Ni= Te = Ti = 0.0;
  Vi=0.;
  Nn = Tn= Vn=0.0;       //NB: it is **NECESSARY** to set values for evolving quantities if not read from grid 
  Nm = Vmx = 0.0;
  Ni0_Hmode = Ti0_Hmode = Te0_Hmode=0.0;
  Vort = 0.;
  phi = 0.0;
  Jpar = 0.0;
  V=0.0;
  psi = 0.0;
  SOLVE_FOR4(Ni, Vi, Te, Ti);
  SOLVE_FOR3(Nn,Tn,Vn);
  //SOLVE_FOR2(Nm,Vmx);
  SOLVE_FOR4(Nm,Vm,Vort,psi);

  if(profiles_lasttimestep)
    {
     output.write("Initial Profiles are loaded from .txt files of all evolving quatities at last time step \n");

     BoutReal lstimeNi[NX][NY][NZ],lstimeTi[NX][NY][NZ],lstimeTe[NX][NY][NZ],lstimeVi[NX][NY][NZ];
     BoutReal lstimeNn[NX][NY][NZ],lstimeNm[NX][NY][NZ],lstimeVmx[NX][NY][NZ],lstimeVort[NX][NY][NZ],lstimepsi[NX][NY][NZ];  // Vn Tn are not evloved in recent version

	ifstream pFile1,pFile2,pFile3,pFile4,pFile5,pFile6,pFile7,pFile8,pFile9;
	pFile1.open ("data/lstime_ni.txt", ios::in );
	pFile2.open ("data/lstime_ti.txt", ios::in );
	pFile3.open ("data/lstime_te.txt", ios::in );
	pFile4.open ("data/lstime_vi.txt", ios::in );
	pFile5.open ("data/lstime_nn.txt", ios::in );
	pFile6.open ("data/lstime_nm.txt", ios::in );
	pFile7.open ("data/lstime_vmx.txt", ios::in );
        pFile8.open ("data/lstime_Vort.txt", ios::in );
        pFile9.open ("data/lstime_psi.txt", ios::in );

	output.write("\tCheck point 4 : files are open\n" ); 
	for(int jx=0;jx<NX;jx++) 
	 for(int jy=0;jy<NY;jy++) 
	   for(int jz=0;jz<NZ;jz++) 
             {
	       pFile1 >> lstimeNi[jx][jy][jz]; 
	       pFile2 >> lstimeTi[jx][jy][jz]; 
	       pFile3 >> lstimeTe[jx][jy][jz]; 
	       pFile4 >> lstimeVi[jx][jy][jz]; 
	       pFile5 >> lstimeNn[jx][jy][jz]; 
	       pFile6 >> lstimeNm[jx][jy][jz]; 
	       pFile7 >> lstimeVmx[jx][jy][jz]; 
               pFile8 >> lstimeVort[jx][jy][jz];
               pFile9 >> lstimepsi[jx][jy][jz];
             }
	pFile1.close();
	pFile2.close();
	pFile3.close();
	pFile4.close();
	pFile5.close();
	pFile6.close();
	pFile7.close();
        pFile8.close();
        pFile9.close();

	for(int jx=0;jx<mesh->ngx;jx++) 
	for(int jy=0;jy<mesh->ngy;jy++) 
	for(int jz=0;jz<mesh->ngz;jz++) 
        { 
	  Ni[jx][jy][jz] = lstimeNi[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
	  Ti[jx][jy][jz] = lstimeTi[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
	  Te[jx][jy][jz] = lstimeTe[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
	  Vi[jx][jy][jz] = lstimeVi[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
	  Nn[jx][jy][jz] = lstimeNn[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
	  Nm[jx][jy][jz] = lstimeNm[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
	  Vmx[jx][jy][jz] = lstimeVmx[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
          Vort[jx][jy][jz] = lstimeVort[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
psi[jx][jy][jz] = lstimepsi[mesh->XGLOBAL (jx)][mesh->YGLOBAL (jy)][jz];
	}

    Vm.x=Vmx;
    Vm.y=0.;
    Vm.z=0.;

   if(initial_profile_Hmode) 
     {
      output.write("Restart from one chosen time step, Hmode profiles at early beginning also given\n");
      for (int jx=0;jx<mesh->ngx;jx++)
       {
         for (int jy=0;jy<mesh->ngy;jy++)
	  {
            BoutReal psi_normal = (psixy[jx][jy] - psi_axis)/(psi_bndry-psi_axis);  
            BoutReal x_prime = (x0_ped- psi_normal)/width_ped;
            BoutReal Hmode_tanh=(exp(-x_prime)+(1.+ coef_coregrad_ped*x_prime)*exp(x_prime)*coef_corenlarge_ped)/(exp(-x_prime)+exp(x_prime));
 	   for (int jz=0;jz<mesh->ngz;jz++)
            {            
             Ni0_Hmode[jx][jy][jz] = Ni_edge+Ni_core*(Hmode_tanh - 1.);
             Ti0_Hmode[jx][jy][jz] = Ti_edge+Ti_core*(Hmode_tanh - 1.);
             Te0_Hmode[jx][jy][jz] = Te_edge+Te_core*(Hmode_tanh - 1.);

	    }
	  }
        }
      }//end initial_Hmode
    }
  else{

       if (load_grid_profiles)
        {
         output.write("\tInitial profiles of Ti Te Ni are loaded from grid file\n" ); 
         Te = Te_grid;
         Ti = Ti_grid;
         Ni = Ni_grid;
         Vn=0.0;
	 Vi=0.0;                                             
         Tn = minimum_val;
         Nn= minimum_val;
         Nm= minimum_val;
	 Vmx=0.;
         Vort = 0.0;
         phi = 0.0;
         psi = 0.0;
         V=0.0;
//         Jpar = 0.0;
        }  
  
      if (load_experiment_profiles)
       {
         output.write("\tInitial experiment profiles of Ti Te Ni are loaded from grid file\n" ); 
         Te = Te_exp;
         Ti = Ti_exp;
         Ni = Ni_exp;
         Vn=0.0;
	 Vi=0.0;                                             
         Tn = minimum_val;
         Nn= minimum_val;
         Nm= minimum_val;
	 Vmx=0.;
         Vort = 0.0;
         phi = 0.0;
         psi = 0.0;
         V=0.0;
         Ni0_Hmode = Ni;
         Ti0_Hmode = Ti;
         Te0_Hmode = Te;

        }  

  // Initialization of Profiles 
  // ****** NB: profiles should be initialized after "SOLVE_FOR()" ****//
  S_pi_ext = 0.;
  S_Ei_ext = 0.;
  S_Ee_ext = 0.;

  for (int jx=0;jx<mesh->ngx;jx++)
     {
      x_rela= mesh->GlobalX(jx); 

      //BoutReal x0=0.3,w_ped=0.1;
      BoutReal temp=exp(2.*(x_rela-Initfile_x0)/Initfile_w_ped);
      BoutReal x0_nn=1.02,w_nn=0.05;
      BoutReal temp2=exp(-(x_rela-x0_nn)*(x_rela-x0_nn)/w_nn/w_nn);

      for (int jy=0;jy<mesh->ngy;jy++)
	{
        BoutReal x_psi_l = psixy[jx][jy]-psi_xout_y0;
        BoutReal psi_normal = (psixy[jx][jy] - psi_axis)/(psi_bndry-psi_axis);  
        BoutReal x_prime = (x0_ped- psi_normal)/width_ped;
        BoutReal Hmode_tanh=(exp(-x_prime)+(1.+ coef_coregrad_ped*x_prime)*exp(x_prime)*coef_corenlarge_ped)/(exp(-x_prime)+exp(x_prime));

        BoutReal y_rela=mesh->GlobalY(jy);
        int jy_global = mesh->YGLOBAL(jy);
        BoutReal y0_nn=0.5,wy_nn=0.05;
        BoutReal temp2_y=exp(-(y_rela-y0_nn)*(y_rela-y0_nn)/wy_nn/wy_nn);
 	for (int jz=0;jz<mesh->ngz;jz++)
          {
	    if(!load_grid_profiles)
	      {

                if(initial_profile_exp)
	          {
	            Ni[jx][jy][jz]=Ni_edge+Ni_core/(1.+temp);
                    Ti[jx][jy][jz]=Te_edge+Te_core/(1.+temp);
                    Te[jx][jy][jz]=Ti_edge+Ti_core/(1.+temp);
	           }

                if(initial_profile_linear)
                  {
                   Ni[jx][jy][jz] = Ni_edge+dNidx_xin_au*x_psi_l;
                   Ti[jx][jy][jz] = Ti_edge+dTidx_xin_au*x_psi_l;
                   Te[jx][jy][jz] = Te_edge+dTedx_xin_au*x_psi_l;
                  }

                if(initial_profile_Hmode) 
                  {
                   Ni[jx][jy][jz] = Ni_edge+Ni_core*(Hmode_tanh - 1.);
                   Ti[jx][jy][jz] = Ti_edge+Ti_core*(Hmode_tanh - 1.);
                   Te[jx][jy][jz] = Te_edge+Te_core*(Hmode_tanh - 1.);
                   Ni0_Hmode[jx][jy][jz]=Ni[jx][jy][jz];
                   Ti0_Hmode[jx][jy][jz]=Ti[jx][jy][jz];
                   Te0_Hmode[jx][jy][jz]=Te[jx][jy][jz];
                  }

		if(initial_PF_edgeval)
                  {
		    if(jy_global<=jyseps11 || jy_global>=NY-1-jyseps11)
		      {
			Ni[jx][jy][jz] = Ni_edge;
                        Ti[jx][jy][jz] = Ti_edge;
                        Te[jx][jy][jz] = Te_edge;
                       }
                   }

	      }

             Vn[jx][jy][jz]=0.0;
	     Vi[jx][jy][jz]=0.0;                                               //~~~~~~~~~~~~~~~~~~
                                                                               // INITIALIZE
	                                                                       //__________________
             Tn[jx][jy][jz] = minimum_val;

             Nn[jx][jy][jz] = minimum_val;
             Nm[jx][jy][jz] = minimum_val;


	     Vmx[jx][jy][jz]=0.;

	     if(external_sources)
               {
                S_pi_ext[jx][jy][jz] = amp_spi_ext*(Hmode_tanh - 1.);
                S_Ei_ext[jx][jy][jz] = amp_sei_ext*(Hmode_tanh - 1.);
                S_Ee_ext[jx][jy][jz] = amp_see_ext*(Hmode_tanh - 1.);
               }

          }
        }
     }

  } // end of else if (profiles_lasttimestep)

  //End of Initialization of Profiles


  // Set step functions of diffusion coefficients
  // //NB: it is **NECESSARY** to set initial Zero values 
  Diffc_ni_step=0.;
  chic_i_step=0.;
  chic_e_step=0.; 
  Diffc_ni_Hmode=0.;
  chic_i_Hmode=0.;
  chic_e_Hmode=0.; 
  
  // calculate radial derivative for Hmode diffusion coefficients
  //if (diffusion_coef_Hmode )
    {
     output.write("Calculated diffusion coefficients for H-mode profiles");
     DDX_Ni = DDX(Ni0_Hmode);
     mesh->communicate(DDX_Ni);
     DDX_Ni.applyBoundary();

     DDX_Ti = DDX(Ti0_Hmode);
     mesh->communicate(DDX_Ti);
     DDX_Ti.applyBoundary();

     DDX_Te = DDX(Te0_Hmode);
     mesh->communicate(DDX_Te);
     DDX_Te.applyBoundary();

     aveY_J=mesh->averageY(mesh->J);
     aveY_g11=mesh->averageY(mesh->g11);
     aveY_g11J=mesh->averageY((mesh->g11*mesh->J));    //NB: mesh->averageY((Field3D var).DC())

    }
  
   BoutReal Gamma_ni_xin = -diffusion_coef_Hmode0*aveYg11J_core*dNidx_xin_au;
   BoutReal Qi_xin = -diffusion_coef_Hmode0*aveYg11J_core*dTidx_xin_au;
   BoutReal Qe_xin = -diffusion_coef_Hmode0*aveYg11J_core*dTedx_xin_au;

  for (int jx=0;jx<mesh->ngx;jx++)
     {
      for (int jy=0;jy<mesh->ngy;jy++)
	{
         int jy_global = mesh->YGLOBAL(jy);
         BoutReal psi_normal = (psixy[jx][jy] - psi_axis)/(psi_bndry-psi_axis);
         for (int jz=0;jz<mesh->ngz;jz++)
          {
            if(psi_normal>1.0) 
              {
               Diffc_ni_step[jx][jy][jz]=diffusion_coef_step1;
               chic_i_step[jx][jy][jz]=diffusion_coef_step1;
               chic_e_step[jx][jy][jz]=diffusion_coef_step1;
              }
            else
              {
               Diffc_ni_step[jx][jy][jz]=diffusion_coef_step0;
               chic_i_step[jx][jy][jz]=diffusion_coef_step0;
               chic_e_step[jx][jy][jz]=diffusion_coef_step0;
              }

	    if(diffusion_coef_Hmode)
	      {
		// BoutReal Gamma_ni_xin = -diffusion_coef_Hmode0*sqrt(mesh->g11[jx][jy])*dNidx_xin_au;
		//BoutReal Qi_xin = -diffusion_coef_Hmode0*sqrt(mesh->g11[jx][jy])*dTidx_xin_au;
		//BoutReal Qe_xin = -diffusion_coef_Hmode0*sqrt(mesh->g11[jx][jy])*dTedx_xin_au;

               
		// BoutReal Gamma_ni_xin = -diffusion_coef_Hmode0*aveY_g11J[jx][jy]*dNidx_xin_au;
		//BoutReal Qi_xin = -diffusion_coef_Hmode0*aveY_g11J[jx][jy]*dTidx_xin_au;
		//BoutReal Qe_xin = -diffusion_coef_Hmode0*aveY_g11J[jx][jy]*dTedx_xin_au;

	       if(DDX_Ni[jx][jy][jz]>-minimum_val) DDX_Ni[jx][jy][jz]=-minimum_val;
	       if(DDX_Ti[jx][jy][jz]>-minimum_val) DDX_Ti[jx][jy][jz]=-minimum_val;
	       if(DDX_Te[jx][jy][jz]>-minimum_val) DDX_Te[jx][jy][jz]=-minimum_val;

	       // Diffc_ni_Hmode[jx][jy][jz] = -Gamma_ni_xin/sqrt(mesh->g11[jx][jy])/DDX_Ni[jx][jy][jz];
               //chic_i_Hmode[jx][jy][jz] = -Qi_xin/sqrt(mesh->g11[jx][jy])/DDX_Ti[jx][jy][jz];
               //chic_e_Hmode[jx][jy][jz] = -Qe_xin/sqrt(mesh->g11[jx][jy])/DDX_Te[jx][jy][jz];

               Diffc_ni_Hmode[jx][jy][jz] = -Gamma_ni_xin/aveY_g11J[jx][jy]/DDX_Ni[jx][jy][jz];
               chic_i_Hmode[jx][jy][jz] = -Qi_xin/aveY_g11J[jx][jy]/DDX_Ti[jx][jy][jz];
               chic_e_Hmode[jx][jy][jz] = -Qe_xin/aveY_g11J[jx][jy]/DDX_Te[jx][jy][jz];


               if(Diffc_ni_Hmode[jx][jy][jz]>diffusion_coef_Hmode1) Diffc_ni_Hmode[jx][jy][jz]=diffusion_coef_Hmode1;
               if(chic_i_Hmode[jx][jy][jz] > diffusion_coef_Hmode1) chic_i_Hmode[jx][jy][jz]=diffusion_coef_Hmode1;
               if(chic_e_Hmode[jx][jy][jz] > diffusion_coef_Hmode1) chic_e_Hmode[jx][jy][jz]=diffusion_coef_Hmode1;

		if(diffusion_PF_constant)
                  {
		    if(jy_global<=jyseps11 || jy_global>=NY-1-jyseps11)
		      {
			Diffc_ni_Hmode[jx][jy][jz]=diffusion_coef_Hmode0;
                        chic_i_Hmode[jx][jy][jz]=diffusion_coef_Hmode0;
                        chic_e_Hmode[jx][jy][jz]=diffusion_coef_Hmode0;
                       }
                   }

              }

          }
        }
      }
  	  
  if(extsrcs_balance_diffusion)
    {
      D2DX2_Ni = D2DX2(Ni);
      mesh->communicate(D2DX2_Ni);
      D2DX2_Ni.applyBoundary();

      D2DX2_Ti = D2DX2(Ti);
      mesh->communicate(D2DX2_Ti);
      D2DX2_Ti.applyBoundary();

      D2DX2_Te = D2DX2(Te);
      mesh->communicate(D2DX2_Te);
      D2DX2_Te.applyBoundary();

      //S_pi_ext= - D2DX2_Ni*Diffc_ni_Hmode*mesh->g11;  
      //S_Ei_ext= - D2DX2_Ti*chic_i_Hmode*Ni*mesh->g11;  
      //S_Ee_ext= - D2DX2_Te*chic_e_Hmode*Ni*mesh->g11;  
      
      tau_ei = 1./tbar/(nueix*Ni/(Te^1.5));
      S_pi_ext= 0.;
      S_Ei_ext= - 2.*fmei*(Te-Ti)*Ni/tau_ei/0.6667;
      S_Ee_ext= - S_Ei_ext;

    }

  ///////////// ADD OUTPUT VARIABLES ////////////////
  //
  // Add any other variables to be dumped to file
  
  SAVE_ONCE5(Te_x, Ti_x, Ni_x, Lbar, tbar); // Normalisation factors
  SAVE_ONCE2(bmag,Vi_x);

  // Set flux limit for kappa
  V_th_e= 4.19e5*sqrt(Te*Te_x);
  V_th_i= 9.79e3*sqrt(Ti*Te_x/AA);
  output.write("\tion thermal velocity: %e -> %e [m/s]\n", min(V_th_i), max(V_th_i));
  output.write("\telectron thermal velocity: %e -> %e [m/s]\n", min(V_th_e), max(V_th_e));
  V_th_e /= Lbar/tbar;
  V_th_i /= Lbar/tbar;
  output.write("\tNormalized ion thermal velocity: %e -> %e [Lbar/tbar]\n", min(V_th_i), max(V_th_i));
  output.write("\tNormalized electron thermal velocity: %e -> %e [Lbar/tbar]\n", min(V_th_e), max(V_th_e));

  kappa_Te = 3.2*(1./fmei)*(1./tbar/nueix)*(Te^2.5);     // power operator '^' works only for Fields
  kappa_Ti = 3.9*(1./tbar/nuiix)*(Ti^2.5);  
  Field3D kappa_Te_realunits, kappa_Ti_realunits;
  kappa_Te_realunits= kappa_Te*Ni_x*pow(Lbar,2.)/tbar;    // pow function works for BoutReal variables     
  kappa_Ti_realunits= kappa_Ti*Ni_x*pow(Lbar,2.)/tbar;

  output.write("\tion para thermal conductivity: %e -> %e [N0 m^2/s]\n", min(kappa_Ti_realunits), max(kappa_Ti_realunits));
  output.write("\telectron para thermal conductivity: %e -> %e [N0 m^2/s]\n", min(kappa_Te_realunits), max(kappa_Te_realunits));   
  output.write("\tNormalzied ion para thermal conductivity: %e -> %e [N0 Lbar^2/tbar]\n", min(kappa_Ti), max(kappa_Ti));
  output.write("\tNormalized electron para thermal conductivity: %e -> %e [N0 Lbar^2/tbar]\n", min(kappa_Te), max(kappa_Te));  

  kappa_Te_fl = V_th_e*q95_input*Ni;    // Ne=Ni quasineutral
  kappa_Ti_fl = V_th_i*q95_input*Ni;

  kappa_Te *= kappa_Te_fl/(kappa_Te_fl+kappa_Te);
  kappa_Ti *= kappa_Ti_fl/(kappa_Ti_fl+kappa_Ti);

  output.write("\tUsed ion para thermal conductivity: %e -> %e [N0 Lbar^2/tbar]\n", min(kappa_Ti), max(kappa_Ti));
  output.write("\tUsed electron para thermal conductivity: %e -> %e [N0 Lbar^2/tbar]\n", min(kappa_Te), max(kappa_Te));  

 // Ionization rate  depending on Nn for Plasmas 
  nu_ionz = 3.e3*Nn*Ni_x*Te*Te*Te_x*Te_x/(3.+0.01*Te*Te*Te_x*Te_x);   //Ni_x in unit 1e^19 
  nu_CX =  1.e5*Nn*Ni_x*(1.7+0.667*(((1.5*Ti*Te_x)^0.333)-2.466));      // empirical formula 

 //Dissociation rate of molecules
  nu_diss = 3.e3*Nm*Ni_x*Te*Te*Te_x*Te_x/(3.+0.01*Te*Te*Te_x*Te_x);     //need be corrected

  // recombination rate of ions and electrons
  Field3D lambda_rec=1.5789e5/(abs(Te)*Te_x*1.1604e4);  //Te trasfer unit from eV to Kelvin  
  nu_rec = Ni*Ni_x*5.197e-1*ZZ*sqrt(lambda_rec)*(0.4288+0.5*log(lambda_rec)+0.469/(lambda_rec)^0.333); //Seaton M F 1959 'Radiative recombination of hydrogenic ions'

  output.write("\tionization rate: %e -> %e [1/s]\n", min(nu_ionz), max(nu_ionz));
  output.write("\tcharge exchange rate: %e -> %e [1/s]\n", min(nu_CX), max(nu_CX)); 
  output.write("\tdissociation rate: %e -> %e [1/s]\n", min(nu_diss), max(nu_diss));
  output.write("\trecombination rate: %e -> %e [1/s]\n", min(nu_rec), max(nu_rec));   

  if(spitzer_resist) {
    // Use Spitzer resistivity 
    output.write("");
    output.write("\tSpizter parameters");
    eta_spitzer = 1.03e-4*Zi*lambda_ei*((Te*Te_x)^(-1.5)); // eta in Ohm-m. 
    output.write("\tSpitzer resistivity: %e -> %e [Ohm m]\n", min(eta_spitzer), max(eta_spitzer));
    eta_spitzer /= MU0 * Lbar * Lbar/tbar;
    output.write("\t -> Lundquist %e -> %e\n", 1.0/max(eta_spitzer), 1.0/min(eta_spitzer));
    dump.add(eta_spitzer, "eta_spitzer", 0);
  }

    SAVE_REPEAT(phi);
    SAVE_REPEAT(Jpar);
V = -Grad_perp(phi);
 V.applyBoundary();
    mesh->communicate(V);
dump.add(V, "V", 1);


///Pararmeters for current
  pei= Ni*(Te+Ti);
  BoutReal pnorm = max(pei, true); // Maximum over all processors
  vacuum_pressure *= pnorm; // Get pressure from fraction
  vacuum_trans *= pnorm;
// Transitions from 0 in core to 1 in vacuum
  vac_mask = (1.0 - tanh( (pei - vacuum_pressure) / vacuum_trans )) / 2.0;
  vac_resist = 1. / vac_lund;
  core_resist = 1. / core_lund;
  eta = core_resist + (vac_resist - core_resist) * vac_mask;

  //Bootstrap current calculated by using Sauter's formula 
 if (BScurrent)
    {
      q95=q95_input;
      Jpar_BS0.setLocation(CELL_YLOW);
      Jpar_BS0.setBoundary("Tn");
//      pei= Ni*(Te+Ti);
      Pe = Ni*Te;
      Pi = Ni*Ti;

      nu_estar = 100.*nueix * q95*tbar / (V_th_e) / Aratio^(1.5);
      nu_istar = 100.*nuiix * q95*tbar / (V_th_i) / Aratio^(1.5);
      //nu_estar = 0.012 * N0*Nbar*density/1.e20*Zi*Zi*q95*Lbar/(Te0*Tebar/1000. * Aratio^1.5);
      //nu_istar = 0.012 * N0*Nbar*density/1.e20*Zi*q95*Lbar/(Ti0*Tibar/1000. * Aratio^1.5);

      output.write("Bootstrap current is included: \n");
      output.write("Normalized electron collisionality: nu_e* = %e\n", max(nu_estar));
      output.write("Normalized ion collisionality: nu_i* = %e\n", max(nu_istar));
      ft = BS_ft(100);
      output.write("modified collisional trapped particle fraction: ft = %e\n", max(ft));
      f31 = ft / (1.+(1.-0.1*ft)*sqrt(nu_estar) + 0.5*(1.-ft)*nu_estar/Zi);
      f32ee = ft / (1.+0.26*(1.-ft)*sqrt(nu_estar) + 0.18*(1.-0.37*ft)*nu_estar/sqrt(Zi));
      f32ei = ft / (1.+(1.+0.6*ft)*sqrt(nu_estar) + 0.85*(1.-0.37*ft)*nu_estar*(1.+Zi));
      f34 = ft / (1.+(1.-0.1*ft)*sqrt(nu_estar) + 0.5*(1.-0.5*ft)*nu_estar/Zi);

      L31 = F31(f31) ;
      L32 = F32ee(f32ee)+F32ei(f32ei) ;
      L34 = F31(f34) ;

      BSal0 = - (1.17*(1.-ft))/(1.-0.22*ft-0.19*ft*ft);
      BSal = (BSal0+0.25*(1-ft*ft)*sqrt(nu_istar))/(1.+0.5*sqrt(nu_istar)) + 0.31*nu_istar*nu_istar*ft*ft*ft*ft*ft*ft;
      BSal *= 1./(1.+0.15*nu_istar*nu_istar*ft*ft*ft*ft*ft*ft);

      Jpar_BS0 = L31* DDX(pei)/Pe  + L32*DDX(Te)/Te + L34*DDX(Ti)/(Zi*Te)*BSal;
      Jpar_BS0 *= Field3D( -Rxy*Btxy*Pe*(MU0*KB*Ni_x*density_unit*Te_x*eV_K)/(mesh->Bxy*mesh->Bxy)/(bmag*bmag) );  //NB:   J_hat = MU0*Lbar * J / mesh->Bxy;

      mesh->communicate(Jpar_BS0);
      Jpar_BS0.applyBoundary();

      dump.add(Jpar_BS0, "jpar_BS0", 1);
    }

  dump.add(psixy,"psixy",0);

  // if (diffusion_coef_Hmode)
    {
     dump.add(aveY_J,"aveY_J",0); 
     dump.add(aveY_g11,"aveY_g11",0); 
     dump.add(aveY_g11J,"aveY_g11J",0); 

     dump.add(DDX_Ni,"DDX_Ni",0); 
     dump.add(DDX_Ti,"DDX_Ti",0); 
     dump.add(DDX_Te,"DDX_Te",0); 
    }
  dump.add(Diffc_ni_step,"Diffc_ni_step",0); // output only at initial
  dump.add(chic_i_step,"chic_i_step",0);
  dump.add(chic_e_step,"chic_e_step",0);
  dump.add(Diffc_ni_Hmode,"Diffc_ni_Hmode",0); 
  dump.add(chic_i_Hmode,"chic_i_Hmode",0);
  dump.add(chic_e_Hmode,"chic_e_Hmode",0);
  
  if(extsrcs_balance_diffusion)
    {
     dump.add(S_pi_ext,"S_pi_ext",0);
     dump.add(S_Ei_ext,"S_Ei_ext",0);
     dump.add(S_Ee_ext,"S_Ee_ext",0);
    }
  dump.add(Diff_ni_perp,"Diff_ni_perp",1); 
  dump.add(chi_i_perp,"chi_i_perp",1);
  dump.add(chi_e_perp,"chi_e_perp",1);

  dump.add(kappa_Te,"kappa_Te",1);           // output at any output step
  dump.add(kappa_Ti,"kappa_Ti",1);
  dump.add(Diffc_nn_perp,"Diffc_nn_perp",1);

  dump.add(nu_ionz,"nu_ionz",1);
  dump.add(nu_CX,"nu_CX",1);
  dump.add(nu_ionz_n,"nu_ionz_n",1);
  dump.add(nu_CX_n,"nu_CX_n",1);
  dump.add(nu_diss,"nu_diss",1);
  dump.add(nu_rec,"nu_rec",1);

  dump.add(Si_p,"Si_p",1);
  dump.add(Si_CX,"Si_CX",1);
  dump.add(S_diss,"S_diss",1);
  dump.add(S_rec,"S_rec",1);
  dump.add(Grad_par_pei,"Grad_par_pei",1);
  dump.add(Grad_par_pn,"Grad_par_pn",1);
  dump.add(Grad_par_logNn,"Grad_par_logNn",1);

  dump.add(tau_ei,"tau_ei",1);

  dump.add(Vn_perp,"Vn_perp",1);
  dump.add(q_se_kappa,"q_se_kappa",1);
  dump.add(q_si_kappa,"q_si_kappa",1);
  dump.add(Gamma_ni_Diffc,"Gamma_ni_Diffc",1);
  dump.add(Gamma_nn_Diffc,"Gamma_nn_Diffc",1);

  dump.add(Flux_Ni_aveY,"Flux_Ni_aveY",1);
  dump.add(Flux_Ti_aveY,"Flux_Ti_aveY",1);
  dump.add(Flux_Te_aveY,"Flux_Te_aveY",1);

dump.add(grad_perp_Diffi,"grad_perp_Diffi",1);
dump.add(bracketphiNi,"bracketphiNi",1);
dump.add(bracketphiVi,"bracketphiVi",1);
dump.add(bracketPiVi,"bracketPiVi",1);
  return(0);
}


int physics_run(BoutReal t)
{
  // Communicate variables
  mesh->communicate(Ni, Vi, Te, Ti);
  //mesh->communicate(Nn,Tn,Vn);
  mesh->communicate(Nn);
  mesh->communicate(Nm,Vmx,Vm);
  // NB: Intermediate variables calculated with Grad operators are all necessary to be communicated
  // after being calculated
  
  Ni.applyBoundary();
  Vi.applyBoundary();
  Te.applyBoundary();  
  Ti.applyBoundary();
  Nn.applyBoundary();
  Tn.applyBoundary();
  Vn.applyBoundary();
  Nm.applyBoundary();
  Vm.applyBoundary();
  Vort.applyBoundary();   //ZYLHJ
  psi.applyBoundary();
  //smooth noisies
  if(nlfilter_noisy_data)
    {
      //Ni=nl_filter(Ni,filter_para);
      //Vi=nl_filter(Vi,filter_para);
      Nn=nl_filter(Nn,filter_para);
      Vn=nl_filter_y(Vn,filter_para);

    }
  
  //*****@!!@*****
  // NB: Any value re-assignment should be given HERE ONLY!
  //*****@!!@***** 

  temp_Ni=field_larger(Ni,minimum_val);
  temp_Nn=field_larger(Nn,minimum_val);   // in case divided by zero 
  temp_Nm=field_larger(Nm,minimum_val);   //1/temp_N only used to replay 1/N
  temp_Te=field_larger(Te,minimum_val);   // necessary for sheath BC
  temp_Ti=field_larger(Ti,minimum_val);   // necessary for sheath BC

  Ni=field_larger(Ni,minimum_val);  
  Nn=field_larger(Nn,minimum_val);  
  Nm=field_larger(Nm,minimum_val);
  //Ti=field_larger(Ti,minimum_val); 
  //Te=field_larger(Te,minimum_val);  

  Nelim = floor(Ni, 1e-5);
  Telim = floor(Te, 0.1 / Te_x);


   Field3D temp_Pi= temp_Ni*temp_Ti;

   U = Vort;

   U -= Upara1/B0 * Delp2(temp_Pi);
   U /= B0;

TRACE("Electrostatic potential");

if (phi3d) 
{
#ifdef PHISOLVER
      phiSolver3D->setCoefC(temp_Ni / SQ(mesh->Bxy));      //////Ne=Ni

      if (mesh->lastX()) {
        for (int i = mesh->xend + 1; i < mesh->ngx; i++)
          for (int j = mesh->ystart; j <= mesh->yend; j++)
            for (int k = 0; k < mesh->ngz - 1; k++) {
              phi(i, j, k) = 3. * Te(i, j, k);
            }
      }
      phi = phiSolver3D->solve(U, phi);
#endif

    } else {
      BoutReal sheathmult = log(0.5 * sqrt(mi_me / PI));

      if (boussinesq) {

        if (split_n0) {
          Field2D U2D = U.DC(); // n=0 component
          phi2D.setBoundaryTo(sheathmult * Telim.DC());

          phi2D = laplacexy->solve(U2D, phi2D);
          
          // Solve non-axisymmetric part using X-Z solver

          if (newXZsolver) {
            newSolver->setCoefs(1. / SQ(mesh->Bxy), 0.0);
            phi = newSolver->solve(U - U2D, phi);
          } else {
            phiSolver->setCoefC(1. / SQ(mesh->Bxy));
            // phi = phiSolver->solve((Vort-Vort2D)*SQ(mesh->Bxy), phi);

            phi = phiSolver->solve((U - U2D) * SQ(mesh->Bxy),
                                   sheathmult * (Telim - Telim.DC()));
          }
          phi += phi2D; // Add axisymmetric part
       
        } else {

          ////////////////////////////////////////////
          // Boussinesq, non-split
          // Solve all components using X-Z solver

          if (newXZsolver) {
            phi = newSolver->solve(U, phi);
          } else {
            phi = phiSolver->solve(U * SQ(mesh->Bxy), sheathmult * Telim);

          }
        }
      } else {
        phiSolver->setCoefC(Nelim / SQ(mesh->Bxy));
        phi = phiSolver->solve(U * SQ(mesh->Bxy) / Nelim, sheathmult * Telim);
      }
    }
    phi.applyBoundary();
    mesh->communicate(phi);
  
V = -Grad(phi);
 V.applyBoundary();
    mesh->communicate(V);

/////////////////// 
  Tn=temp_Ti;

  Vmx=Vm.x;
  if(SMBI_LFS)
  {
  Nm=ret_const_flux_BC(Nm, Nm0);
  Vmx=ret_const_flux_BC(Vmx, Vm0); 
  }
  else
 {
  Nm=ret_const_flux_BC(Nm,minimum_val);
  Vmx=ret_const_flux_BC(Vmx,0.0);
  }

  // Update non-linear coefficients on the mesh
  //kappa_Te = 3.2*(1./fmei)*(wci/nueix)*(Tet^2.5);
  //kappa_Ti = 3.9*(wci/nuiix)*(Tit^2.5);
  kappa_Te = 3.2*(1./fmei)*(1./tbar/nueix)*(temp_Te^2.5);
  kappa_Ti = 3.9*(1./tbar/nuiix)*(temp_Ti^2.5);  

// Set flux limit for kappa
  V_th_e= 4.19e5*sqrt(temp_Te*Te_x)*tbar/Lbar;
  V_th_i= 9.79e3*sqrt(temp_Ti*Te_x/AA)*tbar/Lbar;

  kappa_Te_fl = V_th_e*q95_input*temp_Ni;  // Ne=Ni quasineutral
  kappa_Ti_fl = V_th_i*q95_input*temp_Ni;

  // Thermal Speed normalized in ion thermal speed V_thi
  c_se = sqrt((temp_Te+temp_Ti)/Mi); 

  //parallel heat fluxes to calculate Te,i gradients at SBC
  q_se_kappa = -C_fe_sheat*Ni*Te*c_se/kappa_Te;                                     // '-' means out-flowing to Sheath
  q_si_kappa = -C_fi_sheat*Ni*Ti*c_se/kappa_Ti;   

   
  // Apply Sheath Boundary Condition at outside of Separitrix 
  // NB: SBC applied at ydown (theta=0) and yup (theta=2PI) are different due to fluxes flowing 
  // towards X point and hitting on the plates, thus SBC_ydown(var,-value) while SBC_yup(var,value) 
  // If particle recycling, Ni flux Gamma_ni outflows to divertor plates while Nn flux Gamma_nn inflows 
  // from the plates, Gamma_nn = Gamma_ni * Rate_recycle 
  
  if(Sheath_BC)
    {
      SBC_Dirichlet(Vi, c_se); 
      SBC_Gradpar(Ni, 0.);
      SBC_Gradpar(Te, q_se_kappa);
      SBC_Gradpar(Ti, q_si_kappa);
     }


  // no flux limitation for kappa at Sheath BC 
  kappa_Te *= kappa_Te_fl/(kappa_Te_fl+kappa_Te);
  kappa_Ti *= kappa_Ti_fl/(kappa_Ti_fl+kappa_Ti);

  // Collisional time ion-electrons
  tau_ei = 1./tbar/(nueix*temp_Ni/(temp_Te^1.5));

  //ion viscosity
  eta0_i=0.96*(1./tbar/nuiix)*temp_Ni*temp_Ti;
  eta0_n=0.96*(1./tbar/nuiix)*temp_Nn*temp_Ti;
 // eta0_i=0.96*(1./tbar/nuiix)*(temp_Ti^2.5);
 // eta0_n=0.96*(1./tbar/nuiix)*(Tn^2.5)*Nn/temp_Nn;
  
  //updata collisional rates 
 
  // Ionization rate  depending on Nn for Plasmas 
  nu_ionz = 3.e3*tbar*temp_Nn*Ni_x*temp_Te*temp_Te*Te_x*Te_x/(3.+0.01*temp_Te*temp_Te*Te_x*Te_x);   //Ni_x in unit 1e^19 
  nu_CX =  1.e5*tbar*temp_Nn*Ni_x*(1.7+0.667*(((1.5*temp_Ti*Te_x)^0.333)-2.466));      // empirical formula 
  
  //Diag_neg_value(nu_ionz,nu_CX,Nn,Nm);


  // Ionization rate  depending on Ni for Neutrals 
  nu_ionz_n = 3.e3*tbar*temp_Ni*Ni_x*temp_Te*temp_Te*Te_x*Te_x/(3.+0.01*temp_Te*temp_Te*Te_x*Te_x);    //Ni_x in unit 1e^19 
  nu_CX_n = 1.e5*tbar*temp_Ni*Ni_x*(1.7+0.667*(((1.5*temp_Ti*Te_x)^0.333)-2.466));      // empirical formula 

 //Dissociation rate of molecules
  nu_diss = 3.e3*tbar*temp_Nm*Ni_x*temp_Te*temp_Te*Te_x*Te_x/(3.+0.01*temp_Te*temp_Te*Te_x*Te_x);     //need be corrected
  //nu_diss =0.;

  // recombination rate of ions and electrons
  Field3D lambda_rec=1.5789e5/(temp_Te*Te_x*1.1604e4);  //Te trasfer unit from eV to Kelvin  
  nu_rec = tbar*temp_Ni*Ni_x*5.197e-1*ZZ*sqrt(lambda_rec)*(0.4288+0.5*log(lambda_rec)+0.469/(lambda_rec)^0.333); //Seaton M F 1959 'Radiative recombination of hydrogenic ions'

  // source terms
  Si_p = temp_Ni*nu_ionz;
  Si_CX = temp_Ni*nu_CX;
  S_diss = temp_Ni*nu_diss;  
  S_rec = temp_Ni*nu_rec;
/*
////ZYLHJ
Jpara1= 1.96*1.71*ee*MU0*Ni_x*tbar*KB*Te_x*eV_K/bmag/Me;

Jpara2= 1.96*ee*MU0*Ni_x*tbar*KB*Te_x*eV_K/bmag/Me;

Jpara3= 1.96*ee*ee*Lbar*Lbar*MU0*Ni_x/Me;

Jpar = Jpara1*Grad_par(Te)

      +(Jpara2*temp_Te*Grad_par(Ni))/temp_Ni

      -Jpara3*Grad_par(phi);

Jpar *= tau_ei*temp_Ni;

         mesh->communicate(Jpar);
           Jpar.applyBoundary();
*/



  //Set Boundary and Apply Boundary for output variables
  tau_ei.applyBoundary();
  nu_ionz.applyBoundary();
  nu_CX.applyBoundary();
  nu_ionz_n.applyBoundary();
  nu_CX_n.applyBoundary();
  Si_p.applyBoundary();
  Si_CX.applyBoundary();
  S_diss.applyBoundary();

  if(t==0.)  output.write("\n **** t %e  nu_I %e nu_I_n %e Si_p %e tau_ei %e ****\n",t,max(nu_ionz),max(nu_ionz_n),max(Si_p),max(tau_ei)); 
  
  // Perpensicular 3D Diffusion coefficients of Plasmas, set constant by default
  Diff_ni_perp = Diffc_ni_perp;        
  chi_i_perp = chic_i_perp;
  chi_e_perp = chic_e_perp;
  if(diffusion_coef_step_function)
    {
      Diff_ni_perp = Diffc_ni_step;        
      chi_i_perp = chic_i_step;
      chi_e_perp = chic_e_step;
     }
  if(diffusion_coef_Hmode)
    {
      Diff_ni_perp = Diffc_ni_Hmode;        
      chi_i_perp = chic_i_Hmode;
      chi_e_perp = chic_e_Hmode;
     }

  // Fluxes average over Y calculated
     DDX_Ni = DDX(Ni);
     mesh->communicate(DDX_Ni);
     DDX_Ni.applyBoundary();

     DDX_Ti = DDX(Ti);
     mesh->communicate(DDX_Ti);
     DDX_Ti.applyBoundary();

     DDX_Te = DDX(Te);
     mesh->communicate(DDX_Te);
     DDX_Te.applyBoundary();

     Flux_Ni_aveY=-Diff_ni_perp*DDX_Ni*aveY_g11J;
     Flux_Ti_aveY=-chi_i_perp*DDX_Ti*aveY_g11J;
     Flux_Te_aveY=-chi_e_perp*DDX_Te*aveY_g11J;

     mesh->communicate(Flux_Ni_aveY);
     Flux_Ni_aveY.applyBoundary();

     mesh->communicate(Flux_Ti_aveY);
     Flux_Ti_aveY.applyBoundary();

     mesh->communicate(Flux_Te_aveY);
     Flux_Te_aveY.applyBoundary();

  //parallel particle fluxes to calculate Ni,Nn gradients at SBC
  //Diffc_nn_perp = Tn/Mn/nu_CX_n;   
  Diffc_nn_perp = Tn/Mn/(nu_CX_n+2.*S_diss/temp_Nn);
  if(terms_recombination) Diffc_nn_perp = Tn/Mn/(nu_CX_n+2.*S_diss/temp_Nn+S_rec/temp_Nn);
  V_th_n=sqrt(Tn/Mn);
  Diffc_nn_perp_fl=V_th_n*Lnn_min/Lbar;
  Diffc_nn_perp *= Diffc_nn_perp_fl/(Diffc_nn_perp+Diffc_nn_perp_fl);                 // Diffc_nn_perp calculated once for SBC and again later
  
  Diffc_nn_par = Tn/Mn/(nu_CX_n+2.*S_diss/temp_Nn);
  if(terms_recombination) Diffc_nn_par = Tn/Mn/(nu_CX_n+2.*S_diss/temp_Nn+S_rec/temp_Nn);
  Diffc_nn_par *= Diffc_nn_perp_fl/(Diffc_nn_par+Diffc_nn_perp_fl); 

  Gamma_ni_wall = Diff_ni_perp*Ni/Lni_wall;
  Gamma_nn_wall = Rate_recycle*Gamma_ni_wall;
  Gamma_nnw_Diffc = Gamma_nn_wall/Diffc_nn_perp;
  Gamma_ni = Ni*c_se;
  Gamma_nn = Rate_recycle*Gamma_ni;

  Gamma_ni_Diffc= -Gamma_ni/Diff_ni_perp;                                    // '-' means out-flowing to Sheath
  Gamma_nn_Diffc=  Gamma_nn/Diffc_nn_perp;                                    // '+' in-flowing from Sheath

  if(Wall_particle_recycle)
    {

      //WallBC_Xout_GradX(Ni,Gamma_ni_Diffc);                                                     // '-' means out-flowing to wall
     WallBC_Xout_GradX_len(Ni, -1./Lni_wall);                                                   //fixed Gradient length (real unit m) at wall
     WallBC_Xout_GradX(Nn,Gamma_nnw_Diffc);                                                     // '+' means in-flowing from wall

     } 

  if(SBC_particle_recycle)
     {      
       /*
       Vn_perp = Diffc_nn_perp*abs(DDX(Nn))*Rxy*abs(Bpxy)/temp_Nn;
       mesh->communicate(Vn_perp);
       Vn_perp.applyBoundary();
       Vn_th_plate = sqrt(Tn_plate/Mn);
       Vn_perp = field_larger(Vn_perp,Vn_th_plate);

       //Vn_pol = -abs(Vn)*sin(angle_B_plate)+abs(Vn_perp)*cos(angle_B_plate);
       Vn_pol = abs(Vn_perp)*cos(angle_B_plate);   //Assumption Vn_perp >> Vn_par
       Vn_pol = field_larger(Vn_pol,Vn_th_plate);
       if(nlfilter_Vnpol)nl_filter(Vn_pol);
     
       SBC_Dirichlet(Nn, Gamma_nn/Vn_pol);
       */
       
       SBC_Gradpar(Nn, Gamma_nn/Diffc_nn_par);

     }
  


//************************************
  // DENSITY EQUATION
//************************************
  // output.write("Now updata Ni \n"); 
  //ddt(Ni) = 0.; 
  
  ddt(Ni) =   Diff_ni_perp*Delp2(Ni)
            - Ni*Grad_par(Vi)
            - Vpar_Grad_par(Vi,Ni)
            + Si_p; 
  if(terms_recombination) ddt(Ni) -= S_rec;
  if(external_sources) ddt(Ni) += S_pi_ext;
  if(terms_Gradperp_diffcoefs) 
    {
      grad_perp_Diffi=Grad_perp(Diff_ni_perp);
      grad_perp_Diffi.applyBoundary();
      mesh->communicate(grad_perp_Diffi);  
//      ddt(Ni) +=  V_dot_Grad(grad_perp_Diffi,Ni); 
      ddt(Ni) +=  grad_perp_Diffi*Grad_perp(temp_Ni); 
      //ddt(Ni) += grad_perp_Diffi.x+grad_perp_Diffi.z;//DDX(Diff_ni_perp);//*DDX(Ni); 
    }

if(term_driftExB)
 {
bracketphiNi =  bracket(phi, temp_Ni, BRACKET_ARAKAWA); 
bracketphiNi.applyBoundary();
mesh->communicate(bracketphiNi);
 ddt(Ni) -= bracketphiNi;
}

if(term_driftcurvature)
 {
   Nipara0 = KB*Te_x*eV_K*tbar/(ee*Lbar*Lbar*bmag);

//   Curlb_B = Curl(B0vec)/B0;
//   Ni_curvature1 = V_dot_Grad(Curlb_B,Pi);
//   Ni_curvature2 = temp_Ni*V_dot_Grad(Curlb_B,phi);

   Ni_curvature1 = V_dot_Grad(b0xcv,Pi);
   Ni_curvature2 = temp_Ni*V_dot_Grad(b0xcv,phi);

Ni_curvature1.applyBoundary();
mesh->communicate(Ni_curvature1);

Ni_curvature2.applyBoundary();
mesh->communicate(Ni_curvature2);

   ddt(Ni) -= Nipara0*Ni_curvature1;
   ddt(Ni) -= Ni_curvature2;
}

  

  /*
  if(t>0. && t<1.e4) {

         ddt(Ni)+=Src_ni;
         output.write("\n  t %e source injected at jx %d Ni %e \n",t,jxs_ni,Ni[jxs_ni][32][0]);
    }
  */          
  

//************************************
  // ELECTRON TEMPERATURE   ---Te---
//************************************

   //output.write("Now updata Te \n");
 
   //ddt(Te)=0.0;
   
  ddt(Te) =  0.6667*kappa_Te*Grad2_par2(Te)/temp_Ni
            // + 0.6667*Grad_par(kappa_Te)*Grad_par(Te)/temp_Ni
             + 0.6667*chi_e_perp*Delp2(Te)
             - nu_ionz*(Te+0.6667*W_ionz)
             - 0.6667*nu_diss*(W_diss+W_bind)
             - 2.*fmei*(Te-Ti)/tau_ei
             ;
   if(terms_recombination) ddt(Te) += nu_rec*W_rec;
   if(term_GparkappaTe_GparTe) ddt(Te) += 0.6667*Grad_par(kappa_Te)*Grad_par(Te)/temp_Ni;
   if(external_sources) ddt(Te) += 0.6667*S_Ee_ext/temp_Ni;
   if(terms_Gradperp_diffcoefs) 
    {
      grad_perp_chie=Grad_perp(chi_e_perp);
      grad_perp_chie.applyBoundary();
      mesh->communicate(grad_perp_chie);  
      ddt(Te) +=  0.6667*V_dot_Grad(grad_perp_chie,Te); 
    }

if(term_driftExB)
 {
  bracketphiTe =  bracket(phi, temp_Te, BRACKET_ARAKAWA);
  bracketphiTe.applyBoundary();
  mesh->communicate(bracketphiTe);
  ddt(Te) -= bracketphiTe;
}
 
if(term_driftcurvature)
 {
    Tepara0 = KB*Te_x*eV_K*tbar/(ee*Lbar*Lbar*bmag);
    Te_curvature1 = 0.66667*temp_Te*V_dot_Grad(b0xcv,Pe)/temp_Ni;
    Te_curvature2 = 0.66667*temp_Te*V_dot_Grad(b0xcv,phi);
    Te_curvature3 = 1.66667*temp_Te*V_dot_Grad(b0xcv,Te);

Te_curvature1.applyBoundary();
mesh->communicate(Te_curvature1);

Te_curvature2.applyBoundary();
mesh->communicate(Te_curvature2);

Te_curvature3.applyBoundary();
mesh->communicate(Te_curvature3);

   ddt(Te) += Tepara0*Te_curvature1;
   ddt(Te) -= Te_curvature2;
   ddt(Te) += Tepara0*Te_curvature3;
}

if(term_parallel_current)
{
    Tepara1 = bmag*tbar/(ee*Ni_x*density_unit*Lbar*Lbar*MU0);
//    Tepara2 = Bmag*Bmag*Me
Te_current1 = 0.66667*0.71*temp_Te*Grad_par(Jpar);
Te_current1.applyBoundary();
mesh->communicate(Te_current1);

 ddt(Te) += Tepara1*Te_current1;
}

//************************************
  // ION TEMPERATURE   ---Ti---
//************************************
  
   //output.write("Now updata Ti \n");

  //ddt(Ti)=0.0;
  
   ddt(Ti) = - Vpar_Grad_par(Vi,Ti)
             - 0.6667*Ti*Grad_par(Vi)
             + 0.6667*kappa_Ti*Grad2_par2(Ti)/temp_Ni
             + 0.6667*Grad_par(kappa_Ti)*Grad_par(Ti)/temp_Ni
             + 0.6667*chi_i_perp*Delp2(Ti)
             - nu_ionz*Ti
             - 0.6667*nu_CX*(Ti-Tn)
             + 2.*fmei*(Te-Ti)/tau_ei
          ;
   if(terms_recombination)  ddt(Ti) += nu_rec*Ti;
   if(external_sources) ddt(Ti) += 0.6667*S_Ei_ext/temp_Ni;
   if(terms_Gradperp_diffcoefs) 
    {
      grad_perp_chii=Grad_perp(chi_i_perp);
      grad_perp_chii.applyBoundary();
      mesh->communicate(grad_perp_chii);  
      ddt(Ti) +=  0.6667*V_dot_Grad(grad_perp_chii,Ti); 
    }

if(term_driftExB)
 {
  bracketphiTi =  bracket(phi, temp_Ti, BRACKET_ARAKAWA);
  bracketphiTi.applyBoundary();
  mesh->communicate(bracketphiTi);

  bracketphipei =  bracket(phi, pei, BRACKET_ARAKAWA);
  bracketphipei.applyBoundary();
  mesh->communicate(bracketphipei);

  ddt(Ti) -= bracketphiTi;
  ddt(Ti) -= bracketphipei;

}

if(term_driftcurvature)
 {
    Tipara0 = KB*Te_x*eV_K*tbar/(ee*Lbar*Lbar*bmag);
    Ti_curvature1 = 0.66667*temp_Ti*V_dot_Grad(b0xcv,Pi)/temp_Ni;
    Ti_curvature2 = 0.66667*temp_Ti*V_dot_Grad(b0xcv,phi);
    Ti_curvature3 = 1.66667*temp_Ti*V_dot_Grad(b0xcv,Ti);

Ti_curvature1.applyBoundary();
mesh->communicate(Ti_curvature1);

Ti_curvature2.applyBoundary();
mesh->communicate(Ti_curvature2);

Ti_curvature3.applyBoundary();
mesh->communicate(Ti_curvature3);

   ddt(Ti) -= Tipara0*Ti_curvature1;
   ddt(Ti) -= Ti_curvature2;
   ddt(Ti) -= Tipara0*Ti_curvature3;
}

if(term_parallel_current)
{
   Tipara1 = bmag*bmag/(Ni_x*density_unit*KB*Te_x*eV_K*MU0);
   Ti_current1 = 0.66667*Jpar*Grad_par(phi)/temp_Ni;
   Ti_current1.applyBoundary();
   mesh->communicate(Ti_current1);

    ddt(Ti) += Tipara1*Ti_current1;
}
   


  //^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // Turbulent Diffusion Update for Ni Te Ti
  //___________________________________________  

  pei = (Te+Ti)*Ni;

//************************************
  // ION Paralell VELOCITY  ---Vi---
//************************************

  //output.write("Now updata Vi \n");
   Grad_par_pei=Grad_par(pei);
   Grad_par_pei.applyBoundary();
   //mesh->communicate(Grad_par_pei);
   if(NOT_SOLVE_FOR_Vi)
     {
      ddt(Vi)=0.0;
     }
   else
   {
   ddt(Vi) = - Vpar_Grad_par(Vi,Vi)
             - Grad_par(pei)/temp_Ni/Mi
             + 2.*0.6667*Grad_par(eta0_i)*Grad_par(Vi)/temp_Ni/Mi
             + 2.*0.6667*eta0_i*Grad2_par2(Vi)/temp_Ni/Mi
             - (nu_ionz+nu_CX)*(Vi-Vn)
             ;             
   }

if(term_driftExB)
{
   bracketphiVi =  bracket(phi, Vi, BRACKET_ARAKAWA);
   bracketphiVi.applyBoundary();
   mesh->communicate(bracketphiVi);
   ddt(Vi) -= bracketphiVi;
}

if(term_drift_diamag)
 {
   Vipara0 = KB*Te_x*eV_K*tbar/(ee*Lbar*Lbar*bmag);
   bracketPiVi = bracket(Pi, Vi, BRACKET_ARAKAWA) / temp_Ni;

   bracketPiVi.applyBoundary();
   mesh->communicate(bracketPiVi);
 
   ddt(Vi) -= Vipara0*bracketPiVi;
}


////////////ZYLHJ
//////Current///////
    Pe = temp_Ni*temp_Te;
Jpara0 = 1.;
Jpara1 = tbar/Lbar/Lbar;
Jpara2 = KB*Te_x*eV_K*tbar/(ee*bmag*Lbar*Lbar);
Jpara3 = 0.71*KB*Te_x*eV_K*tbar/(ee*bmag*Lbar*Lbar);

Dperp2psi = Delp2(psi);
Dperp2psi.applyBoundary();
mesh->communicate(Dperp2psi);

ddt(psi) = -Jpara0*Grad_par(phi)/B0 + Jpara1*eta*Dperp2psi;
ddt(psi) += Jpara2*Grad_par(Pe)/(temp_Ni*B0);
ddt(psi) += Jpara3*Grad_par(temp_Te)/B0;

Jpar = -Delp2(psi);
Jpar.applyBoundary();
mesh->communicate(Jpar);

//************************************
//////// //Vorticity   ---U--- ZYLHJ
//////// //************************************
////////
//
          GradPhi2 = Grad_perp(phi)*Grad_perp(phi) / (B0^2);
          GradPhi2.applyBoundary();
          mesh->communicate(GradPhi2);

          bracketPhiP = bracket(phi, Pi, BRACKET_ARAKAWA);
          bracketPhiP.applyBoundary();
          mesh->communicate(bracketPhiP);

          Dperp2Phi = Delp2(phi);
          Dperp2Phi.applyBoundary();
          mesh->communicate(Dperp2Phi);

          Dperp2Pi = Delp2(Pi);
          Dperp2Pi.applyBoundary();
          mesh->communicate(Dperp2Pi);

ddt(Vort) = -Vpar_Grad_par(Vi,Vort);
         - bracket(phi, Vort, BRACKET_ARAKAWA)
         + Upara0*(B0^2)*Grad_par(Jpar/B0);

          ddt(Vort) -=0.5*Upara1*bracket(Pi,Dperp2Phi, BRACKET_ARAKAWA)/B0;
          ddt(Vort) +=0.5*Upara2*B0*bracket(Ni,GradPhi2,BRACKET_ARAKAWA);
          ddt(Vort) += 0.5*Upara1*bracket(phi, Dperp2Pi, BRACKET_ARAKAWA)/B0;
          ddt(Vort) -= 0.5*Upara1*Delp2(bracketPhiP)/B0;


//************************************
 //Neutral Paralell Velocity   ---Vn---
//************************************

   //output.write("Now updata Vn \n");
  
   pn=Nn*Tn;                            
   Grad_par_pn=Grad_par(pn);
   Grad_par_pn.applyBoundary();
   // mesh->communicate(Grad_par_pn);
   Grad_par_Tn=Grad_par(Tn);
   Grad_par_Tn.applyBoundary();
   //mesh->communicate(Grad_par_Tn);
   temp_Nn=field_larger(Nn,minimum_val);
   Grad_par_logNn=Grad_par(log(temp_Nn));
   Grad_par_logNn.applyBoundary();
   mesh->communicate(Grad_par_logNn);
   if(nlfilter_Gradpar_logNn)nl_filter(Grad_par_logNn,filter_para);
   /*
   ddt(Vn) = - Vpar_Grad_par(Vn,Vn)       // upwind convection
             + 2.*0.6667*eta0_n*Grad2_par2(Vn)/temp_Nn/Mn
             + nu_CX_n*(Vi-Vn)
             - 2.*S_diss*Vn/temp_Nn
             ;    
   if(terms_recombination) ddt(Vn) += S_rec*(Vi-Vn)/temp_Nn;
   if(terms_Gradpar_pn)    ddt(Vn) -= (Grad_par(Tn)/Mn;                 // Part I of Grad_par(Pn) term
                                     + Grad_par(log(temp_Nn))*Tn/Mn);   // Part II of Grad_par(Pn) term
   if(terms_Gradpar_eta0n) ddt(Vn) += 2.*0.6667*Grad_par(log(temp_Nn))*Grad_par(Vn)*eta0_n/temp_Nn/Mn  // Part I of Grad_par(eta0_n) term
    	                             +2.*0.6667*Grad_par(Tn)*Grad_par(Vn)*eta0_n/Tn/temp_Nn/Mn;       // Part II of Grad_par(eta0_n) term
   */
   ddt(Vn)=0.0;          // Vn-->Diffc_nn_par
   //Vn = (Vi*nu_CX_n -Grad_par_Tn/Mn-Grad_par_logNn*Tn/Mn)/(nu_CX_n+2.*S_diss/temp_Nn);
   //Vn = Vi-(Grad_par_Tn/Mn+Grad_par_logNn*Tn/Mn)/nu_CX_n;

//************************************
// Neutral Density       ---Nn---
//************************************

//output.write("Before Nn \n");
  
   //Diffc_nn_perp = Tn/Mn/nu_CX_n;               // Diffc_nn_perp re-calculated   
   //V_th_n=sqrt(Tn/Mn);         
   //Diffc_nn_perp_fl=V_th_n*Lnn_min/Lbar;
   //Diffc_nn_perp *= Diffc_nn_perp_fl/(Diffc_nn_perp+Diffc_nn_perp_fl); 

 // ddt(Nn)=0.0;

   ddt(Nn)=//- Vpar_Grad_par(Vn,Nn)  
          //- Nn*Grad_par(Vn)  re-written in more terms below
          + Diffc_nn_perp*Delp2(Nn)
          - Si_p
          + 2.*S_diss
          ;
  if(terms_recombination) ddt(Nn) += S_rec;
  if(terms_NnGradpar_Vn) ddt(Nn) -= ( Nn*Grad_par(Vi)
				     -Nn*Grad2_par2(Tn)/Mn/nu_CX_n
				     -Nn*Grad_par(Grad_par_logNn)*Tn/Mn/nu_CX_n
				      ); 
  if(terms_Diffcnn_par)  ddt(Nn) += ( Diffc_nn_par*Nn*Grad_par(Grad_par_logNn)
                                      +Nn*Grad_par_logNn*Grad_par(Diffc_nn_par)
                                     );
  /*   // tested with some zig-zag problems at SOL region
  if(terms_Gradperp_diffcoefs) 
    {
      grad_perp_Diffn=Grad_perp(Diffc_nn_perp);
      grad_perp_Diffn.applyBoundary();
      mesh->communicate(grad_perp_Diffn);  
      ddt(Nn) +=  V_dot_Grad(grad_perp_Diffn,Nn); 
    }
   */ 
  //Nn.applyBoundary();

//************************************
// Neutral Temperature       ---Tn---
//************************************

  ddt(Tn)=0.0;
   /*
  ddt(Tn)= - Vpar_Grad_par(Vn,Tn)
          - 0.6667*Tn*Grad_par(Vn)
          + 0.6667*chic_n_perp*Delp2(Tn)
    // + (Si_p-2.*S_diss)*Tn/temp_Nn
          + 0.6667*nu_CX_n*(Ti-Tn)
    // + 0.6667*S_diss*W_diss/temp_Nn
          ;

  */
//************************************
  //Molecule Perpendicular Velocity in X ---Vmx---
 //************************************
  
 pm=Nm*Tm_x; 
//   ddt(Vmx) =0.;
 Vm.x=Vmx;
 Vm.y=0.;
 Vm.z=0.;
 ddt(Vm) = - V_dot_Grad(Vm,Vm) 
           - Grad(pm)/temp_Nm/Mm
          ;

 /* 
 ddt(Vmx) = - VDDX(Vmx,Vmx)//(mesh->J*sqrt(mesh->g_22))
            - DDX(pm)/temp_Nm/Mm
            ;
 */
//************************************
// Molecule Density       ---Nm---
//************************************
 //ddt(Nm)=0.0;

 ddt(Nm)=-V_dot_Grad(Vm,Nm)-Nm*Div(Vm) - S_diss;

 /*
 ddt(Nm)=- VDDX(Vmx,Nm)//(mesh->J*sqrt(mesh->g_22))
         - Nm*DDX(Vmx)
    //+ 1.e5*Diffc_nm_perp*Delp2(Nm)
   // - S_diss
          ; 
 */

  //Diagnose whether there are negative value

  //Diag_neg_value(Nn,Nm,Tn,Te);

  //Bootstrap current calculated by using Sauter's formula 
 if (BScurrent)
    {

      q95=q95_input;
      pei= Ni*(Te+Ti);
      Pe = Ni*Te;
      Pi = Ni*Ti;

      nu_estar = 100.*nueix * q95*tbar / (V_th_e) / Aratio^(1.5);
      nu_istar = 100.*nuiix * q95*tbar / (V_th_i) / Aratio^(1.5);

      ft = BS_ft(100);
      f31 = ft / (1.+(1.-0.1*ft)*sqrt(nu_estar) + 0.5*(1.-ft)*nu_estar/Zi);
      f32ee = ft / (1.+0.26*(1.-ft)*sqrt(nu_estar) + 0.18*(1.-0.37*ft)*nu_estar/sqrt(Zi));
      f32ei = ft / (1.+(1.+0.6*ft)*sqrt(nu_estar) + 0.85*(1.-0.37*ft)*nu_estar*(1.+Zi));
      f34 = ft / (1.+(1.-0.1*ft)*sqrt(nu_estar) + 0.5*(1.-0.5*ft)*nu_estar/Zi);

      L31 = F31(f31) ;
      L32 = F32ee(f32ee)+F32ei(f32ei) ;
      L34 = F31(f34) ;

      BSal0 = - (1.17*(1.-ft))/(1.-0.22*ft-0.19*ft*ft);
      BSal = (BSal0+0.25*(1-ft*ft)*sqrt(nu_istar))/(1.+0.5*sqrt(nu_istar)) + 0.31*nu_istar*nu_istar*ft*ft*ft*ft*ft*ft;
      BSal *= 1./(1.+0.15*nu_istar*nu_istar*ft*ft*ft*ft*ft*ft);

      Jpar_BS0 = L31* DDX(pei)/Pe  + L32*DDX(Te)/Te + L34*DDX(Ti)/(Zi*Te)*BSal;
      Jpar_BS0 *= Field3D( -Rxy*Btxy*Pe*(MU0*KB*Ni_x*density_unit*Te_x*eV_K)/(mesh->Bxy*mesh->Bxy)/(bmag*bmag) ); //NB:   J_hat = MU0*Lbar * J / mesh->Bxy;

      mesh->communicate(Jpar_BS0);
      Jpar_BS0.applyBoundary();

    }



  return(0);
}




void update_turb_effects(Field3D &f, const BoutReal Difft_coef, const BoutReal value_crit)
{

bool first_time=true;
       Field3D dNi_dx=DDX(Ni);
       BoutReal lowest_Lp=1.;
       int jx_lowest_Lp;
       Heavistep=0.;
      dpei_dx=DDX(pei);
      for (int jx=0;jx<mesh->ngx;jx++)
        {
           for (int jy=0;jy<mesh->ngy;jy++)
	      {
 	        for (int jz=0;jz<mesh->ngz;jz++)
                   {
                                
                     if(abs(dpei_dx[jx][jy][jz])<1.e-5) dpei_dx[jx][jy][jz]=1.e-5;
                     if(jx<=1 )pei[jx][jy][jz]=Ni[jx+2][jy][jz]*(Te[jx+2][jy][jz]+Ti[jx+2][jy][jz]);
                     if(jx>=mesh->ngx-2 )pei[jx][jy][jz]=Ni[jx-2][jy][jz]*(Te[jx-2][jy][jz]+Ti[jx-2][jy][jz]);
		     Lp=pei[jx][jy][jz]/(dpei_dx[jx][jy][jz]);
	             if(abs(Lp)<lowest_Lp) {lowest_Lp =abs(Lp);jx_lowest_Lp=jx;}
                     //if (abs(Lp) <= Lp_crit)
                       if (abs(dNi_dx[jx][jy][jz])>=9.e2)
                        {
                             Heavistep[jx][jy][jz]=1.;

                          }
		     else Heavistep[jx][jy][jz]=0.;

	              }
        	}
     	}
      ddt(f) += Heavistep*Difft_coef*Delp2(f);


}


//3D Boundaries of constant neutral flux injection
  

const Field3D ret_const_flux_BC(const Field3D &var, const BoutReal value)

{

  Field3D result;

  result.allocate();

    for (int jx=0;jx<mesh->ngx;jx++)
        {

          BoutReal x_glb= mesh->GlobalX(jx); 

           for (int jy=0;jy<mesh->ngy;jy++)
	      {
                 BoutReal y_glb=mesh->GlobalY(jy);

 	        for (int jz=0;jz<mesh->ngz;jz++)
                   {
                     BoutReal z_glb=(BoutReal)jz/(BoutReal)(mesh->ngz-1);    // no parallization in Z   
                      
		     if(x_glb>=CF_BC_x0 && y_glb>=CF_BC_y0 && y_glb<=CF_BC_y1 && z_glb>=CF_BC_z0 && z_glb<=CF_BC_z1) 
		       { 
			BoutReal CF_BC_yhalf=0.5*(CF_BC_y0+CF_BC_y1);
                        BoutReal CF_ywidth=CF_BC_yhalf-CF_BC_y0;
                        BoutReal CF_exp_decay=exp(-(y_glb-CF_BC_yhalf)*(y_glb-CF_BC_yhalf)/CF_ywidth/CF_ywidth);
			result[jx][jy][jz]=value*CF_exp_decay; 
                         }
                     else 
                         result[jx][jy][jz]=var[jx][jy][jz];

	              }
        	}
     	}

  mesh->communicate(result);

  return(result);


}

     

void Diag_neg_value (const Field3D &f1,const Field3D &f2,const Field3D &f3,const Field3D &f4 )
{
for (int jx=0;jx<mesh->ngx;jx++)
   {
     BoutReal x_glb=mesh->GlobalX(jx);
      for( int jy=0;jy<mesh->ngy;jy++)
        {
         BoutReal y_glb=mesh->GlobalY(jy);
         for (int jz=0;jz<mesh->ngz;jz++)
           {
             if(f1[jx][jy][jz]<0.) output.write("Field 1 becomes negative %e at x %e y %e \n",f1[jx][jy][jz],x_glb,y_glb);
             if(f2[jx][jy][jz]<0.) output.write("Field 2 becomes negative %e at x %e y %e \n", f2[jx][jy][jz],x_glb,y_glb);
             if(f3[jx][jy][jz]<0.) output.write("Field 3 becomes negative %e at x %e y %e \n",f3[jx][jy][jz],x_glb,y_glb);
             if(f4[jx][jy][jz]<0.) output.write("Field 4 becomes negative %e at x %e y %e \n", f4[jx][jy][jz],x_glb,y_glb);
            }
         }    
    }


}


const Field3D field_larger(const Field3D &f, const BoutReal limit)

{

  Field3D result;

  result.allocate();

//  #pragma omp parallel for

  for(int jx=0;jx<mesh->ngx;jx++)

    for(int jy=0;jy<mesh->ngy;jy++)

      for(int jz=0;jz<mesh->ngz;jz++)

      {

        if(f[jx][jy][jz] >= limit)

             result[jx][jy][jz] = f[jx][jy][jz];

         else

             result[jx][jy][jz] = limit;

      }

  mesh->communicate(result);

  return(result);

}

BoutReal floor(const BoutReal &var, const BoutReal &f)
{
  if (var < f)
    return f;
  return var;
}



const Field3D field_smaller(const Field3D &f, const BoutReal limit)

{

  Field3D result;

  result.allocate();

//  #pragma omp parallel for

  for(int jx=0;jx<mesh->ngx;jx++)

    for(int jy=0;jy<mesh->ngy;jy++)

      for(int jz=0;jz<mesh->ngz;jz++)

      {

        if(f[jx][jy][jz] <= limit)

             result[jx][jy][jz] = f[jx][jy][jz];

         else

             result[jx][jy][jz] = limit;

      }

  mesh->communicate(result);

  return(result);

}



/****************BOUNDARY FUNCTIONS*****************************/

// Sheath Boundary Conditions 

// Linearized

void SBC_Dirichlet_SWidth1(Field3D &var, const Field3D &value) //let the boundary equall to the value next to the boundary

{
  RangeIterator xrup = mesh->iterateBndryUpperY();

  //for(xrup->first(); !xrup->isDone(); xrup->next())
  for (; !xrup.isDone(); xrup++)
    {
       {
     int xind = xrup.ind;
          for(int jy=mesh->yend+1-1; jy<mesh->ngy; jy++)
            {
              for(int jz=0; jz<mesh->ngz; jz++) 
                {

                  var[xind][jy][jz] = value[xind][mesh->yend][jz];

                }
	    }
       }
    }
   RangeIterator xrdn = mesh->iterateBndryLowerY();

  //for(xrdn->first(); !xrdn->isDone(); xrdn->next())
  for (; !xrdn.isDone(); xrdn++)
    {
     int xind = xrdn.ind;
       for(int jy=mesh->ystart-1+1; jy>=0; jy--)

            for(int jz=0; jz<mesh->ngz; jz++) 

                {

                  var[xind][jy][jz] = value[xind][mesh->ystart][jz];

                 }
       
    }
}
void SBC_Dirichlet(Field3D &var, const Field3D &value) //let the boundary equall to the value next to the boundary

{

  SBC_yup_eq(var, value);    

  SBC_ydown_eq(var, -value);   // Fluxes go towards X point and hit on plates, thus SBC at ydown or theta=0 should be negative 

}

 

void SBC_Gradpar(Field3D &var, const Field3D &value)

{

  SBC_yup_Grad_par(var, value);  

  SBC_ydown_Grad_par(var, -value); // Fluxes go towards X point and hit on plates, thus SBC at ydown or theta=0 should be different

}

 

// Boundary to specified Field3D object

void SBC_yup_eq(Field3D &var, const Field3D &value)

{
  RangeIterator xrup = mesh->iterateBndryUpperY();

  //for(xrup->first(); !xrup->isDone(); xrup->next())
  for (; !xrup.isDone(); xrup++)
    {
     // BoutReal x_glb= mesh->GlobalX(xrup->ind); 

      //if(x_glb>=Sheath_BC_x0) 
      int  xind = xrup.ind;
       {
          for(int jy=mesh->yend+1-Sheath_width; jy<mesh->ngy; jy++)
            {
              for(int jz=0; jz<mesh->ngz; jz++) 
                {

                  var[xind][jy][jz] = value[xind][mesh->yend][jz];

                }
	    }
       }
    }

}
 

void SBC_ydown_eq(Field3D &var, const Field3D &value)

{

   RangeIterator xrdn = mesh->iterateBndryLowerY();

  //for(xrdn->first(); !xrdn->isDone(); xrdn->next())
  for (; !xrdn.isDone(); xrdn++)
    {
     int xind = xrdn.ind;

      //BoutReal x_glb= mesh->GlobalX(xrdn->ind); 

      //if(x_glb>=Sheath_BC_x0) 

       {
         for(int jy=mesh->ystart-1+Sheath_width; jy>=0; jy--)

            for(int jz=0; jz<mesh->ngz; jz++) 

                {

                  var[xind][jy][jz] = value[xind][mesh->ystart][jz];

                 }
       }
    }
}      

 

// Boundary gradient to specified Field3D object

void SBC_yup_Grad_par(Field3D &var, const Field3D &value)

{
   RangeIterator xrup = mesh->iterateBndryUpperY();

  //for(xrup->first(); !xrup->isDone(); xrup->next())
  for (; !xrup.isDone(); xrup++)

    {
     int xind = xrup.ind;
      //BoutReal x_glb= mesh->GlobalX(xrup->ind); 

      //if(x_glb>=Sheath_BC_x0) 

        {
          for(int jy=mesh->yend+1-Sheath_width; jy<mesh->ngy; jy++)

             for(int jz=0; jz<mesh->ngz; jz++) 

                {

                  var[xind][jy][jz] = var[xind][jy-1][jz] + mesh->dy[xind][jy]*sqrt(mesh->g_22[xind][jy])*value[xind][jy][jz];
 
                }
        }
    }

}

 

void SBC_ydown_Grad_par(Field3D &var, const Field3D &value)

{
  RangeIterator xrdn = mesh->iterateBndryLowerY();
  //for(xrdn->first(); !xrdn->isDone(); xrdn->next())
  for (; !xrdn.isDone(); xrdn++)

    {
     int xind = xrdn.ind;
      //BoutReal x_glb= mesh->GlobalX(xrdn->ind); 

      //if(x_glb>=Sheath_BC_x0) 

        {
           for(int jy=mesh->ystart-1+Sheath_width; jy>=0; jy--)

              for(int jz=0; jz<mesh->ngz; jz++) 

                {

		   var[xind][jy][jz] = var[xind][jy+1][jz] - mesh->dy[xind][jy]*sqrt(mesh->g_22[xind][jy])*value[xind][jy][jz];

                }
	}
    }

}      

void WallBC_Xout_GradX(Field3D &var, const Field3D &value)
{
  // NB: input value of Gradient X in real R space 
  for(int jx=0;jx<mesh->ngx;jx++) 
    {
     if ( mesh->XGLOBAL (jx) > NX - 3 ) 
       {
         for(int jy=0;jy<mesh->ngy;jy++) 
             for(int jz=0;jz<mesh->ngz;jz++) 
                { 
                  var[jx][jy][jz] = var[jx-1][jy][jz] + value[jx][jy][jz]*mesh->dx[jx][jy]/sqrt(mesh->g11[jx][jy]);  // calculated in BOUT psi coordinate
                 }
         }
     }
}

void WallBC_Xout_GradX_len(Field3D &var, BoutReal value)
{
  // NB: input value of 'Gradient_X(var)/(var)' or '1/Gradient X length with sign of Gradient_X' in real R space 
  BoutReal temp;
   for(int jx=0;jx<mesh->ngx;jx++) 
    {
     if ( mesh->XGLOBAL (jx) > NX - 3 ) 
       {
         for(int jy=0;jy<mesh->ngy;jy++) 
             for(int jz=0;jz<mesh->ngz;jz++) 
                { 
                  temp=0.5*value*mesh->dx[jx][jy]/sqrt(mesh->g11[jx][jy]);     // transfer to BOUT psi coordinate
                  var[jx][jy][jz] = var[jx-1][jy][jz]*(1.+temp)/(1.-temp);  
                 }
         }
     }
}


const Field3D BS_ft(const int index)
{
  Field3D result, result1;
  result.allocate();
  result1.allocate();
  result1=0.;
  
  BoutReal xlam, dxlam;
  //dxlam = 1./bmag/index;     // wrong since normalization of bmag is needed
  //dxlam = 1./index;          // right only when global max(Bxy)=bmag=Bbar
  //dxlam = 1./max(mesh->Bxy)/index;       // It is still not perfect since max(mesh->Bxy) is the maximum value at some core in parallel computation
  //output.write("maxmum normalized Bxy %e \n",max(mesh->Bxy));
  if(max(mesh->Bxy) > 1.0) dxlam = 1./max(mesh->Bxy)/index;
  else dxlam = 1./index; 

  xlam = 0.;

  for(int i=0; i<index; i++)
    {
      result1 += xlam*dxlam/sqrt(1.-xlam*mesh->Bxy);
      xlam += dxlam;
    }
  result = 1.- 0.75*mesh->Bxy*mesh->Bxy * result1;

  return(result);
}

const Field3D F31(const Field3D input)
{
  Field3D result;
  result.allocate();

  result = ( 1 + 1.4/(Zi+1.) ) * input;
  result -= 1.9/(Zi+1.) * input*input;
  result += 0.3/(Zi+1.) * input*input*input;
  result += 0.2/(Zi+1.) * input*input*input*input;

  return(result);
}

const Field3D F32ee(const Field3D input)
{
  Field3D result;
  result.allocate();
  
  result = (0.05+0.62*Zi)/(Zi*(1+0.44*Zi))*(input-input*input*input*input);
  result +=1./(1.+0.22*Zi)*( input*input-input*input*input*input-1.2*(input*input*input-input*input*input*input) );
  result += 1.2/(1.+0.5*Zi)*input*input*input*input;

  return(result);
}

const Field3D F32ei(const Field3D input)
{
  Field3D result;
  result.allocate();
  
  result = -(0.56+1.93*Zi)/(Zi*(1+0.44*Zi))*(input-input*input*input*input);
  result += 4.95/(1.+2.48*Zi)*( input*input-input*input*input*input-0.55*(input*input*input-input*input*input*input) );
  result -= 1.2/(1.+0.5*Zi)*input*input*input*input;

  return(result);
}










