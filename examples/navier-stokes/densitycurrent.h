// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

/// @file
/// Density current initial condition and operator for Navier-Stokes example using PETSc

// Model from:
//   Semi-Implicit Formulations of the Navier-Stokes Equations: Application to
//   Nonhydrostatic Atmospheric Modeling, Giraldo, Restelli, and Lauter (2010).

#ifndef densitycurrent_h
#define densitycurrent_h

#include <math.h>

// *****************************************************************************
// This QFunction sets the the initial conditions and boundary conditions
//
// These initial conditions are given in terms of potential temperature and
//   Exner pressure and then converted to density and total energy.
//   Initial momentum density is zero.
//
// Initial Conditions:
//   Potential Temperature:
//     theta = thetabar + deltatheta
//       thetabar   = theta0 exp( N**2 z / g )
//       deltatheta = r <= rc : theta0(1 + cos(pi r/rc)) / 2
//                     r > rc : 0
//         r        = sqrt( (x - xc)**2 + (y - yc)**2 + (z - zc)**2 )
//         with (xc,yc,zc) center of domain, rc characteristic radius of thermal bubble
//   Exner Pressure:
//     Pi = Pibar + deltaPi
//       Pibar      = g**2 (exp( - N**2 z / g ) - 1) / (cp theta0 N**2)
//       deltaPi    = 0 (hydrostatic balance)
//   Velocity/Momentum Density:
//     Ui = ui = 0
//
// Conversion to Conserved Variables:
//   rho = P0 Pi**(cv/Rd) / (Rd theta)
//   E   = rho (cv theta Pi + (u u)/2)
//
//  Boundary Conditions:
//    Mass Density:
//      0.0 flux
//    Momentum Density:
//      0.0
//    Energy Density:
//      0.0 flux
//
// Constants:
//   theta0          ,  Potential temperature constant
//   thetaC          ,  Potential temperature perturbation
//   P0              ,  Pressure at the surface
//   N               ,  Brunt-Vaisala frequency
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   Rd     = cp - cv,  Specific heat difference
//   g               ,  Gravity
//   rc              ,  Characteristic radius of thermal bubble
//   lx              ,  Characteristic length scale of domain in x
//   ly              ,  Characteristic length scale of domain in y
//   lz              ,  Characteristic length scale of domain in z
//
// *****************************************************************************
CEED_QFUNCTION(ICsDC)(void *ctx, CeedInt Q,
                      const CeedScalar *const *in, CeedScalar *const *out) {

#ifndef M_PI
#define M_PI    3.14159265358979323846
#endif

  // Inputs
  const CeedScalar (*X)[Q] = (CeedScalar(*)[Q])in[0];
  // Outputs
  CeedScalar (*q0)[Q] = (CeedScalar(*)[Q])out[0],
             (*coords)[Q] = (CeedScalar(*)[Q])out[1];
  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
  const CeedScalar theta0 = context[0];
  const CeedScalar thetaC = context[1];
  const CeedScalar P0     = context[2];
  const CeedScalar N      = context[3];
  const CeedScalar cv     = context[4];
  const CeedScalar cp     = context[5];
  const CeedScalar Rd     = context[6];
  const CeedScalar g      = context[7];
  const CeedScalar rc     = context[8];
  const CeedScalar lx     = context[9];
  const CeedScalar ly     = context[10];
  const CeedScalar lz     = context[11];
  const CeedScalar *periodic = &context[12];
  // Setup
  const CeedScalar tol = 1.e-14;
  const CeedScalar center[3] = {0.5*lx, 0.5*ly, 0.5*lz};

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Coordinates
    const CeedScalar x = X[0][i];
    const CeedScalar y = X[1][i];
    const CeedScalar z = X[2][i];
    // -- Potential temperature, density current
    const CeedScalar r = sqrt(pow((x - center[0]), 2) +
                              pow((y - center[1]), 2) +
                              pow((z - center[2]), 2));
    const CeedScalar deltatheta = r <= rc ? thetaC*(1. + cos(M_PI*r/rc))/2. : 0.;
    const CeedScalar theta = theta0*exp(N*N*z/g) + deltatheta;
    // -- Exner pressure, hydrostatic balance
    const CeedScalar Pi = 1. + g*g*(exp(-N*N*z/g) - 1.) / (cp*theta0*N*N);
    // -- Density
    const CeedScalar rho = P0 * pow(Pi, cv/Rd) / (Rd*theta);

    // Initial Conditions
    q0[0][i] = rho;
    q0[1][i] = 0.0;
    q0[2][i] = 0.0;
    q0[3][i] = 0.0;
    q0[4][i] = rho * (cv*theta*Pi);

    // Homogeneous Dirichlet Boundary Conditions for Momentum
    if ((!periodic[0] && (fabs(x - 0.0) < tol || fabs(x - lx) < tol))
        || (!periodic[1] && (fabs(y - 0.0) < tol || fabs(y - ly) < tol))
        || (!periodic[2] && (fabs(z - 0.0) < tol || fabs(z - lz) < tol))) {
      q0[1][i] = 0.0;
      q0[2][i] = 0.0;
      q0[3][i] = 0.0;
    }

    // Coordinates
    coords[0][i] = x;
    coords[1][i] = y;
    coords[2][i] = z;

  } // End of Quadrature Point Loop

  // Return
  return 0;
}

// *******************************************************************************
// This QFunction implements the following formulation of Navier-Stokes
//
// This is 3D compressible Navier-Stokes in conservation form with state
//   variables of density, momentum density, and total energy density.
//
// State Variables: q = ( rho, U1, U2, U3, E )
//   rho - Mass Density
//   Ui  - Momentum Density   ,  Ui = rho ui
//   E   - Total Energy Density,  E  = rho cv T + rho (u u) / 2
//
// Navier-Stokes Equations:
//   drho/dt + div( U )                               = 0
//   dU/dt   + div( rho (u x u) + P I3 ) + rho g khat = div( Fu )
//   dE/dt   + div( (E + P) u )          + rho g u[z] = div( Fe )
//
// Viscous Stress:
//   Fu = mu (grad( u ) + grad( u )^T + lambda div ( u ) I3)
// Thermal Stress:
//   Fe = u Fu + k grad( T )
//
// Equation of State:
//   P = (gamma - 1) (E - rho (u u) / 2)
//
// Temperature:
//   T = (E / rho - (u u) / 2 ) / cv
//
//********************************************
// Stabilization:
//
//   Tau = [TauC, TauM, TauM, TauM, TauE]
//      f1 = rho  sqrt(2 / (C1  dt) + ui uj gij + C2 mu^2 gij gij)   
//           gij = dXi/dX * dXi/dX 
// TauC = Cc f1 / (8 gii)                          
// TauM = 1 / f1
// TauE = TauM / (Ce cv)                          
//
//  SU   = Galerkin + grad(v) . ( Ai^T * Tau * (Aj q,j) )
//  SUPG = Galerkin + grad(v) . ( Ai^T * Tau * (qdot + Aj q,j - body force) )   
//                                       (diffussive terms will be added later)
// Constants:
//   lambda = - 2 / 3,  From Stokes hypothesis
//   mu              ,  Dynamic viscosity
//   k               ,  Thermal conductivity
//   cv              ,  Specific heat, constant volume
//   cp              ,  Specific heat, constant pressure
//   g               ,  Gravity
//   gamma  = cp / cv,  Specific heat ratio
//
// We require the product of the inverse of the Jacobian (dXdx_j,k) and
// its transpose (dXdx_k,j) to properly compute integrals of the form:
// int( gradv gradu )
//
// *******************************************************************************
CEED_QFUNCTION(DC)(void *ctx, CeedInt Q,
                   const CeedScalar *const *in, CeedScalar *const *out) {
  // Inputs
  const CeedScalar (*q)[Q] = (CeedScalar(*)[Q])in[0],
                   (*dq)[5][Q] = (CeedScalar(*)[5][Q])in[1],
                   (*qdata)[Q] = (CeedScalar(*)[Q])in[2],
                   (*x)[Q] = (CeedScalar(*)[Q])in[3];
  // Outputs
  CeedScalar (*v)[Q] = (CeedScalar(*)[Q])out[0],
             (*dv)[5][Q] = (CeedScalar(*)[5][Q])out[1];
  // Context
  const CeedScalar *context = (const CeedScalar *)ctx;
  const CeedScalar lambda = context[0];
  const CeedScalar mu     = context[1];
  const CeedScalar k      = context[2];
  const CeedScalar cv     = context[3];
  const CeedScalar cp     = context[4];
  const CeedScalar g      = context[5];
  const CeedScalar Rd     = context[6];
  const CeedScalar gamma  = cp / cv;

  CeedPragmaSIMD
  // Quadrature Point Loop
  for (CeedInt i=0; i<Q; i++) {
    // Setup
    // -- Interp in
    const CeedScalar rho        =    q[0][i];
    const CeedScalar u[3]       =   {q[1][i] / rho,
                                     q[2][i] / rho,
                                     q[3][i] / rho
                                    };
    const CeedScalar E          =    q[4][i];
    // -- Grad in
    const CeedScalar drho[3]    =   {dq[0][0][i],
                                     dq[1][0][i],
                                     dq[2][0][i]
                                    };
    const CeedScalar dU[3][3]   = {{dq[0][1][i],
                                    dq[1][1][i],
                                    dq[2][1][i]},
                                   {dq[0][2][i],
                                    dq[1][2][i],
                                    dq[2][2][i]},
                                   {dq[0][3][i],
                                    dq[1][3][i],
                                    dq[2][3][i]}
                                  };                                
    const CeedScalar dE[3]      =   {dq[0][4][i],
                                     dq[1][4][i],
                                     dq[2][4][i]
                                    };
    // -- Interp-to-Interp qdata
    const CeedScalar wJ         =    qdata[0][i];
    // -- Interp-to-Grad qdata
    // ---- Inverse of change of coordinate matrix: X_i,j
    const CeedScalar dXdx[3][3] =  {{qdata[1][i],
                                     qdata[2][i],
                                     qdata[3][i]},
                                    {qdata[4][i],
                                     qdata[5][i],
                                     qdata[6][i]},
                                    {qdata[7][i],
                                     qdata[8][i],
                                     qdata[9][i]}
                                   };
    // -- Grad-to-Grad qdata
    // dU/dx
    CeedScalar du[3][3];
    CeedScalar drhodx[3];
    CeedScalar dEdx[3];
    CeedScalar dUdx[3][3];
    CeedScalar dudx[3][3];
    CeedScalar dXdxdXdxT[3][3];
    for (int j=0; j<3; j++) {
      drhodx[j] =0;
      dEdx[j] = 0;
      for (int k=0; k<3; k++) {
        du[j][k] = (dU[j][k] - drho[k]*u[j]) / rho;
        drhodx[j] += drho[k] * dXdx[k][j];
        dEdx[j] += dE[k] * dXdx[k][j];
        dUdx[j][k] = 0;
        dudx[j][k] = 0;
        dXdxdXdxT[j][k] = 0;
        for (int l=0; l<3; l++) {
          dUdx[j][k] += dU[j][l] * dXdx[l][k];
          dudx[j][k] += du[j][l] * dXdx[l][k];
          dXdxdXdxT[j][k] += dXdx[j][l]*dXdx[k][l];  //dXdx_j,k * dXdx_k,j
      }}}
      

   // -- gradT
    const CeedScalar gradT[3]  = {(dEdx[0]/rho - E*drhodx[0]/(rho*rho) -
                                  (u[0]*dudx[0][0] + u[1]*dudx[1][0] + u[2]*dudx[2][0]))/cv,
                                  (dEdx[1]/rho - E*drhodx[1]/(rho*rho) -
                                  (u[0]*dudx[0][1] + u[1]*dudx[1][1] + u[2]*dudx[2][1]))/cv,
                                  (dEdx[2]/rho - E*drhodx[2]/(rho*rho) -
                                  (u[0]*dudx[0][2] + u[1]*dudx[1][2] + u[2]*dudx[2][2]))/cv
                                 };
    // -- Fuvisc
    // ---- Symmetric 3x3 matrix
    const CeedScalar Fu[6]     =  { mu *(dudx[0][0] * (2 + lambda) +
                                         lambda * (dudx[1][1] + dudx[2][2])),
                                    mu *(dudx[0][1] + dudx[1][0]),
                                    mu *(dudx[0][2] + dudx[2][0]),
                                    mu *(dudx[1][1] * (2 + lambda) +
                                         lambda * (dudx[0][0] + dudx[2][2])),
                                    mu *(dudx[1][2] + dudx[2][1]),
                                    mu *(dudx[2][2] * (2 + lambda) +
                                         lambda * (dudx[0][0] + dudx[1][1]))
                                  };
    // -- Fevisc
    const CeedScalar Fe[3]     =  { u[0]*Fu[0] + u[1]*Fu[1] + u[2]*Fu[2] +
                                    k *gradT[0],
                                    u[0]*Fu[1] + u[1]*Fu[3] + u[2]*Fu[4] +
                                    k *gradT[1],
                                    u[0]*Fu[2] + u[1]*Fu[4] + u[2]*Fu[5] +
                                    k *gradT[2]
                                  };
    // ke = kinetic energy
    const CeedScalar ke = ( u[0]*u[0] + u[1]*u[1] + u[2]*u[2] ) / 2.;
    // P = pressure
    const CeedScalar P  =  ( E - ke * rho ) * (gamma - 1.);

    // Tau = [TauC, TauM, TauM, TauM, TauE]
    CeedScalar uX[3];
    for (int j=0; j<3; j++) uX[j] = dXdx[j][0]*u[0] + dXdx[j][1]*u[1] + dXdx[j][2]*u[2];
    const CeedScalar uiujgij = uX[0]*uX[0] + uX[1]*uX[1] + uX[2]*uX[2];
    //const CeedScalar gijgij =     //find later
    const CeedScalar C1   = 1.;
    const CeedScalar C2   = 1.;
    const CeedScalar Cc   = 1.;
    const CeedScalar Ce   = 1.; 
    const CeedScalar f1   = rho * sqrt(uiujgij);   //This is correct for SU
    const CeedScalar TauC = (Cc * f1) / ( 8 * (dXdxdXdxT[0][0] + dXdxdXdxT[1][1] + dXdxdXdxT[2][2]));      
    const CeedScalar TauM = 1./f1;
    const CeedScalar TauE = TauM / (Ce * cv);

        //- Stabilizing terms
    //----- Convection Ai^T * Tau * Aj * q,j
    const CeedScalar Stab_Conv[5][3] ={{TauM*(u[0]*u[0] - (Rd*ke)/cv)*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] -
                                     dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] +
                                     drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv) + 
                                     TauM*u[0]*u[1]*(drho[1]*(u[1]*u[1] - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - dU[2][2]*u[1] -
                                     dU[0][0]*u[1] + dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv +
                                     (Rd*dU[0][1]*u[0])/cv + (Rd*dU[2][1]*u[2])/cv) + TauM*u[0]*u[2]*(drho[2]*(u[2]*u[2]  - 
                                     (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - dU[2][1]*u[1] - dU[0][0]*u[2] + dU[2][2]*u[2]*(Rd/cv - 2) + 
                                     drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + (Rd*dU[1][2]*u[1])/cv) + 
                                     TauE*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv)*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[2]*u[2] + ke))/cv) - dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) -
                                     dU[0][0]*(E*(Rd/cv + 1) - (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) +
                                     drho[2]*u[2]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv +
                                     (Rd*dU[1][0]*u[0]*u[1])/cv + (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + 
                                     (Rd*dU[2][1]*u[1]*u[2])/cv),
                                  
                                     TauM*(u[1]*u[1] - (Rd*ke)/cv)*(drho[1]*(u[1]*u[1] - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] -
                                     dU[2][2]*u[1] - dU[0][0]*u[1] + dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] -
                                     (Rd*dE[1])/cv + (Rd*dU[0][1]*u[0])/cv + (Rd*dU[2][1]*u[2])/cv) + TauM*u[0]*u[1]*(drho[0]*(u[0]*u[0] -
                                     (Rd*ke)/cv) - dU[0][2]*u[2] - dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) +
                                     drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv) +
                                     TauM*u[1]*u[2]*(drho[2]*(u[2]*u[2]  - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - dU[2][1]*u[1] -
                                     dU[0][0]*u[2] + dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + 
                                     (Rd*dU[0][2]*u[0])/cv + (Rd*dU[1][2]*u[1])/cv) + TauE*u[1]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv)*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - dU[1][1]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[1]*u[1] + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - dE[0]*u[0]*(Rd/cv + 1) -
                                     dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - (Rd*(u[0]*u[0] + ke))/cv) +
                                     drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + 
                                     (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv),
                                     
                                     TauM*(u[2]*u[2] - (Rd*ke)/cv)*(drho[2]*(u[2]*u[2] - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - 
                                     dU[2][1]*u[1] - dU[0][0]*u[2] + dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - 
                                     (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + (Rd*dU[1][2]*u[1])/cv) + TauM*u[0]*u[2]*(drho[0]*(u[0]*u[0] -
                                     (Rd*ke)/cv) - dU[0][2]*u[2] - dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) +
                                     drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv) + 
                                     TauM*u[1]*u[2]*(drho[1]*(u[1]*u[1]  - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - dU[2][2]*u[1] - 
                                     dU[0][0]*u[1] + dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + 
                                     (Rd*dU[0][1]*u[0])/cv + (Rd*dU[2][1]*u[2])/cv) + TauE*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv)*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - dU[1][1]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[1]*u[1] + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv)}, 
                                   
                                    {TauC*(dU[0][0] + dU[1][1] + dU[2][2]) - TauM*u[1]*(drho[1]*(u[1]*u[1]  - (Rd*ke)/cv) - dU[1][0]*u[0] - 
                                     dU[1][2]*u[2] - dU[2][2]*u[1] - dU[0][0]*u[1] + dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + 
                                     drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + (Rd*dU[0][1]*u[0])/cv + (Rd*dU[2][1]*u[2])/cv) - 
                                     TauM*u[2]*(drho[2]*(u[2]*u[2]  - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - dU[2][1]*u[1] - dU[0][0]*u[2] + 
                                     dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + 
                                     (Rd*dU[1][2]*u[1])/cv) - TauE*(E*(Rd/cv + 1) - (Rd*(u[0]*u[0] + ke))/cv)*(drho[0]*u[0]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) - dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[2]*u[2]  + ke))/cv) - dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - 
                                     dU[0][0]*(E*(Rd/cv + 1) - (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + 
                                     drho[2]*u[2]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + 
                                     (Rd*dU[1][0]*u[0]*u[1])/cv + (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + 
                                     (Rd*dU[2][1]*u[1]*u[2])/cv) + TauM*u[0]*(Rd/cv - 2)*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - 
                                     dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + 
                                     drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv),
                                    
                                     (Rd*TauM*u[0]*(drho[1]*(u[1]*u[1] - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - dU[2][2]*u[1] - dU[0][0]*u[1] + 
                                     dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + (Rd*dU[0][1]*u[0])/cv + 
                                     (Rd*dU[2][1]*u[2])/cv))/cv - TauM*u[1]*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - dU[1][1]*u[0] - 
                                     dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - 
                                     (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv) + (Rd*TauE*u[0]*u[1]*(drho[0]*u[0]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) - dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[2]*u[2]  + ke))/cv) - dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - 
                                     dU[0][0]*(E*(Rd/cv + 1) - (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + 
                                     drho[2]*u[2]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + 
                                     (Rd*dU[1][0]*u[0]*u[1])/cv + (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + 
                                     (Rd*dU[2][1]*u[1]*u[2])/cv))/cv,
                                    
                                     (Rd*TauM*u[0]*(drho[2]*(u[2]*u[2] - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - dU[2][1]*u[1] - dU[0][0]*u[2] + 
                                     dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + 
                                     (Rd*dU[1][2]*u[1])/cv))/cv - TauM*u[2]*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - dU[1][1]*u[0] - 
                                     dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + 
                                     (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv) + (Rd*TauE*u[0]*u[2]*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv))/cv},
                                   
                                    {(Rd*TauM*u[1]*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + 
                                     dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + 
                                     (Rd*dU[2][0]*u[2])/cv))/cv - TauM*u[0]*(drho[1]*(u[1]*u[1]  - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - 
                                     dU[2][2]*u[1] - dU[0][0]*u[1] + dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + 
                                     (Rd*dU[0][1]*u[0])/cv + (Rd*dU[2][1]*u[2])/cv) + (Rd*TauE*u[0]*u[1]*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv))/cv,
                                   
                                     TauC*(dU[0][0] + dU[1][1] + dU[2][2]) - TauM*u[0]*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - 
                                     dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - 
                                     (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv) - TauM*u[2]*(drho[2]*(u[2]*u[2] - (Rd*ke)/cv) - 
                                     dU[1][1]*u[2] - dU[2][0]*u[0] - dU[2][1]*u[1] - dU[0][0]*u[2] + dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + 
                                     drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + (Rd*dU[1][2]*u[1])/cv) - TauE*(E*(Rd/cv + 1) - 
                                     (Rd*(u[1]*u[1]  + ke))/cv)*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - dU[1][1]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - dE[0]*u[0]*(Rd/cv + 1) - 
                                     dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - (Rd*(u[0]*u[0] + ke))/cv) + 
                                     drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + 
                                     (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + (Rd*dU[1][2]*u[1]*u[2])/cv + 
                                     (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv) + TauM*u[1]*(Rd/cv - 2)*(drho[1]*(u[1]*u[1]  - (Rd*ke)/cv) - 
                                     dU[1][0]*u[0] - dU[1][2]*u[2] - dU[2][2]*u[1] - dU[0][0]*u[1] + dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + 
                                     drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + (Rd*dU[0][1]*u[0])/cv + (Rd*dU[2][1]*u[2])/cv),
                                    
                                     (Rd*TauM*u[1]*(drho[2]*(u[2]*u[2] - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - dU[2][1]*u[1] - dU[0][0]*u[2] + 
                                     dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + 
                                     (Rd*dU[1][2]*u[1])/cv))/cv - TauM*u[2]*(drho[1]*(u[1]*u[1]  - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - 
                                     dU[2][2]*u[1] - dU[0][0]*u[1] + dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + 
                                     (Rd*dU[0][1]*u[0])/cv + (Rd*dU[2][1]*u[2])/cv) + (Rd*TauE*u[1]*u[2]*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv))/cv}, 
                                   
                                    {(Rd*TauM*u[2]*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + 
                                     dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + 
                                     (Rd*dU[2][0]*u[2])/cv))/cv - TauM*u[0]*(drho[2]*(u[2]*u[2]  - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - 
                                     dU[2][1]*u[1] - dU[0][0]*u[2] + dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + 
                                     (Rd*dU[0][2]*u[0])/cv + (Rd*dU[1][2]*u[1])/cv) + (Rd*TauE*u[0]*u[2]*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv))/cv,
                                   
                                     (Rd*TauM*u[2]*(drho[1]*(u[1]*u[1] - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - dU[2][2]*u[1] - dU[0][0]*u[1] + 
                                     dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + (Rd*dU[0][1]*u[0])/cv + 
                                     (Rd*dU[2][1]*u[2])/cv))/cv - TauM*u[1]*(drho[2]*(u[2]*u[2]  - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - 
                                     dU[2][1]*u[1] - dU[0][0]*u[2] + dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + 
                                     (Rd*dU[0][2]*u[0])/cv + (Rd*dU[1][2]*u[1])/cv) + (Rd*TauE*u[1]*u[2]*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv))/cv,
                                   
                                     TauC*(dU[0][0] + dU[1][1] + dU[2][2]) - TauM*u[0]*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - 
                                     dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + 
                                     drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + (Rd*dU[2][0]*u[2])/cv) - 
                                     TauM*u[1]*(drho[1]*(u[1]*u[1]  - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - dU[2][2]*u[1] - dU[0][0]*u[1] + 
                                     dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + (Rd*dU[0][1]*u[0])/cv + 
                                     (Rd*dU[2][1]*u[2])/cv) - TauE*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv)*(drho[0]*u[0]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) - dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[2]*u[2]  + ke))/cv) - dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - 
                                     dU[0][0]*(E*(Rd/cv + 1) - (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + 
                                     drho[2]*u[2]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + 
                                     (Rd*dU[1][0]*u[0]*u[1])/cv + (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + 
                                     (Rd*dU[2][1]*u[1]*u[2])/cv) + TauM*u[2]*(Rd/cv - 2)*(drho[2]*(u[2]*u[2]  - (Rd*ke)/cv) - dU[1][1]*u[2] - 
                                     dU[2][0]*u[0] - dU[2][1]*u[1] - dU[0][0]*u[2] + dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + 
                                     drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + (Rd*dU[1][2]*u[1])/cv)}, 
                                   
                                    {-(Rd*TauM*(drho[0]*(u[0]*u[0] - (Rd*ke)/cv) - dU[0][2]*u[2] - dU[1][1]*u[0] - dU[2][2]*u[0] - dU[0][1]*u[1] + 
                                     dU[0][0]*u[0]*(Rd/cv - 2) + drho[1]*u[0]*u[1] + drho[2]*u[0]*u[2] - (Rd*dE[0])/cv + (Rd*dU[1][0]*u[1])/cv + 
                                     (Rd*dU[2][0]*u[2])/cv))/cv - TauE*u[0]*(Rd/cv + 1)*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv),
                                   
                                     -(Rd*TauM*(drho[1]*(u[1]*u[1] - (Rd*ke)/cv) - dU[1][0]*u[0] - dU[1][2]*u[2] - dU[2][2]*u[1] - dU[0][0]*u[1] + 
                                     dU[1][1]*u[1]*(Rd/cv - 2) + drho[0]*u[0]*u[1] + drho[2]*u[1]*u[2] - (Rd*dE[1])/cv + (Rd*dU[0][1]*u[0])/cv + 
                                     (Rd*dU[2][1]*u[2])/cv))/cv - TauE*u[1]*(Rd/cv + 1)*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv),
                                   
                                     -(Rd*TauM*(drho[2]*(u[2]*u[2] - (Rd*ke)/cv) - dU[1][1]*u[2] - dU[2][0]*u[0] - dU[2][1]*u[1] - dU[0][0]*u[2] + 
                                     dU[2][2]*u[2]*(Rd/cv - 2) + drho[0]*u[0]*u[2] + drho[1]*u[1]*u[2] - (Rd*dE[2])/cv + (Rd*dU[0][2]*u[0])/cv + 
                                     (Rd*dU[1][2]*u[1])/cv))/cv - TauE*u[2]*(Rd/cv + 1)*(drho[0]*u[0]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) - 
                                     dU[1][1]*(E*(Rd/cv + 1) - (Rd*(u[1]*u[1]  + ke))/cv) - dU[2][2]*(E*(Rd/cv + 1) - (Rd*(u[2]*u[2]  + ke))/cv) - 
                                     dE[0]*u[0]*(Rd/cv + 1) - dE[1]*u[1]*(Rd/cv + 1) - dE[2]*u[2]*(Rd/cv + 1) - dU[0][0]*(E*(Rd/cv + 1) - 
                                     (Rd*(u[0]*u[0] + ke))/cv) + drho[1]*u[1]*(E*(Rd/cv + 1) - (2*Rd*ke)/cv) + drho[2]*u[2]*(E*(Rd/cv + 1) - 
                                     (2*Rd*ke)/cv) + (Rd*dU[0][1]*u[0]*u[1])/cv + (Rd*dU[0][2]*u[0]*u[2])/cv + (Rd*dU[1][0]*u[0]*u[1])/cv + 
                                     (Rd*dU[1][2]*u[1]*u[2])/cv + (Rd*dU[2][0]*u[0]*u[2])/cv + (Rd*dU[2][1]*u[1]*u[2])/cv)}
                                  };

    // The Physics

    // -- Density
    // ---- u rho
    for (int j=0; j<3; j++)
      dv[j][0][i]  = wJ*(rho*u[0]*dXdx[j][0] + rho*u[1]*dXdx[j][1] +
                         rho*u[2]*dXdx[j][2]);
    // ---- No Change
    v[0][i] = 0;

    // -- Momentum
    // ---- rho (u x u) + P I3
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i]  = wJ*((rho*u[j]*u[0] + (j==0?P:0))*dXdx[k][0] +
                             (rho*u[j]*u[1] + (j==1?P:0))*dXdx[k][1] +
                             (rho*u[j]*u[2] + (j==2?P:0))*dXdx[k][2]);
    // ---- Fuvisc
    const CeedInt Fuviscidx[3][3] = {{0, 1, 2}, {1, 3, 4}, {2, 4, 5}}; // symmetric matrix indices
    for (int j=0; j<3; j++)
      for (int k=0; k<3; k++)
        dv[k][j+1][i] -= wJ*(Fu[Fuviscidx[j][0]]*dXdx[k][0] +
                             Fu[Fuviscidx[j][1]]*dXdx[k][1] +
                             Fu[Fuviscidx[j][2]]*dXdx[k][2]);
    // ---- -rho g khat
    v[1][i] = 0;
    v[2][i] = 0;
    v[3][i] = -rho*g*wJ;

    // -- Total Energy Density
    // ---- (E + P) u
    for (int j=0; j<3; j++)
      dv[j][4][i]  = wJ * (E + P) * (u[0]*dXdx[j][0] + u[1]*dXdx[j][1] +
                                     u[2]*dXdx[j][2]);
    // ---- Fevisc
    for (int j=0; j<3; j++)
      dv[j][4][i] -= wJ * (Fe[0]*dXdx[j][0] + Fe[1]*dXdx[j][1] +
                           Fe[2]*dXdx[j][2]);
    // ---- No Change
    v[4][i] = -rho*g*u[2]*wJ;

  } // End Quadrature Point Loop

  // Return
  return 0;
}

// *****************************************************************************
#endif
