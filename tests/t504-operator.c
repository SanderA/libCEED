/// @file
/// Test creation reuse of the same QFunction for multiple operators
/// \test Test creation reuse of the same QFunction for multiple operators
#include <ceed.h>
#include <stdlib.h>
#include <math.h>
#include "t502-operator.h"

int main(int argc, char **argv) {
  Ceed ceed;
  CeedElemRestriction Erestrictx, Erestrictu, Erestrictxi,
                      Erestrictui_small, Erestrictui_large;
  CeedBasis bx_small, bx_large, bu_small, bu_large;
  CeedQFunction qf_setup, qf_mass;
  CeedOperator op_setup_small, op_mass_small,
               op_setup_large, op_mass_large;
  CeedVector qdata_small, qdata_large, X, U, V;
  CeedScalar *hu;
  const CeedScalar *hv;
  CeedInt nelem = 15, P = 5, Q = 8, scale = 3;
  CeedInt Nx = nelem+1, Nu = nelem*(P-1)+1;
  CeedInt indx[nelem*2], indu[nelem*P];
  CeedScalar x[Nx];
  CeedScalar sum1, sum2;

  CeedInit(argv[1], &ceed);
  for (CeedInt i=0; i<Nx; i++) x[i] = (CeedScalar) i / (Nx - 1);
  for (CeedInt i=0; i<nelem; i++) {
    indx[2*i+0] = i;
    indx[2*i+1] = i+1;
  }
  // Restrictions
  CeedElemRestrictionCreate(ceed, nelem, 2, Nx, 1, CEED_MEM_HOST,
                            CEED_USE_POINTER, indx, &Erestrictx);
  CeedElemRestrictionCreateIdentity(ceed, nelem, 2, nelem*2, 1, &Erestrictxi);

  for (CeedInt i=0; i<nelem; i++) {
    for (CeedInt j=0; j<P; j++) {
      indu[P*i+j] = i*(P-1) + j;
    }
  }
  CeedElemRestrictionCreate(ceed, nelem, P, Nu, 2, CEED_MEM_HOST,
                            CEED_USE_POINTER, indu, &Erestrictu);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q, Q*nelem, 1,
                                    &Erestrictui_small);
  CeedElemRestrictionCreateIdentity(ceed, nelem, Q*scale, Q*nelem*scale, 1,
                                    &Erestrictui_large);

  // Bases
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q, CEED_GAUSS, &bx_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q, CEED_GAUSS, &bu_small);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 1, 2, Q*scale, CEED_GAUSS, &bx_large);
  CeedBasisCreateTensorH1Lagrange(ceed, 1, 2, P, Q*scale, CEED_GAUSS, &bu_large);

  // QFunctions
  CeedQFunctionCreateInterior(ceed, 1, setup, setup_loc, &qf_setup);
  CeedQFunctionAddInput(qf_setup, "_weight", 1, CEED_EVAL_WEIGHT);
  CeedQFunctionAddInput(qf_setup, "x", 1, CEED_EVAL_GRAD);
  CeedQFunctionAddOutput(qf_setup, "rho", 1, CEED_EVAL_NONE);

  CeedQFunctionCreateInterior(ceed, 1, mass, mass_loc, &qf_mass);
  CeedQFunctionAddInput(qf_mass, "rho", 1, CEED_EVAL_NONE);
  CeedQFunctionAddInput(qf_mass, "u", 2, CEED_EVAL_INTERP);
  CeedQFunctionAddOutput(qf_mass, "v", 2, CEED_EVAL_INTERP);

  // Input vector
  CeedVectorCreate(ceed, Nx, &X);
  CeedVectorSetArray(X, CEED_MEM_HOST, CEED_USE_POINTER, x);

  // 'Small' Operators
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup_small);
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass_small);

  CeedVectorCreate(ceed, nelem*Q, &qdata_small);

  CeedOperatorSetField(op_setup_small, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx_small, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_small, "x", Erestrictx, CEED_NOTRANSPOSE,
                       bx_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_small, "rho", Erestrictui_small, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass_small, "rho", Erestrictui_small, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata_small);
  CeedOperatorSetField(op_mass_small, "u", Erestrictu, CEED_TRANSPOSE,
                       bu_small, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_small, "v", Erestrictu, CEED_TRANSPOSE,
                       bu_small, CEED_VECTOR_ACTIVE);

  // 'Large' operators
  CeedOperatorCreate(ceed, qf_setup, NULL, NULL, &op_setup_large);
  CeedOperatorCreate(ceed, qf_mass, NULL, NULL, &op_mass_large);

  CeedVectorCreate(ceed, nelem*Q*scale, &qdata_large);

  CeedOperatorSetField(op_setup_large, "_weight", Erestrictxi, CEED_NOTRANSPOSE,
                       bx_large, CEED_VECTOR_NONE);
  CeedOperatorSetField(op_setup_large, "x", Erestrictx, CEED_NOTRANSPOSE,
                       bx_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_setup_large, "rho", Erestrictui_large, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, CEED_VECTOR_ACTIVE);

  CeedOperatorSetField(op_mass_large, "rho", Erestrictui_large, CEED_NOTRANSPOSE,
                       CEED_BASIS_COLLOCATED, qdata_large);
  CeedOperatorSetField(op_mass_large, "u", Erestrictu, CEED_TRANSPOSE,
                       bu_large, CEED_VECTOR_ACTIVE);
  CeedOperatorSetField(op_mass_large, "v", Erestrictu, CEED_TRANSPOSE,
                       bu_large, CEED_VECTOR_ACTIVE);

  // Setup
  CeedOperatorApply(op_setup_small, X, qdata_small, CEED_REQUEST_IMMEDIATE);
  CeedOperatorApply(op_setup_large, X, qdata_large, CEED_REQUEST_IMMEDIATE);

  CeedVectorCreate(ceed, 2*Nu, &U);
  CeedVectorGetArray(U, CEED_MEM_HOST, &hu);
  for (int i = 0; i < Nu; i++) {
    hu[2*i] = 1.0;
    hu[2*i+1] = 2.0;
  }
  CeedVectorRestoreArray(U, &hu);
  CeedVectorCreate(ceed, 2*Nu, &V);

  // 'Small' operator
  CeedOperatorApply(op_mass_small, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum1 = 0.; sum2 = 0.;
  for (CeedInt i=0; i<Nu; i++) {
    sum1 += hv[2*i];
    sum2 += hv[2*i+1];
  }
  if (fabs(sum1-1.)>1e-10) printf("Computed Area: %f != True Area: 1.0\n", sum1);
  if (fabs(sum2-2.)>1e-10) printf("Computed Area: %f != True Area: 2.0\n", sum2);
  CeedVectorRestoreArrayRead(V, &hv);

  // 'Large' operator
  CeedOperatorApply(op_mass_large, U, V, CEED_REQUEST_IMMEDIATE);

  // Check output
  CeedVectorGetArrayRead(V, CEED_MEM_HOST, &hv);
  sum1 = 0.; sum2 = 0.;
  for (CeedInt i=0; i<Nu; i++) {
    sum1 += hv[2*i];
    sum2 += hv[2*i+1];
  }
  if (fabs(sum1-1.)>1e-10) printf("Computed Area: %f != True Area: 1.0\n", sum1);
  if (fabs(sum2-2.)>1e-10) printf("Computed Area: %f != True Area: 2.0\n", sum2);
  CeedVectorRestoreArrayRead(V, &hv);

  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedOperatorDestroy(&op_setup_small);
  CeedOperatorDestroy(&op_mass_small);
  CeedOperatorDestroy(&op_setup_large);
  CeedOperatorDestroy(&op_mass_large);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedElemRestrictionDestroy(&Erestrictui_small);
  CeedElemRestrictionDestroy(&Erestrictui_large);
  CeedElemRestrictionDestroy(&Erestrictxi);
  CeedBasisDestroy(&bu_small);
  CeedBasisDestroy(&bx_small);
  CeedBasisDestroy(&bu_large);
  CeedBasisDestroy(&bx_large);
  CeedVectorDestroy(&X);
  CeedVectorDestroy(&U);
  CeedVectorDestroy(&V);
  CeedVectorDestroy(&qdata_small);
  CeedVectorDestroy(&qdata_large);
  CeedDestroy(&ceed);
  return 0;
}
