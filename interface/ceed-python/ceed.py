# Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
# the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
# reserved. See files LICENSE and NOTICE for details.
#
# This file is part of CEED, a collection of benchmarks, miniapps, software
# libraries and APIs for efficient high-order finite element and spectral
# element discretizations for exascale applications. For more information and
# source code availability see http://github.com/ceed.
#
# The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
# a collaborative effort of two U.S. Department of Energy organizations (Office
# of Science and the National Nuclear Security Administration) responsible for
# the planning and preparation of a capable exascale ecosystem, including
# software, applications, hardware, advanced system engineering and early
# testbed platforms, in support of the nation's exascale computing imperative.

from _ceed import ffi, lib
from ceed_vector import Vector
from ceed_elemrestriction import ElemRestriction, IdentityElemRestriction, BlockedElemRestriction
from ceed_qfunction import QFunction, QFunctionByName, IdentityQFunction
from ceed_operator import Operator, CompositeOperator

# ------------------------------------------------------------------------------
# Ceed Enums
# ------------------------------------------------------------------------------
# CeedMemType
MEM_HOST       = lib.CEED_MEM_HOST
MEM_DEVICE     = lib.CEED_MEM_DEVICE
ceed_memtypes  = {mem_host:    "ceed_mem_host",
                  mem_device:  "ceed_mem_device"}

# CeedCopyMode
COPY_VALUES    = lib.CEED_COPY_VALUES
USE_POINTER    = lib.CEED_USE_POINTER
OWN_POINTER    = lib.CEED_OWN_POINTER
ceed_copymodes = {copy_values: "ceed_copy_values",
                  use_pointer: "ceed_use_pointer",
                  own_pointer: "ceed_own_pointer"}

# CeedEvalMode
EVAL_NONE      = lib.CEED_EVAL_NONE
EVAL_INTERP    = lib.CEED_EVAL_INTERP
EVAL_GRAD      = lib.CEED_EVAL_GRAD
EVAL_DIV       = lib.CEED_EVAL_DIV
EVAL_CURL      = lib.CEED_EVAL_CURL
EVAL_WEIGHT    = lib.CEED_EVAL_WEIGHT
ceed_evalmodes = {eval_none:   "ceed_eval_none",
                  eval_interp: "ceed_eval_interp",
                  eval_grad:   "ceed_eval_grad",
                  eval_div:    "ceed_eval_div",
                  eval_curl:   "ceed_eval_curl",
                  eval_weight: "ceed_eval_weight"}

# CeedTransposeMode
TRANSPOSE           = lib.CEED_TRANSPOSE
NOTRANSPOSE         = lib.CEED_NOTRANSPOSE
ceed_transposemodes = {transpose:     "ceed_transpose",
                       notranspose:   "ceed_notranspose"}

# CeedElemTopology
SHAPE_LINE          = lib.CEED_LINE
SHAPE_TRIANGLE      = lib.CEED_TRIANGLE
SHAPE_QUAD          = lib.CEED_QUAD
SHAPE_TET           = lib.CEED_TET
SHAPE_PYRAMID       = lib.CEED_PYRAMID
SHAPE_PRISM         = lib.CEED_PRISM
SHAPE_HEX           = lib.CEED_HEX
ceed_elemtopologies = {shape_line:     "ceed_line",
                       shape_triangle: "ceed_triangle",
                       shape_quad:     "ceed_quad",
                       shape_tet:      "ceed_tet",
                       shape_pyramid:  "ceed_pyramid",
                       shape_prism:    "ceed_prism",
                       shape_hex:      "ceed_hex"}

# ------------------------------------------------------------------------------
class Ceed():
  """Ceed: core components."""
  # Attributes
  _pointer = ffi.NULL

  # Constructor
  def __init__(self, resource = "/cpu/self"):
    # libCEED object
    self._pointer = ffi.new("Ceed *")

    # libCEED call
    resourceAscii = ffi.new("char[]", resource.encode("ascii"))
    lib.CeedInit(resourceAscii, self._pointer)

  # Get Resource
  def get_resource(self):
    """Get the full resource name for a Ceed context."""
    # libCEED call
    resource = ffi.new("char **")
    lib.CeedGetResource(self._pointer[0], resource)

    return ffi.string(resource[0]).decode("UTF-8")

  # Preferred MemType
  def get_preferred_memtype(self):
    """Return Ceed preferred memory type."""
    # libCEED call
    memtype = ffi.new("CeedMemType *", ceed_mem_host)
    lib.CeedGetPreferredMemType(self._pointer[0], memtype)

    return memtype[0]

  # CeedVector
  def Vector(self, size):
    """CeedVector: storing and manipulating vectors."""
    return Vector(self, size)

  # CeedElemRestriction
  def ElemRestriction(self, nelem, elemsize, nnodes, ncomp, mtype, cmode,
                      indices):
    """CeedElemRestriction: restriction from vectors to elements."""
    return ElemRestriction(self, nelem, elemsize, nnodes, ncomp, mtype,
                            cmode, indices)

  def IdentityElemRestriction(self, nelem, elemsize, nnodes, ncomp, mtype,
                              cmode):
    """CeedElemRestriction: identity restriction from vectors to elements."""
    return IdentityElemRestriction(self, nelem, elemsize, nnodes, ncomp, mtype,
                                    cmode)

  def BlockedElemRestriction(self, nelem, elemsize, blksize, nnodes, ncomp,
                             mtype, cmode, indices):
    """CeedElemRestriction: blocked restriction from vectors to elements."""
    return BlockedElemRestriction(self, nelem, elemsize, blksize, nnodes,
                                   ncomp, mtype, cmode, indices)

  # CeedBasis

  # CeedQFunction
  def QFunction(self, vlength, f, source):
    """CeedQFunction: independent operations at quadrature points."""
    return QFunction(self, vlength, f, source)

  def QFunctionByName(self, name):
    """CeedQFunctionByName: independent operations at quadrature points from gallery."""
    return QFunctionByName(self, name)

  def IdentityQFunction(self, size):
    """CeedIdenityQFunction: identity qfunction operation."""
    return IdentityQFunction(self, size)

  # CeedOperator
  def Operator(self, qf, dqf, qdfT):
    """CeedOperator: composed FE-type operations on vectors."""
    return Operator(self, qf, dqf, qdfT)

  def CompositeOperator(self):
    """CompositeCeedOperator: composition of multiple CeedOperators."""
    return CompositeOperator(self)

  # Destructor
  def __del__(self):
    # libCEED call
    lib.CeedDestroy(self._pointer)

# ------------------------------------------------------------------------------
