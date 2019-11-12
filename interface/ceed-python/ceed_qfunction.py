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
from abc import ABC

# ------------------------------------------------------------------------------
class _QFunctionBase(ABC):

  # Attributes
  _ceed = ffi.NULL
  _pointer = ffi.NULL

  # Apply CeedQFunction
  def apply(self, q, u, v):
    """Apply the action of a CeedQFunction."""
    # libCEED call
    lib.CeedQFunctionApply(self._pointer[0], q, u._pointer[0], v._pointer[0])

  # Destructor
  def __del__(self):
    # libCEED call
    lib.CeedQFunctionDestroy(self._pointer)

# ------------------------------------------------------------------------------
class _QFunction(_QFunctionBase):
  """CeedQFunction: independent operations at quadrature points."""

  # Constructor
  def __init__(self, ceed, vlength, f, source):
    # libCEED object
    self._pointer = ffi.new("CeedQFunction *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    sourceAscii = ffi.new("char[]", source.encode('ascii'))
    lib.CeedQFunctionCreateInterior(self._ceed._pointer[0], vlength, f,
                                    sourceAscii, self._pointer)

  # Add fields to CeedQFunction
  def addInput(self, fieldname, size, emode):
    """Add a CeedQFunction input."""
    # libCEED call
    fieldnameAscii = ffi.new("char[]", fieldname.encode('ascii'))
    lib.CeedQFunctionAddInput(self._pointer[0], fieldnameAscii, size, emode)

  def addOutput(self, fieldname, size, emode):
    """Add a CeedQFunction output."""
    # libCEED call
    fieldnameAscii = ffi.new("char[]", fieldname.encode('ascii'))
    lib.CeedQFunctionAddOutput(self._pointer[0], fieldnameAscii, size, emode)

# ------------------------------------------------------------------------------
class _QFunctionByName(_QFunctionBase):
  """CeedQFunctionByName: independent operations at quadrature points from gallery."""

  # Constructor
  def __init__(self, ceed, name):
    # libCEED object
    self.pointer = ffi.new("CeedQFunction *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    nameAscii = ffi.new("char[]", name.encode('ascii'))
    lib.CeedQFunctionCreateByName(self._ceed._pointer[0], nameAscii,
                                  self._pointer)

# ------------------------------------------------------------------------------
class _QFunctionIdentity(_QFunctionBase):
  """CeedIdentityQFunction: identity qfunction operation."""

  # Constructor
  def __init__(self, ceed, size):
    # libCEED object
    self._pointer = ffi.new("CeedQFunction *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    lib.CeedQFunctionCreateIdentity(self._ceed._pointer[0], size, self._pointer)

# ------------------------------------------------------------------------------
