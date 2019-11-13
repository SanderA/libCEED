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
import sys
import numpy as np

# ------------------------------------------------------------------------------
class Vector:
  """CeedVector: storing and manipulating vectors."""

  # Attributes
  _ceed = ffi.NULL
  _pointer = ffi.NULL

  # Constructor
  def __init__(self, ceed, size):
    # CeedVector object
    self._pointer = ffi.new("CeedVector *")

    # Reference to Ceed
    self._ceed = ceed

    # libCEED call
    lib.CeedVectorCreate(self._ceed._pointer[0], size, self._pointer)

  # Set Vector's data array
  def set_array(self, mtype, cmode, array):
    """Set the array used by a CeedVector, freeing any previously allocated
       array if applicable."""
    # Setup the numpy array for the libCEED call
    array_pointer = ffi.new("CeedScalar *")
    array_pointer = ffi.cast("CeedScalar *", array.__array_interface__['data'][0])

    # libCEED call
    lib.CeedVectorSetArray(self._pointer[0], mtype, cmode, array_pointer)

  # Get Vector's data array
  def get_array(self, mtype):
    """Get read/write access to a CeedVector via the specified memory type."""

    # Retrieve the length of the array
    length_pointer = ffi.new("CeedInt *")
    lib.CeedVectorGetLength(self._pointer[0], length_pointer)

    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorGetArray(self._pointer[0], mtype, array_pointer)

    # Create buffer object from returned pointer
    buff = ffi.buffer(array_pointer[0], ffi.sizeof("CeedScalar") * length_pointer[0])
    # Return numpy array created from buffer
    return np.frombuffer(buff, dtype="float64")

  # Get Vector's data array in read-only mode
  def get_array_read(self, mtype):
    """Get read-only access to a CeedVector via the specified memory type."""

    # Retrieve the length of the array
    length_pointer = ffi.new("CeedInt *")
    lib.CeedVectorGetLength(self._pointer[0], length_pointer)

    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorGetArrayRead(self._pointer[0], mtype, array_pointer)

    # Create buffer object from returned pointer
    buff = ffi.buffer(array_pointer[0], ffi.sizeof("CeedScalar") * length_pointer[0])
    # Create numpy array from buffer
    ret = np.frombuffer(buff, dtype="float64")
    # Make the numpy array read-only
    ret.flags['WRITEABLE'] = False
    return ret

  # Restore the Vector's data array
  def restore_array(self):
    """Restore an array obtained using getArray()."""
    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorRestoreArray(self._pointer[0], array_pointer)

  # Restore an array obtained using getArrayRead
  def restore_array_read(self):
    """Restore an array obtained using getArrayRead()."""
    # Setup the pointer's pointer
    array_pointer = ffi.new("CeedScalar **")

    # libCEED call
    lib.CeedVectorRestoreArrayRead(self._pointer[0], array_pointer)

  # Get the length of a Vector
  def get_length(self):
    """Get the length of a CeedVector."""
    length_pointer = ffi.new("CeedInt *")

    # libCEED call
    lib.CeedVectorGetLength(self._pointer[0], length_pointer)

    return length_pointer[0]

  # Set the Vector to a given constant value
  def set_value(self, value):
    """Set the Vector to a constant value."""

    # libCEED call
    lib.CeedVectorSetValue(self._pointer[0], value)

  # Sync the Vector to a specified memtype
  def sync_array(self, mtype):
    """Sync the Vector to a specified memtype."""

    # libCEED call
    lib.CeedVectorSyncArray(self._pointer[0], mtype)

  # View a Vector
  def view(self, format = ffi.NULL, file = sys.stdout):
    """View a Vector."""

    # Check if format is a string before encoding it
    if type(format) == "str":
      fstr = format.encode("ascii", "strict")
    else:
      fstr = format

    # libCEED call
    lib.CeedVectorView(self._pointer[0], fstr, file)

  # Destructor
  def __del__(self):
    # libCEED call
    lib.CeedVectorDestroy(self._pointer)

# ------------------------------------------------------------------------------
class _VectorClone:
  """Copy a CeedVector """

  # Constructor
  def __init__(self, ceed, pointer):
    # CeedVector object
    self._pointer = pointer

    # Reference to Ceed
    self._ceed = ceed

# ------------------------------------------------------------------------------
