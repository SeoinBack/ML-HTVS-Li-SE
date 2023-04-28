import ctypes
import os, sys

# get filename
files = os.listdir(os.path.dirname(__file__))
for file in files:
  if("_bulk_enumerator" == file[:16] and ".so" in file):
    filename = file
package = ctypes.CDLL(os.path.join(os.path.dirname(__file__), filename))

class BULK(object):

  def __init__(self, tolerance=1e-3):
    '''
      Initialize the bulk class.
      Input:
        tolerance:  precision for floating point maths
      Output:
        None
    '''

    package.bulk_new.argtypes = [ctypes.c_double, \
                                  ctypes.POINTER(ctypes.c_char)]
    package.bulk_new.restype = ctypes.c_void_p

    package.bulk_get_spacegroup.argtypes = [ctypes.c_void_p]
    package.bulk_get_spacegroup.restype = ctypes.c_int
    
    package.bulk_delete.argtypes = [ctypes.c_void_p]
    package.bulk_delete.restype = None

    package.bulk_set_structure_from_file.argtypes = [ctypes.c_void_p, \
                                 ctypes.POINTER(ctypes.c_char)]
    package.bulk_set_structure_from_file.restype = None

    package.bulk_get_name.argtypes = [ctypes.c_void_p, \
                                        ctypes.POINTER(ctypes.c_char_p)]
    package.bulk_get_name.restype = None

    package.bulk_free_name.argtypes = [ctypes.c_char_p]
    package.bulk_free_name.restype = None

    package.bulk_get_species.argtypes = [ctypes.c_void_p, \
                            ctypes.POINTER(ctypes.c_int), \
               ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p))]
    package.bulk_get_species.restype = None

    package.bulk_free_species.argtypes = [ctypes.c_int, \
                                          ctypes.POINTER(ctypes.c_char_p)]
    package.bulk_free_species.restype = None


    package.bulk_get_wyckoff.argtypes = [ctypes.c_void_p, \
                            ctypes.POINTER(ctypes.c_int), \
               ctypes.POINTER(ctypes.POINTER(ctypes.c_char_p))]
    package.bulk_get_wyckoff.restype = None

    package.bulk_free_wyckoff.argtypes = [ctypes.c_int, \
                                          ctypes.POINTER(ctypes.c_char_p)]
    package.bulk_free_wyckoff.restype = None

    temp_path = ctypes.create_string_buffer(
                    os.path.abspath(os.path.dirname(__file__)).encode("utf-8"))
    self.obj = package.bulk_new(ctypes.c_double(tolerance), temp_path);


  def delete(self):
    '''
      free the c++ heap memory.
      Input:
        None
      Output:
        None
    '''

    package.bulk_delete(self.obj);


  def set_structure_from_file(self, file_content):
    '''
      set the bulk from vasp file
      Input:
        file_content: poscar file content
      Output:
        None
    '''

    temp_file_content = \
                    ctypes.create_string_buffer(file_content.encode("utf-8"))
    package.bulk_set_structure_from_file(self.obj, temp_file_content);


  def get_wyckoff(self):
    '''
      get wyckoffs for the set material
      Input:
        None
      Output:
        list of wyckoffs
    '''

    temp_num = ctypes.c_int();
    temp_wyckoff = \
            ctypes.POINTER(ctypes.c_char_p)()
    package.bulk_get_wyckoff(self.obj, ctypes.byref(temp_num),
                                            ctypes.byref(temp_wyckoff));

    wyckoff = ["" for i in range(temp_num.value)]
    for i in range(temp_num.value):
      wyckoff[i] = temp_wyckoff[i].decode('utf-8')

    package.bulk_free_wyckoff(temp_num, temp_wyckoff);

    return wyckoff


  def get_species(self):
    '''
      get species for the set material
      Input:
        None
      Output:
        list of species
    '''

    temp_num = ctypes.c_int();
    temp_species = \
            ctypes.POINTER(ctypes.c_char_p)()
    package.bulk_get_species(self.obj, ctypes.byref(temp_num),
                                            ctypes.byref(temp_species));

    species = ["" for i in range(temp_num.value)]
    for i in range(temp_num.value):
      species[i] = temp_species[i].decode('utf-8')

    package.bulk_free_species(temp_num, temp_species);

    return species

  def get_spacegroup(self):
    '''
      get the spacegroup number
      Input: None
      Output: spacegroup number
    '''

    temp = package.bulk_get_spacegroup(self.obj);
    return int(temp)

  def get_name(self):
    '''
      get name for the set material
      Input:
        None
      Output:
        name
    '''

    temp_name = ctypes.c_char_p()
    package.bulk_get_name(self.obj, ctypes.byref(temp_name));

    name = temp_name.value.decode('utf-8')

    package.bulk_free_name(temp_name);

    return name

