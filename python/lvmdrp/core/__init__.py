#from header import *
#from positionTable import *
#from cube import *
#from image import *
#from rss import *
#from spectrum1d import *
#from passband import *
#from tracemask import *
#from apertures import *
#import exceptions, copy_reg
#from types import *


#def _pickle_method(method):
      #func_name = method.im_func.__name__
      #obj = method.im_self
      #cls = method.im_class
      #if func_name.startswith('__') and not func_name.endswith('__'):
	#cls_name = cls.__name__.lstrip('_')
	#if cls_name: func_name = '_' + cls_name + func_name
      #return _unpickle_method, (func_name, obj, cls)
    
#def _unpickle_method(func_name, obj, cls):
      #for cls in cls.mro():
	#try:
	  #func = cls.__dict__[func_name]
	#except KeyError:
	  #pass
	#else:
	  #break
      #return func.__get__(obj, cls)
#copy_reg.pickle(MethodType,_pickle_method, _unpickle_method)
