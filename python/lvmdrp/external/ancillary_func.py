from lvmdrp.core.spectrum1d import *
import numpy

def extinctCAHA(wave, extinct_v, type='mean'):
    if type=='mean':
        data = (0.0935*(wave/5450.0)**(-4)+(((0.8*extinct_v)-0.0935)*(wave/5450.0)**(-0.8)))
        
    elif type=='winter' or type=='summer':
        if type=='winter':
            (f1, f2, f3) = (1.02, 0.94, 0.29)
            
        elif type=='summer':
            (f1, f2, f3) = (1.18, 4.52, 0.19)
        k1 = f1*7.25e-3*(wave/10000.0)**(-4)
        k2 = f2*0.006*(wave/10000.0)**(-0.8)
        k3 = f3*0.015*numpy.exp(-((wave-6000.0)/1200.0))
        data = k1+k2+k3
        scale_idx = numpy.argsort((wave-5500.0)**2)[0]
        scale_offset = extinct_v-data[scale_idx]
        data = data+scale_offset
        
    spec = Spectrum1D(wave=wave, data=data)    
    return spec
    
def extinctParanal(wave):
  wave_base = numpy.concatenate((numpy.arange(3325,6780,50),numpy.array([7060,7450,7940,8500,8675,8850,10000])))
  extinct = numpy.array([0.686,0.606,0.581,0.552,0.526,0.504,0.478,0.456,0.430,0.409,0.386,0.378,0.363,0.345,0.330,0.316,0.298,0.285,0.274,0.265,0.253,0.241,0.229,0.221,0.212,0.204,0.198,0.190,0.185,0.182,0.176,0.169,0.162,0.157,0.156,0.153,0.146,0.143,0.141,0.139,0.139,0.134,0.133,0.131,0.129,0.127,0.128,0.130,0.134,0.132,0.124,0.122,0.125,0.122,0.117,0.115,0.108,0.104,0.102,0.099,0.095,0.092,0.085,0.086,0.083,0.081,0.076,0.072,0.068,0.064,0.064,0.048,0.042,0.032,0.030,0.029,0.022])
  spec_raw = Spectrum1D(wave=wave_base,data=extinct)
  spec =spec_raw.resampleSpec(wave)
  return spec
    
def galExtinct(wave, Rv):
    
    m=wave/10000.0
    x=1.0/m
    y=(x-1.82)
    ax=1+(0.17699*y)-(0.50447*y**2)-(0.02427*y**3)+(0.72085*y**4)+(0.01979*y**5)-(0.77530*y**6)+(0.32999*y**7)
    bx=(1.41338*y)+(2.28305*y**2)+(1.07233*y**3)-(5.38434*y**4)-(0.62251*y**5)+(5.30260*y**6)-(2.09002*y**7)
    

    Arat=(ax+(bx/Rv)).astype(numpy.float32)
    spec = Spectrum1D(wave=wave, data=Arat)
    return spec
