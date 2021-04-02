import torch
import numpy as np
from torch.nn import Module, Sequential, Identity
from torch.nn import ZeroPad2d
from PyTorchAberrations.aberration_layers import ComplexDeformation
from PyTorchAberrations.aberration_layers import ComplexZernike, ComplexScaling
from PyTorchAberrations.aberration_layers import FreeSpacePropagation
from PyTorchAberrations.aberration_functions import crop_center, complex_fftshift
from PyTorchAberrations.aberration_functions import complex_ifftshift, conjugate, normalize
from PyTorchAberrations.aberration_functions import complex_fft, complex_ifft

class AberrationModes(torch.nn.Module):
    '''
    Model for input and output aberrations.
    Apply an `Aberration` model to the input and output mode basis.
    '''
    def __init__(self, 
                     inpoints,
                     onpoints,
                     padding_coeff = 0.,
                     list_zernike_ft = list(range(3)),
                     list_zernike_direct = list(range(3)),
                     propagation = False,
                     deformation = 'single'):
        super(AberrationModes, self).__init__()
        self.abberation_output = Aberration(onpoints,
                                            list_zernike_ft = list_zernike_ft,
                                            list_zernike_direct = list_zernike_direct, 
                                            padding_coeff = padding_coeff,
                                            propagation = propagation,
                                            deformation = deformation)
        self.abberation_input = Aberration(inpoints,
                                            list_zernike_ft = list_zernike_ft,
                                            list_zernike_direct = list_zernike_direct, 
                                            padding_coeff = padding_coeff,
                                            propagation = propagation,
                                            deformation = deformation)
        self.inpoints = inpoints
        self.onpoints = onpoints

    def forward(self,input, output):
        
        output_modes = output
        output_modes = self.abberation_output(output_modes)
        # output_modes = normalize(output_modes.reshape((-1,self.onpoints**2,2)),device = self.device).reshape((-1,self.onpoints,self.onpoints,2))
        

        input_modes = input
        input_modes = self.abberation_input(input_modes)
        # input_modes = normalize(input_modes.reshape((-1,self.inpoints**2,2)),device = self.device).reshape((-1,self.inpoints,self.inpoints,2))

        return output_modes, input_modes

    
class Aberration(torch.nn.Module):
    '''
    Model that apply aberrations (direct and Fourier plane) and a global scaling
    at the input dimension of a matrix.
    '''
    def __init__(self, 
                 shape,
                 list_zernike_ft,
                 list_zernike_direct,
                 padding_coeff = 0., 
                 propagation = False,
                 deformation = 'single',
                 features = None):
        # Here we define the type of Model we want to be using, the number of polynoms and if we want to implement a deformation.
        super(Aberration, self).__init__()
        
        #Check whether the model is given the lists of zernike polynoms to use or simply the total number to use
        if type(list_zernike_direct) not in [list, np.ndarray]:
            list_zernike_direct = range(0,list_zernike_direct)
        if type(list_zernike_ft) not in [list, np.ndarray]:
            list_zernike_ft = range(0,list_zernike_ft)

        self.nxy = shape
        
        # padding layer, to have a good FFT resolution
        # (requires to crop after IFFT)
        padding = int(padding_coeff*self.nxy)
        self.pad = ZeroPad2d(padding)
        
        # scaling x, y
        if deformation == 'single':
            self.deformation = ComplexDeformation()
        elif deformation == 'scaling':
            self.deformation = ComplexScaling()
        else:
            self.deformation = Identity()
            
        self.propagation = FreeSpacePropagation(dx = torch.tensor([2.]), 
                                                lambda_ = torch.tensor([1.])) if propagation else None
        
        self.zernike_ft = Sequential(*(ComplexZernike(j=j + 1) for j in list_zernike_ft))
        self.zernike_direct = Sequential(*(ComplexZernike(j=j + 1) for j in list_zernike_direct))
       
      
    def forward(self,input):
        assert(input.shape[1] == input.shape[2])
        
        
        input = torch.view_as_complex(input)
        # free-space propagation
        print(f'>>>{input.shape=}')
        print(self.propagation(input[0,...][None,None,...]).shape)
        if self.propagation:
#             print(input.dtype)
            input = torch.cat(
                    [self.propagation(input[i,...][None,None,...]) for i in range(input.shape[0])],
                    dim = 1
                 )
#             print('ok')
            input = input.squeeze(0)

        print(f'{input.shape=}', '---')
        
        # padding
        input = self.pad(input)

        # scaling
        input = self.deformation(input)
        #self.deformation(input)
        
        # to Fourier domain
        input = complex_ifftshift(input)
        input = complex_fft(input, 2)
        input = complex_fftshift(input)
#         input = torch.view_as_real(complex_fftshift(input))

        # Zernike layers in the Fourier plane
        input = self.zernike_ft(input)

        # to direct domain
#         input = torch.view_as_complex(input)
        input = complex_ifftshift(input)
        input = complex_ifft(input, 2)
        input = complex_fftshift(input)
#         input = torch.view_as_real(input)
         
        # Zernike layers in the direct plane
        input = self.zernike_direct(input)
        
        # Crop at the center (because of coeff) 
        input = crop_center(input,self.nxy)

        return torch.view_as_real(input)
      