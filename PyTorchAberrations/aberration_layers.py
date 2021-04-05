import torch
from torch.nn import Module, ZeroPad2d
from PyTorchAberrations.aberration_functions import complex_mul, conjugate
from PyTorchAberrations.aberration_functions import pi2_shift, complex_exp, crop_center2
from PyTorchAberrations.aberration_functions import complex_conv2d, complex_conv_transpose2d

from torch.nn import ConvTranspose2d
from torch.nn.functional import conv2d

from math import ceil
PI = 3.14159265358979323846264338327950288419716939937510582


################################################################
################### AUTOGRAD FUNCTIONS #########################
################################################################
     
class ComplexZernikeFunction(torch.autograd.Function):
    '''
    Function that apply a complex Zernike polynomial to the phase of a batch 
    of compleximages (or a matrix).
    '''
    @staticmethod
    def forward(ctx, input, alpha, j):
        
        
        nx = torch.arange(0,1,1./input.shape[1], dtype = torch.float32)
        ny = torch.arange(0,1,1./input.shape[2], dtype = torch.float32)

        X0, Y0 = 0.5+0.5/input.shape[1], 0.5+0.5/input.shape[2]
        X,Y = torch.meshgrid(nx,ny)
        X = X.to(input.device)-X0
        Y = Y.to(input.device)-Y0
        
        # see https://en.wikipedia.org/wiki/Zernike_polynomials
        if j == 0:
            F = torch.ones_like(X)
        elif j == 1:
            F = X
        elif j == 2:
            F = Y
        elif j == 3:
            # Oblique astigmatism
            F = 2.*X.mul(Y)
        elif j == 4:
            # Defocus
            F = X**2+Y**2
        elif j == 5:
            # Vertical astigmatism
            F = X**2-Y**2
        else:
            R = torch.sqrt(X**2+Y**2)
            THETA = torch.atan2(Y, X)
            if j == 6:
                # Vertical trefoil 
                F = torch.mul(R**3, torch.sin(3.*THETA))
            elif j == 7:
                # Vertical coma
                F = torch.mul(3.*R**3,torch.sin(3.*THETA))
            elif j == 8:
                # Horizontal coma 
                F = torch.mul(3.*R**3,torch.cos(3.*THETA))
            elif j == 9:
                # Oblique trefoil 
                F = torch.mul(R**3, torch.cos(3.*THETA))
            elif j == 10:
                # Oblique quadrafoil 
                F = 2.*torch.mul(R**4, torch.sin(4.*THETA))
            elif j == 11:
                # Oblique secondary astigmatism 
                F = 2.*torch.mul(4.*R**4-3.*R**2, torch.sin(2.*THETA))
            elif j == 12:
                # Primary spherical
                F = 6.*R**4-6.*R**2 + torch.ones_like(R)
            elif j == 13:
                # Vertical secondary astigmatism 
                F = 2.*torch.mul(4.*R**4-3.*R**2, torch.cos(2.*THETA))
            elif j == 14:
                # Vertical quadrafoil 
                F = 2.*torch.mul(R**4, torch.cos(4.*THETA))
            else:
                raise
        
        weight = torch.exp(1j*alpha*F)
        
        ctx.save_for_backward(input, alpha, F)
        output = input*weight
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, alpha, F = ctx.saved_tensors
        
        weight = torch.exp(1j*alpha*F)
        
        grad_input = grad_alpha = None
        if ctx.needs_input_grad[0]:
            grad_input = grad_output*weight.conj()

        if ctx.needs_input_grad[1]:
            grad_alpha = torch.sum(grad_output*(1j*F*weight*input).conj()).real
            grad_alpha.unsqueeze_(0)
            
        return grad_input, grad_alpha, None
    

class FreeSpacePropagationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, z, dx, lambda_): 
        
        real_dtype = input.real.dtype
        
        k = 2*PI/lambda_
        
        # c determines the size of the kernel, to allow to have a resolution
        # of the final image after convolution the same as the input, 
        # we need c = 2
        c = 2
        
        nx = torch.arange(0,c*input.shape[-2]*dx-1, dx, dtype = real_dtype)
        ny = torch.arange(0,c*input.shape[-1]*dx-1, dx, dtype = real_dtype)
        
        # to avoid the singularity for z = 0 and R = 0
        eps = 1e-4

        X0 = .5*dx*(c*input.shape[-2]+eps)-1.
        Y0 = .5*dx*(c*input.shape[-1]+eps)-1.

        # X, Y grid
        X,Y = torch.meshgrid(nx,ny)
        X = X.to(input.device)-X0
        Y = Y.to(input.device)-Y0
        
        # polar coordinates for each position on the grid
        R = torch.sqrt(X**2+Y**2+z**2)
        RHO = torch.sqrt(X**2+Y**2)
        THETA = torch.atan2(RHO, z)
        
        
        # kernel for Huygens-Fresnel 
        kernel = 1./R*complex_exp(k*R)
        kernel = kernel[None,None,...]
        
        # inclination factor for  Rayleigh-Sommerfeld
        K = torch.cos(THETA)
        
        # save stuff for gradient computation during backward
        ctx.save_for_backward(input, kernel, K, z, R, THETA, RHO, k)      

        output = complex_conv2d(K*kernel, input, stride = 1)
        output = output.flip(-1,-2)
    
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, kernel, K, z, R, THETA, RHO, k = ctx.saved_tensors
        grad_input = grad_z = None
        if ctx.needs_input_grad[0]:
            # gradient wrt input
            # gradient of conv is conv_transpose
            grad_input = complex_conv_transpose2d(K*kernel,grad_output.conj())
            # take only the center part to match input/output size
            grad_input = crop_center2(grad_input,input.shape[-2],input.shape[-1]).conj()
            
        if ctx.needs_input_grad[1]:
            # gradient wrt distance z
            grad_z = complex_conv_transpose2d(input,grad_output.flip(-1,-2).conj())
            # derivative of the kernel
            grad_kernel = kernel*(1j*k-1./R[None,None,...])*z/R[None,None,...]
            grad_K = torch.sin(THETA)*1/(1+z**2/RHO[None,None,...]**2)*1/RHO
            grad_z *= (grad_kernel*K+grad_K*kernel)
            grad_z = grad_z.conj().real.type(z.dtype)
            
        # the last two input dx and lambda_ are supposed to be fixed (requires_grad = False),
        # we do not compute the gradient
        return grad_input, grad_z, None, None
    

#######################################################
#################### MODULES ##########################
#######################################################

class FreeSpacePropagation(Module):
    '''
    Layer representing free-space propagation.
    Only one parameter, the propagation distance z, is learned.
    Initial value is 0.
    '''
    def __init__(self, dx, lambda_, z_init_value = 0.):
        super(FreeSpacePropagation, self).__init__()
        self.z = torch.nn.Parameter(torch.tensor(z_init_value), requires_grad=True)
        self.dx = torch.tensor(dx)
        self.dx._requires_grad = False
        self.lambda_ = torch.tensor(lambda_)
        self.lambda_._requires_grad = False

    def forward(self, input):
        return FreeSpacePropagationFunction.apply(input, self.z, self.dx, self.lambda_)

class ComplexZernike(Module):
    '''
    Layer that apply a complex Zernike polynomial to the phase of a batch 
    of compleximages (or a matrix).
    Only one parameter, the strenght of the polynomial, is learned.
    Initial value is 0.
    '''
    def __init__(self, j):
        super(ComplexZernike, self).__init__()
        assert j in range(15)
        self.j = j
        self.alpha = torch.nn.Parameter(torch.zeros(1), requires_grad=True)


    def forward(self, input):
        return ComplexZernikeFunction.apply(input, self.alpha, self.j)
    
    
from PyTorchAberrations.aberration_functions import complex_exp, crop_center2

class PhasePlane(Module):
    """
    Layers that simulate the transmission through a thin diffuser.
    """
    
    def __init__(self, 
                 shape, 
                 corr_length, 
                 dx, init_to_zero = True,
                 overlap_coeff = 2.):
        super(PhasePlane, self).__init__()
        
        self.shape = shape
        sigma_k = .5*corr_length/dx
        # create the kernel (Gaussian)
        n_k = int(5*sigma_k)
        # overlap_coeff determines how much neighbouring cells overlaps
        dilation = int(sigma_k*overlap_coeff)
        kernel_size = [ceil(s/(dilation)-1) for s in shape]
        
        x_k = torch.arange(n_k)
        X,Y = torch.meshgrid(x_k,x_k)
        X0 = Y0 = n_k/2-.5
        Rsq = (X-X0)**2+(Y-Y0)**2
        self.K = torch.exp(-Rsq/(2*sigma_k**2))[None,None,...]
        
        self.convt = ConvTranspose2d(
                        in_channels=1,
                        out_channels=1,
                        dilation=dilation,
                        kernel_size = kernel_size
                     )
        
        if init_to_zero:
            # initialize weight (phase) to zero
            self.convt.weight.data.fill_(0.)    
        else:
            # random phase from uniform distribution between 0 and 2pi
            torch.nn.init.uniform_(self.convt.weight, 0., 2*PI)

        
    def forward(self,input):
        # get the mask of phase value
        phase_plane = self.convt(self.K)
        phase_plane = crop_center2(phase_plane, self.shape[0], self.shape[1])
        # multiply input field by the complex phase plane contributions
        input = input*complex_exp(phase_plane)
        return input

class ComplexScaling(Module):
    '''
    Layer that apply a global scaling to a stack of 2D complex images (or matrix).
    Only one parameter, the scaling factor, is learned. 
    Initial value is 1.
    '''
    def __init__(self):
        super(ComplexScaling, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
            input = torch.view_as_real(input).permute((0,3,1,2))

            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta)*(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                      
                                         
            return torch.view_as_complex(torch.nn.functional.grid_sample(input, grid, align_corners=True).permute((0,2,3,1)).contiguous())
        
class ComplexDeformation(Module):
    '''
    Layer that apply a global affine transformation to a stack of 2D complex images (or matrix).
    6 parameters are learned.
    '''
    def __init__(self):
        super(ComplexDeformation, self).__init__()
        
        self.theta = torch.nn.Parameter(torch.tensor([0., 0, 0, 0, 0., 0]))
        # parameters 0 and 4 are the ones corresponding to x and y scaling
        # parameters 1 and 3 are the ones corresponding to shearing
        # parameters 2 and 6 are shifts

    def forward(self, input):
            input = torch.view_as_real(input).permute((0,3,1,2))
            grid = torch.nn.functional.affine_grid(
                ((1.+self.theta).mul(torch.tensor([1, 0., 0., 0., 1, 0.],
                                         dtype=input.dtype).to(input.device))
                ).reshape((2,3)).expand((input.shape[0],2,3)), 
                                 input.size())                 

            return torch.view_as_complex(torch.nn.functional.grid_sample(input, grid, align_corners=True).permute((0,2,3,1)))
