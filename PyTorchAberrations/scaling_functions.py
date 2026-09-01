import numpy as np
from scipy.ndimage import zoom

def cpx_zoom(cpx_img, zoom_factor):

    zoom_facs = [1]*len(cpx_img.shape)
    zoom_facs[-2:] = [zoom_factor, zoom_factor]
    cpx_zoom = zoom(np.real(cpx_img),zoom_facs) + 1j * zoom(np.imag(cpx_img),zoom_facs)

    return cpx_zoom

def pad_to_size(cpx_img, size):
    current_size = np.array(cpx_img.shape[-2:])
    dif_size = size - current_size
    pad_right = dif_size // 2
    pad_left = dif_size - pad_right
    return np.pad(cpx_img,((0,0),(pad_left[0], pad_right[0]),(pad_left[1], pad_right[1])))
    

def crop_center(img, crop_res_in):
    crop_res = np.array([0,0])+crop_res_in
    dif_size = img.shape[-2:] - crop_res
    start = (dif_size)//2
    if img.ndim == 3:        
        cropped_images = img[:,
                             start[0]:start[0]+crop_res[0],
                             start[1]:start[1]+crop_res[1]] 
    elif img.ndim == 2:        
        cropped_images = img[start[0]:start[0]+crop_res[0],
                             start[1]:start[1]+crop_res[1]]    
    return cropped_images

def compute_rms_width(img):
    size = img.shape[-2:]
    X, Y = np.meshgrid(np.arange(size[1]), np.arange(size[0]))
    norm_img = np.mean(img)
    center = [np.mean(X*img),np.mean(Y*img)]/norm_img
    rms_w =np.sqrt(np.mean(((X-center[0])**2 + (Y-center[1])**2)*img)/norm_img)

    return float(rms_w)

def normalize_seq(img):
    return img/np.linalg.norm(img, axis=(-2,-1)).reshape((-1,1,1))

def resize_modes(modes, zoom_factor, size):
    new_modes = cpx_zoom(modes,zoom_factor)
    cur_size = new_modes.shape[-1]
    if cur_size > size:
        new_modes = crop_center(new_modes, size)
    elif cur_size < size:
        new_modes = pad_to_size(new_modes, size)
    else:
        pass
    return normalize_seq(new_modes)

def get_inout_modes(modes, TM, pola_inout=(2,2)):
    
    mshp = np.shape(modes)
    if len(mshp)==2:
        modes_2d = np.reshape(modes, (mshp[0], int(np.sqrt(mshp[1])), int(np.sqrt(mshp[1]))))
    elif len(mshp)==3:
        modes_2d = modes
    else:
        raise ValueError('modes needs to be 2d or 3d')

    # Define various sizes form TM 
    N2_out, N2_in = TM.shape[:2]
    if pola_inout[0]==2:
        N2_in = N2_in//2
    if pola_inout[1]==2:
        N2_out = N2_out//2
    N_in = int(np.sqrt(N2_in))
    N_out = int(np.sqrt(N2_out))

    # Compute incoherent sum of modes
    I_mean_theo = np.mean(np.abs(modes_2d)**2,axis=0)

    I_in_exp = np.mean(np.abs(TM)**2, axis = 0)  
    if pola_inout[0]==2:
        I_in_exp = I_in_exp[:N2_in] + I_in_exp[N2_in:]
    I_in_exp = I_in_exp.reshape([N_in]*2)
    
    I_out_exp = np.mean(np.abs(TM)**2, axis = 1) 
    if pola_inout[1]==2:
        I_out_exp = I_out_exp[:N2_out] + I_out_exp[N2_out:]
    I_out_exp = I_out_exp.reshape([N_out]*2)
    
    # Compute the rms width for each case
    w_in = compute_rms_width(I_in_exp)
    w_out = compute_rms_width(I_out_exp)
    w_hd = compute_rms_width(I_mean_theo)

    modes_in = resize_modes(modes_2d, w_in/w_hd, N_in)
    modes_out = resize_modes(modes_2d, w_out/w_hd, N_out)
    
    if len(mshp)==2:
        return modes_in.reshape(mshp[0], -1), modes_out.reshape(mshp[0], -1)
    else:
        return modes_in, modes_out


