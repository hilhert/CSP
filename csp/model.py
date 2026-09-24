import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .model_classical import  BaseSeqModel


'''
experimental but seems not causual valid.
def haar_multiscale(x):
    """x: [B, T, 2,H] -> multiscale Haar coefficients，Keep [B, T, 2,H]"""
    B, T, _ , H = x.shape
    cur = x
    levels = []
    while cur.size(1) > 1:
        if cur.size(1) % 2 == 1:
            cur = cur[:, :-1,:,:]
        diff = cur[:, 1::2, :,:] - cur[:, 0::2, :,:]
        avg = (cur[:, 0::2, :,:] + cur[:, 1::2, :,:]) / 2
        levels.append(diff)
        cur = avg
    levels.append(cur)
    out = torch.cat(levels, dim=1)
    if out.size(1) < T:
        out = F.pad(out, (0, 0, 0, T - out.size(1)))
    return out[:, :T,:,:].detach()

# this version is extremely accurate for single position tracking with parity, mod3, but not valid for parenthesis 
class CSP_Hidden_AccuMod(nn.Module):
    """
    Pure Mamba Recur with Skip Normalization!
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        # Rotation
        self.theta_proj = nn.Linear(2*hidden_dim, 1)
        # state transform matrix
        self.B_proj = nn.Linear(hidden_dim, hidden_dim)

        # Decay factor (alpha = exp(log_alpha), initialized to 1)
        self.delta_proj =  nn.Linear(2*hidden_dim, 1)
        self.gamma_proj = nn.Linear(2*hidden_dim, 1)

        # Skip gate (per-dimension, initialized to 0.5)
        self.skip_gate = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        
    def forward(self, x):
        """
        Args:
            x: [B, T, 2, H]  0: real, 1: imag
        Returns:
            x: [B, T, 2, H]
        """
        B, T, _, H = x.shape
        
         #1. Rotation with prefetch sensing cum factor!
        
        x_haar = haar_multiscale(x)
        
        theta_all = torch.tanh(self.theta_proj(x_haar.view(B,T,-1)))* math.pi  # [B, T, 1]
        
        cos_a, sin_a = torch.cos(theta_all), torch.sin(theta_all)  # [B, T, 1]

        # construct of rotation matrix
        Rot1 = torch.cat([cos_a, -sin_a], dim=-1)  # [B, T, 1, 2]
        Rot2 = torch.cat([sin_a, cos_a], dim=-1)   # [B, T, 1, 2]
        Rot = torch.stack([Rot1, Rot2], dim=-2)    # [B, T, 2, 2]

       
        x_rot = torch.einsum('btih, btji -> btjh', x, Rot)  
        
        #1. Element construct after rotation
        gamma_all       =     (1 + torch.sin(self.gamma_proj(x_rot.view(B,T,-1)))).unsqueeze(-1)/2 
        x_g             =     self.B_proj(x)* gamma_all
        
        
        #2. Mamba style recur with diagonal set to 1
        delta_all       =     F.softplus(self.delta_proj(x_g[:,:-1,:,:].view(B,T-1,-1)))
        alpha_all       =     torch.exp(-delta_all).unsqueeze(-1)               
        
        alpha_cumprod   =     torch.cat([torch.ones(B,1,1,1,device=x.device).detach(),torch.cumprod(alpha_all,dim=1)],dim=1) 
        x_recur = alpha_cumprod*torch.cumsum(x_g /(alpha_cumprod+1e-9).detach(), dim=1)
         
        #3. Skip connection with elementwise magnitude normalization
        gate = torch.sigmoid(self.skip_gate) 
        x_out = x * gate + F.silu(x_recur)       
        magnitude = torch.sqrt(x_out[:, :, 0, :]**2 + x_out[:, :, 1, :]**2 + 1e-8).detach()
        x_out = x_out / magnitude.unsqueeze(-2)  

        return x_out
'''



class CSP_Hidden_FAST(nn.Module):
    """
    Complex Propagator with Rotation block the parallize one.

    Components:
    - Rotate: element-wise complex rotation
    - Recur: complex-valued linear recurrence
    - Skip: gated skip connection with SiLU activation
    - Norm: element-wise complex normalization (unit circle projection)
    
    Note: This is not extractly matched to vanilla-csp, it is a effective one with experiment intuition! 
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Rotation
        self.theta_proj = nn.Linear(2*hidden_dim, 2,bias=False)
        # Recurrence: input projection B (shared for real and imag)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim,bias=False)

        # Decay factor (alpha = exp(log_alpha), initialized to 1)
        self.delta_proj =  nn.Linear(2*hidden_dim, 1)
        self.gamma_proj = nn.Linear(2*hidden_dim, 1)

        # Skip gate (per-dimension, initialized to 0.5)
        self.skip_gate = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Initialization
        #nn.init.zeros_(self.theta_proj.weight)
        #nn.init.zeros_(self.theta_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, T, 2, H]  0: real, 1: imag
        Returns:
            x: [B, T, 2, H]
        """
        B, T, _, H = x.shape

        #1. Rotation with accumulatated angle.
        
        c  = torch.tanh(self.theta_proj(x.view(B,T,-1)))
        m_c = torch.sqrt(c[:, :, 0]**2 + c[:, :, 1]**2 + 1e-8).detach()
        c = c / m_c.unsqueeze(-1)  
        
        
        theta_all = torch.cumsum(torch.atan2(c[:,:,1],c[:,:,0]+torch.sign(c[:,:,0])*1e-9),dim=1).unsqueeze(-1) # [B, T, 1]
       

        cos_a, sin_a = torch.cos(theta_all), torch.sin(theta_all)  # [B, T, 1]

        # construct of rotation matrix
        Rot1 = torch.cat([cos_a, -sin_a], dim=-1)  # [B, T, 1, 2]
        Rot2 = torch.cat([sin_a, cos_a], dim=-1)   # [B, T, 1, 2]
        Rot = torch.stack([Rot1, Rot2], dim=-2)    # [B, T, 2, 2]

       
        x_rot = torch.einsum('btih, btji -> btjh', x, Rot)  
        
        
        #2. Element construct after rotation
        #gamma_all       =     (1 + torch.sin(self.gamma_proj(x.view(B,T,-1)))).unsqueeze(-1)/2 
        x_g             =     self.B_proj(x_rot)
        gamma_all       =     (1 + torch.sin(self.gamma_proj(x_g.view(B,T,-1)))).unsqueeze(-1)/2
        x_g = x_g*gamma_all
        
        
        #3. Mamba style recur
       
        #alpha_all       =    torch.exp(-delta_all).unsqueeze(-1)                # [B, T, 1] [B, T, 1]
        delta_all       =    F.softplus(self.delta_proj(x_g.view(B,T,-1)))
        #print(x_shifted.shape)
        alpha_all  = torch.exp(-delta_all).unsqueeze(-1)
        
        alpha_cumprod   =     torch.cumprod(alpha_all,dim=1) 
        alpha_cumshift =     torch.cat([torch.ones(B,1,1,1,device=x.device),alpha_cumprod[:,:-1,:,:]],dim=1)
        
        x_recur = alpha_cumprod*torch.cumsum(x_g / (alpha_cumshift+1e-9).detach(), dim=1)
        
        #x_recur = alpha_cumshift*torch.cumsum(x_g / (alpha_cumshift+1e-9).detach(), dim=1) 
        # may also established, and even better
       
        
        #4. Skip connection with elementwise magnitude normalization
        gate = torch.sigmoid(self.skip_gate) 
        x_out = x * gate + F.silu(x_recur)       
        magnitude = torch.sqrt(x_out[:, :, 0, :]**2 + x_out[:, :, 1, :]**2 + 1e-8).detach()
        x_out = x_out / magnitude.unsqueeze(-2)  

        return x_out




class CSP_Hidden_Vanilla(nn.Module):
    """
    Complex Propagator with Rotation block.

    Components:
    - Rotate: element-wise complex rotation
    - Recur: complex-valued linear recurrence
    - Skip: gated skip connection with SiLU activation
    - Norm: element-wise complex normalization (unit circle projection)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Rotation
        self.theta_proj = nn.Linear(2*hidden_dim, 2,bias=False)
        
        # Recurrence: input projection B (shared for real and imag)
        self.B_proj = nn.Linear(hidden_dim, hidden_dim,bias=False)

        # Decay factor (alpha = exp(log_alpha), initialized to 1)
        self.delta_proj =  nn.Linear(2*hidden_dim, 1)
        self.gamma_proj = nn.Linear(2*hidden_dim, 1)

        # Skip gate (per-dimension, initialized to 0.5)
        self.skip_gate = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Initialization
        #nn.init.zeros_(self.theta_proj.weight)
        #nn.init.zeros_(self.theta_proj.bias)

    def forward(self, x):
        """
        Args:
            x: [B, T,2, H]
        Returns:
            x: [B, T,2,H]
        """
        B, T,_, H = x.shape

        # Save input for skip connection
        h_real_input = x[:,:,0,:]
        h_imag_input = x[:,:,1,:]

        h_real_prev = torch.zeros(B, H, device=x.device)
        h_imag_prev = torch.zeros(B, H, device=x.device)
        theta_cum   = torch.zeros(B,1,device=x.device)
        
        #h_real_prev = h_real_input[:,-1,:]
        #h_imag_prev = h_imag_input[:,-1,:]
        
        outputs_real, outputs_imag = [], []

        
        for t in range(T):
            h_real_t = x[:, t, 0 , :]
            h_imag_t = x[:, t, 1 , :]

            # 1. Rotate: theta_t from input
            decision_linear_prev = torch.cat([h_real_prev,h_imag_prev],dim=-1)
            decision_linear_current = torch.cat([h_real_t,h_imag_t],dim=-1)
            c    = torch.tanh(self.theta_proj(decision_linear_current))
            m_c = torch.sqrt(c[:, 0]**2 + c[:, 1]**2 + 1e-8).detach()
            c = c / m_c.unsqueeze(-1)  
            
            theta_t = torch.atan2(c[:,1],c[:,0]+torch.sign(c[:,0])*1e-9).unsqueeze(-1) 
            theta_cum = theta_cum + theta_t
            cos_t, sin_t = torch.cos(theta_cum), torch.sin(theta_cum)

            h_real_rot = cos_t * h_real_t - sin_t * h_imag_t
            h_imag_rot = sin_t * h_real_t + cos_t * h_imag_t

            B_real_t = self.B_proj(h_real_rot)
            B_imag_t = self.B_proj(h_imag_rot)
            
            decision_after_rope = torch.cat([B_real_t,B_imag_t],dim=-1)
            # 2. Recur: alpha * accume_state + gamma * input_projection_after_rotation  
            delta_t = F.softplus(self.delta_proj(decision_after_rope))
           
            alpha = torch.exp(-delta_t)
            #alpha   = torch.tanh(self.delta_proj(decision_linear_prev+decision_linear_current)) 
              
            gamma_t = (1 + torch.sin(self.gamma_proj(decision_after_rope)))/2
            
            

            h_real_new = alpha*h_real_prev  + gamma_t*B_real_t
            h_imag_new = alpha*h_imag_prev  + gamma_t*B_imag_t

            outputs_real.append(h_real_new)
            outputs_imag.append(h_imag_new)

            h_real_prev = h_real_new
            h_imag_prev = h_imag_new

        # 3. Stack recurrence outputs
        h_real_recur = torch.stack(outputs_real, dim=1)  # [B, T, H]
        h_imag_recur = torch.stack(outputs_imag, dim=1)

        # 4. Skip connection: gated input + SiLU(recur)
        gate = torch.sigmoid(self.skip_gate)  # [H]
        h_real_skip = h_real_input * gate + F.silu(h_real_recur)
        h_imag_skip = h_imag_input * gate + F.silu(h_imag_recur)

        # 5. Complex normalization: project onto unit circle
        magnitude = torch.sqrt(h_real_skip**2 + h_imag_skip**2 + 1e-8)
        h_real_out = h_real_skip / magnitude
        h_imag_out = h_imag_skip / magnitude

        return torch.stack([h_real_out, h_imag_out], dim=2)


class CSP(BaseSeqModel):
    """
    Complete Complex State Propagator model.
    """
    
    def __init__(self, vocab_size=2, embed_dim=16 ,hidden_dim=64, output_dim=2, num_layers=3,rope_input=False,mode = "fast" ):
        super().__init__(vocab_size, embed_dim,hidden_dim, output_dim)
        self.rope_input = rope_input
        self.hidden_dim = hidden_dim

        #self.encoder = nn.Linear(1, hidden_dim)
        
        self.theta_proj = nn.Linear(2*hidden_dim,1)
        
        
        if mode=="fast": 
            hidden_class = CSP_Hidden_FAST
        elif mode == "vanilla":
            hidden_class = CSP_Hidden_Vanilla
        elif mode == "accuMod":
            hidden_class = CSP_Hidden_AccuMod
        else:
            hidden_class = CSP_Hidden_FAST
        
        self.layers = nn.ModuleList([
           hidden_class(hidden_dim) for _ in range(num_layers)
        ])
        
        
        self.decoder = nn.Linear(2*hidden_dim, output_dim)
        # Initialization
        nn.init.zeros_(self.theta_proj.weight)
        nn.init.zeros_(self.theta_proj.bias)

        

    def forward(self, x):
        """
        Args:
            x: [B, T, 1] binary sequence (0 or 1)
        Returns:
            logits: [B, output_dim]
        """
        B,T,_ = x.shape
              # [B, T, H]
        x = x.squeeze(-1).long()
             
        x = self._embed(x)
        
        x_real = x
        x_imag = torch.zeros_like(x)
        
        x = torch.stack([x_real,x_imag],dim=2)
        
        # experimental , never used!
        if self.rope_input:
            cum_x = self.theta_proj(torch.cumsum(x.view(B,T,-1),dim=1)/torch.arange(1,T+1,device=x.device).unsqueeze(-1).unsqueeze(0).detach())
       
            theta_all = torch.tanh(cum_x)* math.pi  # [B, T, 1]
       

            cos_a, sin_a = torch.cos(theta_all), torch.sin(theta_all)  # [B, T, 1]

        # construct of rotation matrix
            Rot1 = torch.cat([cos_a, -sin_a], dim=-1)  # [B, T, 1, 2]
            Rot2 = torch.cat([sin_a, cos_a], dim=-1)   # [B, T, 1, 2]
            Rot = torch.stack([Rot1, Rot2], dim=-2)    # [B, T, 2, 2]


            x = torch.einsum('btih, btji -> btjh', x, Rot)   
        
        
        
        for layer in self.layers:
            x = layer(x)  #normed before layer output
        
        # CSP now switch to transformer style sensing head! It is still valid and accurate!
        phase = torch.atan2(x[:,:,1,:], x[:,:,0,:]) 
        
        x_dec  =  torch.cat([torch.cos(phase),torch.sin(phase)],dim=-1) 
        
        return self.decoder(x_dec)



            
        
    
    

    
    
