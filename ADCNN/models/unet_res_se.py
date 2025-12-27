import torch, torch.nn as nn, torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self,c,r=8): super().__init__(); self.fc1=nn.Conv2d(c,c//r,1); self.fc2=nn.Conv2d(c//r,c,1)
    def forward(self,x): s=F.adaptive_avg_pool2d(x,1); s=F.silu(self.fc1(s),inplace=True); s=torch.sigmoid(self.fc2(s)); return x*s

def _norm(c, groups=8):
    g=min(groups,c) if c%groups==0 else 1
    return nn.GroupNorm(g,c)

class ResBlock(nn.Module):
    def __init__(self,c_in,c_out,k=3,act=nn.SiLU,se=True):
        super().__init__(); p=k//2
        self.proj = nn.Identity() if c_in==c_out else nn.Conv2d(c_in,c_out,1)
        self.bn1=_norm(c_in); self.c1=nn.Conv2d(c_in,c_out,k,padding=p,bias=False)
        self.bn2=_norm(c_out); self.c2=nn.Conv2d(c_out,c_out,k,padding=p,bias=False)
        self.act=act(); self.se=SEBlock(c_out) if se else nn.Identity()
    def forward(self,x):
        h=self.act(self.bn1(x)); h=self.c1(h)
        h=self.act(self.bn2(h)); h=self.c2(h)
        h=self.se(h); return h + self.proj(x)

class Down(nn.Module):
    def __init__(self,c_in,c_out): super().__init__(); self.pool=nn.MaxPool2d(2); self.rb=ResBlock(c_in,c_out)
    def forward(self,x): return self.rb(self.pool(x))

class Up(nn.Module):
    def __init__(self,c_in,c_skip,c_out): super().__init__(); self.up=nn.ConvTranspose2d(c_in,c_in,2,stride=2); self.rb1=ResBlock(c_in+c_skip,c_out); self.rb2=ResBlock(c_out,c_out)
    def forward(self,x,skip):
        x=self.up(x)
        dh=skip.size(-2)-x.size(-2); dw=skip.size(-1)-x.size(-1)
        if dh or dw: x=F.pad(x,(0,max(0,dw),0,max(0,dh)))
        x=torch.cat([x,skip],1); x=self.rb1(x); x=self.rb2(x); return x

class ASPP(nn.Module):
    def __init__(self,c,r=[1,6,12,18]):
        super().__init__()
        self.blocks=nn.ModuleList([nn.Sequential(nn.Conv2d(c,c//4,3,padding=d,dilation=d,bias=False), nn.BatchNorm2d(c//4), nn.SiLU(True)) for d in r])
        self.project=nn.Conv2d(c,c,1)
    def forward(self,x): return self.project(torch.cat([b(x) for b in self.blocks],1))

class UNetResSE(nn.Module):
    def __init__(self,in_ch=1,out_ch=1,widths=(32,64,128,256,512)):
        super().__init__(); w=widths
        self.stem=nn.Sequential(nn.Conv2d(in_ch,w[0],3,padding=1,bias=False), nn.BatchNorm2d(w[0]), nn.SiLU(True), ResBlock(w[0],w[0]))
        self.d1=Down(w[0],w[1]); self.d2=Down(w[1],w[2]); self.d3=Down(w[2],w[3]); self.d4=Down(w[3],w[4])
        self.u1=Up(w[4],w[3],w[3]); self.u2=Up(w[3],w[2],w[2]); self.u3=Up(w[2],w[1],w[1]); self.u4=Up(w[1],w[0],w[0])
        self.head=nn.Conv2d(w[0],out_ch,1)
    def forward(self,x):
        s0=self.stem(x); s1=self.d1(s0); s2=self.d2(s1); s3=self.d3(s2); b=self.d4(s3)
        x=self.u1(b,s3); x=self.u2(x,s2); x=self.u3(x,s1); x=self.u4(x,s0); return self.head(x)

class UNetResSEASPP(UNetResSE):
    def __init__(self,in_ch=1,out_ch=1,widths=(32,64,128,256,512)):
        super().__init__(in_ch,out_ch,widths); self.aspp=ASPP(widths[-1]); self.d4=Down(widths[3],widths[4])
    def forward(self,x):
        s0=self.stem(x); s1=self.d1(s0); s2=self.d2(s1); s3=self.d3(s2); b=self.d4(s3); b=self.aspp(b)
        x=self.u1(b,s3); x=self.u2(x,s2); x=self.u3(x,s1); x=self.u4(x,s0); return self.head(x)
