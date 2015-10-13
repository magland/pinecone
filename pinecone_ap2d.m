function [f,resid,error,info]=pinecone_ap2d(u,opts)

if (nargin<1) pinecone_ap2d_test; return; end;

if (~isfield(opts,'tolerance')) opts.tolerance=1e-8; end;
if (~isfield(opts,'max_iterations')) opts.max_iterations=10000; end;
if (~isfield(opts,'num_tries')) opts.num_tries=100; end;
if (~isfield(opts,'ap2d_exe')) opts.ap2d_exe='/home/magland/dev/ap2d/ap2d'; end;
if (~isfield(opts,'num_threads')) opts.num_threads=2; end;
if (~isfield(opts,'alpha1')) opts.alpha1=0.9; end;
if (~isfield(opts,'alpha2')) opts.alpha2=0.95; end;
if (~isfield(opts,'beta')) opts.beta=1.5; end;
if (~isfield(opts,'mask')) opts.mask=ones(size(u)); end;

if (~isfield(opts,'num_jobs')) opts.num_jobs=1; end;

working_path=make_temporary_path;

writemda(u,[working_path,'/u.mda']);
writemda(opts.mask,[working_path,'/mask.mda']);
if (isfield(opts,'reference'))
    writemda(opts.reference,[working_path,'/reference.mda']);
end;
if (isfield(opts,'init'))
    writemda(real(opts.init),[working_path,'/init_re.mda']);
    writemda(imag(opts.init),[working_path,'/init_im.mda']);
    writemda(opts.init_stdevs,[working_path,'/init_stdevs.mda']);
end;

use_srun=1;
if (use_srun)
    opts.num_jobs=10;
    
    cmd=sprintf('/mnt/xfs1/home/magland/dev/ap2d/ap2d_batch.sh %d %d %s %s %s %g %d %d %s %s %s %s %s %g %g %g', ...
        opts.num_jobs,opts.num_threads,[working_path,'/u.mda'],[working_path,'/recon'],[working_path,'/residerr'],opts.tolerance,opts.max_iterations,opts.num_tries,...
        [working_path,'/reference.mda'],[working_path,'/mask.mda'],[working_path,'/init_re.mda'],[working_path,'/init_im.mda'],[working_path,'/init_stdevs.mda'],...
        opts.alpha1,opts.alpha2,opts.beta);
    disp(cmd);
    system(cmd);
    
    recon=zeros(size(u,1),size(u,2),0);
    residerr=zeros(0,2);
    for j=1:opts.num_jobs
        recon0=readmda([working_path,sprintf('/recon-%d.mda',j)]);
        recon(:,:,end+1:end+size(recon0,3))=recon0;
        residerr0=readmda([working_path,sprintf('/residerr-%d.mda',j)]);
        residerr(end+1:end+size(residerr0,1),:)=residerr0;
    end;
else
    cmd=sprintf('%s %s ',opts.ap2d_exe,[working_path,'/u.mda']);
    cmd=[cmd,sprintf('--out-recon=%s ',[working_path,'/recon.mda'])];
    cmd=[cmd,sprintf('--out-resid-err=%s ',[working_path,'/residerr.mda'])];
    cmd=[cmd,sprintf('--tol=%g ',opts.tolerance)];
    cmd=[cmd,sprintf('--maxit=%d ',opts.max_iterations)];
    cmd=[cmd,sprintf('--count=%d ',opts.num_tries)];
    if (isfield(opts,'reference'))
        cmd=[cmd,sprintf('--ref=%s ',[working_path,'/reference.mda'])];
    end;
    cmd=[cmd,sprintf('--num-threads=%d ',opts.num_threads)];
    cmd=[cmd,sprintf('--mask=%s ',[working_path,'/mask.mda'])];
    if (isfield(opts,'init'))
        cmd=[cmd,sprintf('--init-re=%s ',[working_path,'/init_re.mda'])];
        cmd=[cmd,sprintf('--init-im=%s ',[working_path,'/init_im.mda'])];
        cmd=[cmd,sprintf('--init-stdevs=%s ',[working_path,'/init_stdevs.mda'])];
    end;
    cmd=[cmd,sprintf('--alpha1=%g ',opts.alpha1)];
    cmd=[cmd,sprintf('--alpha2=%g ',opts.alpha2)];
    cmd=[cmd,sprintf('--beta=%g ',opts.beta)];
    disp(cmd);
    system(cmd);
    recon=readmda([working_path,'/recon.mda']);
    residerr=readmda([working_path,'/residerr.mda']);
end;

resid=residerr(:,1);
error=residerr(:,2);

[resid,sort_inds]=sort(resid);
error=error(sort_inds);
recon=recon(:,:,sort_inds);

f=recon(:,:,1);

info.recon=recon;

end

function ret=make_temporary_path
path1=[tempdir,'/pinecone'];
mkdir_jfm(path1);
num=1;
while 1
    tmp=[path1,sprintf('/%d',num)];
    if (exist(tmp)==0)
        mkdir(tmp);
        ret=tmp;
        return;
    end;
    num=num+1;
    if (num>10000)
        ret='<notfound>';
        return;
    end;
end
end

function mkdir_jfm(path)
if (exist(path)==7) return; end;
mkdir(path);
end

function A=readmda(fname)
F=fopen(fname,'rb');
code=fread(F,1,'long');
if (code>0) 
    num_dims=code;
    code=-1;
else
    fread(F,1,'long');
    num_dims=fread(F,1,'long');    
end;
S=zeros(1,num_dims);
for j=1:num_dims
    S(j)=fread(F,1,'long');
end;
N=prod(S);
A=zeros(S);
if (code==-1)
    M=zeros(1,N*2);
    M(:)=fread(F,N*2,'float');
    A(:)=M(1:2:prod(S)*2)+i*M(2:2:prod(S)*2);
elseif (code==-2)
    A(:)=fread(F,N,'uchar');
elseif (code==-3)
    A(:)=fread(F,N,'float');
elseif (code==-4)
    A(:)=fread(F,N,'short');
elseif (code==-5)
    A(:)=fread(F,N,'int');
end;
fclose(F);
end

function writemda(X,fname)
num_dims=2;
if (size(X,3)>1) num_dims=3; end;
if (size(X,4)>1) num_dims=4; end;
if (size(X,5)>1) num_dims=5; end;
if (size(X,6)>1) num_dims=6; end;
FF=fopen(fname,'w');
complex=1;
if (isreal(X)) complex=0; end;
if (complex)
    fwrite(FF,-1,'int32');
    fwrite(FF,8,'int32');
    fwrite(FF,num_dims,'int32');
    dimprod=1;
    for dd=1:num_dims
        fwrite(FF,size(X,dd),'int32');
        dimprod=dimprod*size(X,dd);
    end;
    XS=reshape(X,dimprod,1);
    Y=zeros(dimprod*2,1);
    Y(1:2:dimprod*2-1)=real(XS);
    Y(2:2:dimprod*2)=imag(XS);
    fwrite(FF,Y,'float32');
else
    fwrite(FF,-3,'int32');
    fwrite(FF,4,'int32');
    fwrite(FF,num_dims,'int32');
    dimprod=1;
    for dd=1:num_dims
        fwrite(FF,size(X,dd),'int32');
        dimprod=dimprod*size(X,dd);
    end;
    Y=reshape(X,dimprod,1);
    fwrite(FF,Y,'float32');
end;
fclose(FF);
end