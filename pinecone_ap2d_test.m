function pinecone_ap2d_test

%close all;

opts.rng_seed=1;

%test='basic'; example='two_gaussians'; N=64; 
test='basic'; example='5'; N=64;
%test='basic'; example='4'; N=128;
%test='basic'; example='4.01'; N=64;
%test='basic'; example='tensor_product'; N=64;
%test='basic'; example='tensor_product_plus_square'; N=64;
%test='importance_of_oversampling'; example='4'; N=16;
%test='damping_needed_for_compute_time'; example='4'; N=32;

opts.noise=0;

opts.num_tries=6;
%opts.num_cycles=4;
opts.num_threads=12;
opts.tolerance=1e-8;
opts.oversamp=1.5;
opts.max_iterations=50000;
opts.alpha1=0.9;
opts.alpha2=0.95;
opts.beta=1.5;

opts.num_jobs=20;

if (strcmp(test,'basic'))
    opts.tolerance=1e-8;
    
    opts.num_threads=6;
    run_test(N,example,opts);
elseif (strcmp(test,'importance_of_oversampling'))
    opts.oversamp=1;
    opts.title='Without oversampling';
    run_test(N,example,opts);
    
    opts.oversamp=1.5;
    opts.title='With oversampling';
    run_test(N,example,opts);
elseif (strcmp(test,'damping_needed_for_compute_time'))
    N=10;
    opts.alpha1=0.9;
    opts.alpha2=0.95;
    opts.beta=1.5;
    opts.title='With damping';
    tic;
    run_test(N,example,opts);
    t1=toc;
    
    opts.alpha1=1;
    opts.alpha2=1;
    opts.beta=1.5;
    opts.title='Without damping';
    tic;
    run_test(N,example,opts);
    t2=toc;
    
    fprintf('Elapsed time with damping: %g s\n',t1);
    fprintf('Elapsed time without damping: %g s\n',t2);
end

end

function fB=run_test(N,example,opts)

% Fill in some default parameters
if (isfield(opts,'rng_seed')) rng(opts.rng_seed); end;
if (~isfield(opts,'noise')) opts.noise=0; end;
if (~isfield(opts,'title')) opts.title=''; end;

% Create the reference
[xx,yy]=ndgrid(linspace(-1,1,N),linspace(-1,1,N));
reference=create_example(example,xx,yy);

% Pad with zeros
Nfull=2*ceil(N*opts.oversamp/2); % make sure it's even!
Mfull=ceil((Nfull+1)/2);
M=ceil((N+1)/2);
reference_full=zeros(Nfull,Nfull);
reference_full(Mfull-M+1:Mfull-M+N,Mfull-M+1:Mfull-M+N)=reference;

% Add some noise
reference_full=reference_full+randn(size(reference_full))*opts.noise;

% Apodize?
reference_full=real(ifft2b(apodize(fft2b(reference_full),24,Nfull)));

% Set the support mask
use_support_constraint=0;
if (use_support_constraint)
    mask=zeros(Nfull,Nfull);
    mask(Mfull-M+1:Mfull-M+N,Mfull-M+1:Mfull-M+N)=1;
else
    mask=ones(Nfull,Nfull);
end;

% Set u
reference_full_hat=fft2b(reference_full);
u=abs(reference_full_hat);

% Set the options for pinecone_ap2d
opts.reference=reference_full;
opts.mask=mask;

% Show the reference in a figure
fA=figure;
imagesc(opts.reference); colormap('gray'); title('Reference');
drawnow;

opts0=opts;

fff1=figure('Name',opts.title,'NumberTitle','off');
plot(1:10); set(fff1,'position',[600,100,1500,600]);

fff2=figure('Name',opts.title,'NumberTitle','off');
plot(1:10); set(fff2,'position',[600,700,600,600]);

all_resid=[];
all_error=[];

opts0.init=(randn(size(u))+i*randn(size(u))).*u;
best_resid=inf;
best_error=inf;
best_f=zeros(size(u));
for j=1:10000
    opts0.init_stdevs=u*2;
    [f,resid,error,info]=pinecone_ap2d(u,opts0);
    all_resid=[all_resid,resid];
    all_error=[all_error,error];
    if (resid(1)<best_resid)    
        opts0.init=fft2b(f);
        opts0.init_stdevs=u*2;
        best_resid=resid(1);
        best_error=error(1);
        best_f=f;
    end;
    figure(fff1);
    plot(all_resid,all_error,'b.'); hold on;
    plot(resid,error,'r.','markersize',20); hold off;
    figure(fff2);
    imagesc(best_f); colormap('gray');
    title(sprintf('resid = %g, err = %g',best_resid,best_error));
    drawnow;
end;




end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function fB=run_test_old(N,example,opts)

if (isfield(opts,'rng_seed')) rng(opts.rng_seed); end;
if (~isfield(opts,'noise')) opts.noise=0; end;
if (~isfield(opts,'title')) opts.title=''; end;

[xx,yy]=ndgrid(linspace(-1,1,N),linspace(-1,1,N));
reference=create_example(example,xx,yy);

Nfull=2*ceil(N*opts.oversamp/2); % make sure it's even!
Mfull=ceil((Nfull+1)/2);
M=ceil((N+1)/2);
reference_full=zeros(Nfull,Nfull);
reference_full(Mfull-M+1:Mfull-M+N,Mfull-M+1:Mfull-M+N)=reference;
%mask=zeros(Nfull,Nfull);
%mask(Mfull-M+1:Mfull-M+N,Mfull-M+1:Mfull-M+N)=1;
mask=ones(Nfull,Nfull);
reference_full=reference_full+randn(size(reference_full))*opts.noise;
%reference_full=real(ifft2b(apodize(fft2b(reference_full),24,Nfull)));
%reference=reference_full(Mfull-M+1:Mfull-M+N,Mfull-M+1:Mfull-M+N);

reference_full_hat=fft2b(reference_full);
u=abs(reference_full_hat);

opts.reference=reference_full;
opts.mask=mask;

fA=figure;
imagesc(opts.reference); colormap('gray'); title('Reference');
drawnow;

opts0=opts;

fff=figure('Name',opts.title,'NumberTitle','off');
plot(1:10); set(fff,'position',[600,100,1500,600]);

all_resid=[];
all_error=[];

for j=1:opts.num_cycles
    u0=u;
    
    if (j==0)
        sigma=Nfull*0.05;
        NN=2*ceil(Nfull*0.5/2); %Make sure even
        u0=apodize(u,sigma,NN);
        opts0.reference=real(ifft2b(apodize(fft2b(opts.reference),sigma,NN)));
        opts0.mask=ones(size(u0));
        opts0.num_tries=opts.num_tries*2;
    else
        u0=u;
        opts0.reference=opts.reference;
        opts0.mask=opts.mask;
        opts0.num_tries=opts.num_tries;
    end
    if (j>1)
        opts0.init=put_in_center(opts0.init,size(u0));
        opts0.init_stdevs=put_in_center(opts0.init_stdevs,size(u0));
    end;
    
%     sigma=Nfull*j/opts.num_cycles*1.5;
%     u0=apodize(u,sigma,Nfull);
%     if (sigma>=Nfull) u0=u; end;
    [f,resid,error,info]=pinecone_ap2d(u0,opts0);
    f=mean(info.recon(:,:,1:10),3);

    figure(fff);
    subplot(2,ceil(opts.num_cycles*0.5),j);
    all_resid=[all_resid(:);resid(:)];
    all_error=[all_error(:);error(:)];
    plot(all_resid,all_error,'b.'); hold on;
    plot(resid,error,'r.'); hold off;
    title({sprintf('Cycle %d',j),sprintf('Best resid = %.2g',min(resid(:))),sprintf('Best err = %.3g',min(error(:)))});
    drawnow;

    %num=ceil(size(info.recon,3)*0.5);
    num=size(info.recon,3);
    recon0=info.recon(:,:,1:num);
    %stdevs=min(u,sqrt(var(fft2b(recon0),[],3)));
    %opts0.init_stdevs=stdevs;
    %opts0.init=fft2b(info.recon(:,:,1));
    opts0.init=fft2b(f);
    opts0.init_stdevs=u;
end;

display_results(u,opts,f,resid,error,info);

% if (isfield(info,'u')) u=info.u; end;
% if (isfield(info,'reference')) opts.reference=info.reference; end;

close(fA);

end

function display_results(u,opts,f,resid,error,info);

fB=figure;
subplot(2,2,1);
plot(resid,error,'b.'); xlabel('Resid'); ylabel('Error');
if (isfield(opts,'title')) title(opts.title); end;
subplot(2,2,2);
imagesc(opts.reference); colormap('gray'); title('Reference');
subplot(2,2,3);
recon_err=sqrt(sum((f(:)-opts.reference(:)).^2))/sqrt(sum((opts.reference(:)).^2));
imagesc(f); colormap('gray'); title(sprintf('Recon (err=%.3f)',recon_err));
ax=subplot(2,2,4); 
num=ceil(size(info.recon,3)*0.5);
stdevs=min(u,sqrt(var(fft2b(info.recon(:,:,1:num)),[],3)));
imagesc(stdevs./u); title('Variation in Fourier data'); colormap(ax,'parula'); colorbar;
set(fB,'position',[100,100,1000,1000]);

drawnow;

end

function Y=put_in_center(X,sz)
NX=size(X,1); MX=ceil((NX+1)/2);
NY=sz(1); MY=ceil((NY+1)/2);
if (NY>=NX)
    Y=zeros(sz);
    Y(MY-MX+1:MY-MX+NX,MY-MX+1:MY-MX+NX)=X;
else
    Y=X(MX-MY+1:MX-MY+NY,MX-MY+1:MX-MY+NY);
end;
end

function d0=apodize(d,sigma,N2)

N2=min(N2,size(d,1));

N=size(d,1);
M=ceil((N+1)/2);
aa=((1:N)-M); [GX,GY]=ndgrid(aa,aa);
GR=sqrt(GX.^2+GY.^2);
if (~isinf(sigma))
    d0=d.*exp(-GR.^2/sigma^2);
else
    d0=d;
end;

d0=d0(M-N2/2:M+N2/2-1,M-N2/2:M+N2/2-1);

d0=d0*(N2/N)^2; %Needed?

end

function X=create_example(example,xx,yy)

if (strcmp(example,'random'))
    X=rand(size(xx));
elseif (strcmp(example,'random_sparse'))
    X=zeros(size(xx));
    num=ceil(length(X(:))*0.2);
    X(randsample(length(X(:)),num))=rand(1,num);
elseif (strcmp(example,'single_gaussian'))
    X=exp(-(xx.^2+yy.^2)*3^2);
elseif (strcmp(example,'two_gaussians'))
    X=exp(-(xx.^2+yy.^2)*3^2);
    X=X+exp(-((xx-0.5).^2+(yy-0.4).^2)*4^2);
elseif (strcmp(example,'two_gaussians_plus_noise'))
    X=exp(-(xx.^2+yy.^2)*3^2);
    X=X+exp(-((xx-0.5).^2+(yy-0.4).^2)*4^2);
    X=X+randn(size(X))*0.05;
elseif (strcmp(example,'full_square'))
    X=1+rand(size(xx))*0.02;
elseif (strcmp(example,'tensor_product'))
    A=exp(-(xx-0.2).^2/0.1^2) + (abs(xx+0.3)<=0.2);
    B=exp(-(yy-0.2).^2/0.1^2) + (abs(yy+0.3)<=0.2);
    X=A.*B;
elseif (strcmp(example,'tensor_product_plus_square'))
    A=exp(-(xx-0.2).^2/0.1^2) + (abs(xx+0.3)<=0.2);
    B=exp(-(yy-0.2).^2/0.1^2) + (abs(yy+0.3)<=0.2);
    C=(abs(yy-0.3)<=0.1).*(abs(xx-0.4)<=0.1);
    X=A.*B+C;
elseif (strcmp(example,'4'))
    X=zeros(size(xx));
    for kk=1:20
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        X=X+exp(-((xx-cc(1)).^2+(yy-cc(2)).^2)/(rr/2)^2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    for kk=1:5
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        X=X+(abs(xx-cc(1))<=rr).*(abs(yy-cc(2))<=rr);
    end;
elseif (strcmp(example,'5'))
    X=zeros(size(xx));
    for kk=1:50
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        X=X+exp(-((xx-cc(1)).^2+(yy-cc(2)).^2)/(rr/2)^2).*((xx-cc(1)).^2+(yy-cc(2)).^2<=rr^2);
    end;
    for kk=1:5
        cc=(rand(2,1)*2-1)*0.7;
        rr=(rand*2-1)*0.2;
        X=X+(abs(xx-cc(1))<=rr).*(abs(yy-cc(2))<=rr);
    end;
end;

end

