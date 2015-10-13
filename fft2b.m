function Y=fft2b(X)
Y=fftshift(fft2(ifftshift(X)));
end