function [w1,w2,b1,b2] = wrong_images_loopilla(w1,w2,b1,b2,iteration,epoch,eta)

  data = load('mnist.mat');
  images = double(data.trainX);
  labels = double(data.trainY);
  
  vaara_kuva = [];
  tulos = [];
  verkon_vastaus = [];
  kuvan_label = [];
  
  oikein = 0;
  vaarin = 0;
  paikka = round(60000*rand(1));
  
  for T = 1:iteration
    
    rw1 = zeros(30, 784);
    rw2 = zeros(10, 30);
    rb1 = zeros(30, 1);
    rb2 = zeros(10, 1);
    
 for K = 1:epoch
   
   if(paikka == 60000)
     paikka = 0;
   endif
 
  paikka = paikka + 1;
  image = images(paikka,:); 
  label = labels(paikka); 
  
  y = zeros(10,1); 
  y(label+1) = 1;
 
  
   a0 = image';
   z1 = w1*a0 + b1;
   a1 = sigmoid(z1);
       
   z2 = w2*a1 + b2;
   a2 = sigmoid(z2);
   
   dC = a2 - y;
   d_sig_z2 = sigmoid(z2) .* (1-sigmoid(z2));
   
   delta_L2 = dC .* d_sig_z2;
   grad_b2 = delta_L2;
   grad_w2 = delta_L2 * a1';
   
   d_sig_z1 = sigmoid(z1) .* (1-sigmoid(z1));
   
   delta_L1 = ((w2') * delta_L2) .* d_sig_z1;
   grad_b1 = delta_L1;
   grad_w1 = delta_L1 * a0';
   
   rw1 = rw1 + grad_w1;
   rw2 = rw2 + grad_w2;
   rb1 = rb1 + grad_b1;
   rb2 = rb2 + grad_b2;
   
    vastaus = find(a2 == max(a2))-1;
    
    if (vastaus != label)
      
       vaarin = vaarin + 1;
       vaara_kuva = [vaara_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       
    else
      oikein = oikein + 1; 
    endif
      
    prosentit_oikein = (oikein/(oikein+vaarin))*100;
    prosentit_vaarin = (vaarin/(oikein+vaarin))*100;
    
endfor
      w1 = w1 - (eta/epoch) * rw1;
      w2 = w2 - (eta/epoch) * rw2;
      b1 = b1 - (eta/epoch) * rb1;
      b2 = b2 - (eta/epoch) * rb2;  
  
endfor
  
      disp('Harjoituksen tulokset:');
      disp('Oikein lkm ja %');
      disp(oikein);
      disp(prosentit_oikein);
      disp('Väärin lkm ja %');
      disp(vaarin);
      disp(prosentit_vaarin);
      disp('Virheelliset luokittelut:');
  for Q = 1:length(vaara_kuva)
    
      vaara = vaara_kuva(Q);
      naytaKuva = reshape(images(vaara,:),28,28);
      figure(1);imshow(naytaKuva' ./255);
      figure(2);stem(0:9,tulos(:,Q),'-.','fill');
      disp('Kuvan label');
      disp(kuvan_label(Q));
      disp('Verkon vastaus');
      disp(verkon_vastaus(Q));
      pause();
      endfor
      
  
endfunction

function out = sigmoid(in)
  out = 1./(1+exp(-in));
endfunction
