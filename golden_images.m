function [w1,w2,b1,b2] = golden_images(w1,w2,b1,b2,iteration,epoch,eta)

  data = load('mnist.mat');
  images = double(data.trainX);
  labels = double(data.trainY);
  golden_kuva = [];
  tulos = [];
  verkon_vastaus = [];
  kuvan_label = [];
  images_laskuri = 0;
  delta_list = [];
  kytkin0 = 0;
  kytkin1 = 0;
  kytkin2 = 0;
  kytkin3 = 0;
  kytkin4 = 0;
  kytkin5 = 0;
  kytkin6 = 0;
  kytkin7 = 0;
  kytkin8 = 0;
  kytkin9 = 0;
  
  
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
    
    if (vastaus == label)
      oikein = oikein + 1;
 
    if (kytkin0 == 0)
      if (delta_L2 < 0.00001 && label == 0)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin0 = 1;
     endif
    endif
     if (kytkin1 == 0)
      if (delta_L2 < 0.000001 && label == 1)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin1 = 1;
      endif
     endif
     if (kytkin2 == 0)
      if (delta_L2 < 0.00001 && label == 2)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin2 = 1;
      endif
     endif
   
     if (kytkin3 == 0)
      if (delta_L2 < 0.001 && label == 3)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin3 = 1;
      endif
     endif
     
     if (kytkin4 == 0)
      if (delta_L2 < 0.00001 && label == 4)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin4 = 1;
      endif
     endif
     
     if (kytkin5 == 0)
      if (delta_L2 < 0.00001 && label == 5)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin5 = 1;
      endif
     endif
     
     if (kytkin6 == 0)
      if (delta_L2 < 0.00001 && label == 6)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin6 = 1;
      endif 
     endif
     
     if (kytkin7 == 0)
      if (delta_L2 < 0.00001 && label == 7)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin7 = 1;
      endif
     endif
     
     if (kytkin8 == 0)
      if (delta_L2 < 0.001 && label == 8)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin8 = 1;
      endif
     endif
     
     if (kytkin9 == 0)
      if (delta_L2 < 0.0001 && label == 9)
       images_laskuri = images_laskuri +1;
       golden_kuva = [golden_kuva paikka];
       tulos = [tulos a2];
       verkon_vastaus = [verkon_vastaus vastaus];
       kuvan_label = [kuvan_label label];
       delta_list = [delta_list delta_L2];
       kytkin9 = 1;
      endif
     endif   
       
    else
      vaarin = vaarin + 1; 
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
      
      disp('Golden images:');
      disp('Kuvien määrä');
      disp(images_laskuri);
      disp('---');

      
  for Q = 1:length(golden_kuva)
    
      golden = golden_kuva(Q);
      naytaKuva = reshape(images(golden,:),28,28);
      figure(1);imshow(naytaKuva' ./255);
      figure(2);stem(0:9,tulos(:,Q),'-.','fill');
      disp('Kuvan label');
      disp(kuvan_label(Q));
      disp('Verkon vastaus');
      disp(verkon_vastaus(Q));
      disp('Virhe');
      disp(delta_list(Q));
      pause();
      endfor
      
  
endfunction

function out = sigmoid(in)
  out = 1./(1+exp(-in));
endfunction