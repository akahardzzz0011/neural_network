function [w1,w2,b1,b2] = kuvan_demo(w1,w2,b1,b2)

  data = load('mnist.mat');             
  images = double(data.testX);
  labels = double(data.testY);
  paikka = round(10000*rand(1));
  %paikka = [9906]; %9906
  oikein = 0;
  vaarin = 0;
   
  image = images(paikka,:);
  label = labels(paikka);
  imageToShow = reshape(image,28,28);
  
  y = zeros(10,1); 
  y(label + 1) = 1;
 
  figure(1);imshow(imageToShow'./255)
  
   a0 = image';
   z1 = w1*a0 + b1;
   a1 = sigmoid(z1);
       
   z2 = w2*a1 + b2;
   a2 = sigmoid(z2);
   
   figure(2);stem(0:9,a2,'-.','fill');xlabel('Outputs');
    
    vastaus = find(a2 == max(a2))-1;
    
    if (vastaus == labels(paikka))
      oikein = oikein +1;
      disp('Oikein!');
    else
      vaarin = vaarin +1;
      disp('Väärin!');
    end

      disp('Verkon vastaus:');
      disp(vastaus);
      disp('Todellinen luku');
      disp(labels(paikka));
      disp('Kuvan paikka');
      disp(paikka);
      a2
endfunction



function out = sigmoid(in)
  out = 1./(1+exp(-in));
endfunction