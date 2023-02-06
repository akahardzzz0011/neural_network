function y = calculateNNresult_bakkupp(imagePosition)

  data = load('mnist.mat');
  load param;                 % let's load pre-teached NN params.
  images = double(data.testX);
  labels = double(data.testY);
  
  oikein = 0;
  vaarin = 0;
  
 epoch = 50;
 steps = 8;
 learning_rate = 2.0;
 for K = 1:epoch
   paikka = round(10000*rand(1));
   if(paikka == 10000)
   paikka = 1;
   end
   
  y = zeros(1,10);  % let's initialize network result.
  
  
  imagePosition = [paikka];
  image = images(imagePosition,:)';  % one row of 784 pixels
  imageToShow = reshape(image,28,28);

 % figure(1);imshow(imageToShow'./255)
  
   a1 = image;
   z1 = w1*a1 + b1;
   a2 = sigmoid(z1);
       
   z2 = w2*a2 + b2;
   y = sigmoid(z2);
  % figure(2);stem(0:9,y,'-.','fill');xlabel('Outputs');
 %  disp(imageToShow);
    
    vastaus= find(y == max(y))-1;
    
    if (vastaus == labels(paikka))
      oikein = oikein +1;
    else
      vaarin = vaarin +1;
    end
    %  disp('Verkon vastaus:');
    %  disp(vastaus);
    %  disp('oikein menneet:');
    %  disp(oikein);
    %  disp('väärin menneet:')
    %  disp(vaarin);
    %  disp('Todellinen luku');
    %  disp(labels(paikka));
endfor

      disp('oikein menneet:');
      disp(oikein);
      disp('väärin menneet:')
      disp(vaarin);
endfunction



function out = sigmoid(in)
  out = 1./(1+exp(-in));
endfunction