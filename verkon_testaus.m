function [w1,w2,b1,b2] = verkon_testaus(w1,w2,b1,b2,kierros_testing)

  data = load('mnist.mat');
  images = double(data.testX);
  labels = double(data.testY);

  oikein = 0;
  vaarin = 0;
  paikka = round(10000*rand(1));
 % paikka = [1];
  kaavio = 0;            %kytkin kaaviolle
  oikein_tulos = oikein;
  vaarin_tulos = vaarin;
 
 for K = 1:kierros_testing
   
  if(paikka == 10000)
      paikka = 0;
    endif
    
    paikka = paikka + 1;
    
   image = images(paikka,:);
   label = labels(paikka);
   y = zeros(10,1);
   y(label + 1) = 1;
 
   a0 = image';
   z1 = w1*a0 + b1;
   a1 = sigmoid(z1);
       
   z2 = w2*a1 + b2;
   a2 = sigmoid(z2);
    
    maksimi = max(a2);
    vastaus = find(a2 == maksimi)-1;
    

    
    if (vastaus == label)
      oikein = oikein +1;
    else
      vaarin = vaarin +1;
    endif
    
      prosentit_oikein = (oikein/(oikein+vaarin))*100;
      prosentit_vaarin = (vaarin/(oikein+vaarin))*100;
    
      oikein_tulos = [oikein_tulos oikein];
      vaarin_tulos = [vaarin_tulos vaarin];
endfor
      oikein
      prosentit_oikein
      vaarin
      prosentit_vaarin
      
     if (kaavio == 0)
      hold off
      figure(1);
      plot(oikein_tulos,'-',"color","blue")
      hold on
      plot(vaarin_tulos,'-',"color","red")
      title("Testin tulokset");
      legend({"Oikein", "Väärin"});      
      xlabel("Kuvien määrä");
      ylabel("Vastaukset");
     endif
      
      
endfunction
function out = sigmoid(in)
  out = 1./(1+exp(-in));
endfunction